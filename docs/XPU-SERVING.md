# Serving on Intel XPU (Windows) — `add-xpu-support` branch

This branch adds Intel XPU support to kev:

- `--device xpu` everywhere; kev.serve picks XPU by default on Windows when one is present (opt-in elsewhere).
- **SDPA on XPU by default** (eager is ~2x slower on the same request).
- **Hybrid XPU serving on Windows**: kev's fused layers with the DeltaNet chunk on torch
  ops instead of flash-linear-attention's all-Triton chunk, because Triton 3.4's Intel
  backend lowers `tl.dot` to scalar FMA on Xe-LPG+ GPUs (zero DPAS in the emitted LLVM IR),
  making fla's chunk 30-340x slower than oneDNN GEMMs at the dot itself. The hybrid is
  **1.7-2.3x faster than the pure torch reference** on an Arc 130T (kev-0.8B, new and
  prefix-cached requests) with answers agreeing to bf16 noise (max |dp| ≈ 0.008).

## Getting this branch

```bash
git clone --branch add-xpu-support https://github.com/guang384/kev.git
cd kev
git remote add upstream https://github.com/jaredpalmer/kev.git   # optional: pull upstream updates (fetch + rebase)
```

After the [Setup](#setup) below, check your toolchain in one command — every probe
(oneAPI, clang-cl, sycl8, MSVC, Windows SDK, Level Zero) prints what it found, and a
miss names the variable that overrides it:

```bash
python -m kev.xpu_triton_env
```

## Prerequisites (Windows)

| Component | Why | How to get |
|---|---|---|
| Intel GPU | target | Arc dGPU or Core Ultra iGPU (Xe-LPG/LPG+ tested) |
| Windows + Python 3.10-3.13 | target | — |
| oneAPI Base Toolkit 2025.x | clang-cl + sycl8.dll for Triton | intel.com → oneAPI Base Toolkit |
| MSVC "Desktop development with C++" (VS 2022 or BuildTools) | C++ std lib + linker for Triton | Visual Studio Installer |
| Windows 10/11 SDK | ucrt/um/shared headers | ships with the workload above |
| Intel triton wheel (`pytorch-triton-xpu` 3.4) | fla's elementwise kernels; ships the Windows glue natively | `pip install torch --index-url https://download.pytorch.org/whl/xpu`, then the matching `pytorch-triton-xpu` from the same index (stock `triton` is not supported) |

`kev.xpu_triton_env` locates all of the above automatically (oneAPI base dir + compiler
bin, MSVC via `vswhere`, newest Windows SDK) and prints what it found. To override a
probe, set the variable yourself before importing: `ONEAPI_ROOT`,
`LEVEL_ZERO_V1_SDK_PATH`, `CC`, `CXX`, `INCLUDE`, `LIB`.

## Setup

```bash
# in your venv
pip install -e ".[test]"
pip install "flash-linear-attention==0.5.2"
# XPU wheels (Windows): torch from the XPU index pulls the matching Intel Triton (pytorch-triton-xpu)
pip install torch --index-url https://download.pytorch.org/whl/xpu
# or with uv:
uv sync --extra test
uv pip install "flash-linear-attention==0.5.2"
uv pip install torch --index-url https://download.pytorch.org/whl/xpu
```

`flash-linear-attention==0.5.2` is pinned by `kev.fused_qwen35` (any other version raises
when `fused=True`). The pure-reference serve (`KEV_TORCH_DELTA=1`) works without fla.

## Run the server

```bash
export HF_HOME=/path/to/hf-cache    # a HF cache holding the base; otherwise the Hub is used
export HF_HUB_OFFLINE=1             # optional: skip all HEAD requests
python -m kev.serve --run /path/to/kev-0.8b --device xpu --port 8009
```

The first request pays a one-time Triton compile (seconds to minutes for the fused
variants); the on-disk cache makes the next ones instant. A toolchain probe summary is
printed at startup (a miss is a warning, never silent):

```
xpu_triton_env: oneapi: ... | clang-cl: ... | sycl8: preloaded | msvc: ... | winsdk: ... | levelzero: ...
```

## Switches

| Env var | Default | Meaning |
|---|---|---|
| `KEV_TORCH_DELTA` | `0` | `1` = serve the pure torch reference (fla blocked; simplest, most portable, slowest on XPU) |
| `KEV_XPU_TRITON` | `1` | `0` = skip `xpu_triton_env` setup entirely |
| `KEV_FUSED` | auto | `0` / `1` = force fused layers off/on |
| `KEV_PREFIX_CACHE` | `4` | state-prefix entries kept across requests (0 disables) |

## Footprint in the upstream tree

Everything ships as a new file or a guarded branch: on cuda/mps/cpu and on Linux/macOS
the behaviour is the upstream path (the full test suite gives identical results on this
branch and on main; the failures that exist on both are Windows-environment ones, in
upstream files this branch never touches).

| Where | Kind | What a reviewer sees |
|---|---|---|
| `kev/xpu_triton_env.py`, `tests/test_xpu.py`, `tests/conftest.py`, `docs/XPU-SERVING.md`, `.gitignore` (+`.temp/`) | new files | the toolchain setup, its tests (pure torch — CPU parity included), this doc, the scratch ignore |
| `kev/fused_qwen35.py` | guarded branch | `_chunk_gated_delta_rule_xpu` + an `if q.device.type == "xpu"` branch in `deltanet_forward`; every other device takes the upstream call unchanged |
| `kev/serve.py` | opt-in only | the sycl8 preload and the Windows-XPU default run only under `python -m kev.serve` on win32 with `KEV_XPU_TRITON != 0`; `--device` is a new optional flag; library imports and tests are never affected |
| `kev/device.py`, `kev/checkpoint.py`, `kev/model.py`, `kev/benchmark.py`, `kev/experiment.py`, `kev/train.py` | additive | `"xpu"` joins the existing device lists and guards; nothing upstream-existing changes behaviour |
| `kev/contrastive.py`, `kev/suite.py`, `kev/experiment.py`, `kev/train.py` | Windows fixes | `strftime("%-d")` → an f-string, `Path.as_posix`, optional `fcntl`/`resource`: identical output on POSIX, required for Windows to run at all |

The chunk op copies `initial_state` on entry before its in-place update, so a caller's
recurrent state (the serve prefix cache reuses them across passes) is never modified.

## Known issues / porting notes

- **The Intel triton wheel is required.** `pytorch-triton-xpu` ships the Windows glue
  natively (sycl8 preload, `winmode=0` DLL loading, the MSVC compiler/linker flag split in
  `build.py`), so `xpu_triton_env` never patches site-packages. Stock `triton` from PyPI is
  not supported by this branch.
- `KEV_TORCH_DELTA=1` blocks `fla` via `sys.modules` in `python -m kev.serve` only; library
  imports and tests are never affected.
- Linux + XPU and macOS are out of scope for this branch (the Windows-only path is tested).
- CUDA graph capture stays CUDA-only (`torch.xpu` has no capture API).

- **sycl8.dll and the Entry-Point dialog.** `import torch` loads the `intel-sycl-rt` wheel's sycl8.dll (2025.1, under `Library\bin`),
  which lacks symbols the oneAPI-linked `__triton_launcher.pyd` imports (e.g.
  `sycl::handler::setNDRangeDescriptor`); the first Triton XPU JIT in such a process dies with
  0xc0000139 and Windows shows a modal Entry-Point dialog that hangs it. `kev.xpu_triton_env` preloads
  the oneAPI toolkit sycl8.dll *before torch* — `python -m kev.serve` and `tests/conftest.py` do this,
  and any script of yours must too — and keeps preloaded processes' triton cache under
  `~/.triton/cache-xpu-oneapi` so their launchers never leak into cold ones. Calling `setup()` too
  late (torch already imported) is detected and skipped, leaving that process on the wheel's sycl8 end
  to end. `default_device()` therefore stays cuda→mps→cpu: XPU is opt-in (serve picks it on Windows,
  the CLIs take `--device xpu`), keeping library tests and training on CPU as upstream.

## Benchmarks

Served model time per request — the `latency_ms` every kev.serve call reports — median over 5
repeats, a new state per request / a repeated state (prefix-cache hit). kev-0.8B on an Arc 130T:

| request shape | pure torch reference (`KEV_TORCH_DELTA=1`) | hybrid (default) |
|---|---|---|
| 89 tokens, 2 questions | 470 / 230 ms | 245 / 133 ms |
| 253 tokens, 6 questions | 1059 / 792 ms | 465 / 349 ms |
| 567 tokens, 5 questions | 1147 / 634 ms | 524 / 304 ms |
| 2392 tokens, 5 questions | 3654 / 685 ms | 1720 / 334 ms |

To reproduce, serve each configuration and send a state followed by repeats of the same state
(the repeats exercise the prefix cache and report the cached latency):

```bash
python -m kev.serve --run /path/to/kev-0.8b --device xpu --port 8009                       # hybrid
KEV_TORCH_DELTA=1 python -m kev.serve --run /path/to/kev-0.8b --device xpu --port 8009    # reference
```

> AI生成