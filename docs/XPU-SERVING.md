# Serving on Intel XPU (Windows) — `add-xpu-support` branch

This branch adds Intel XPU support to kev:

- `--device xpu` everywhere (`default_device()` auto-detects it).
- **SDPA on XPU by default** (eager is ~2x slower on the same request).
- **Hybrid XPU serving on Windows**: kev's fused layers with the DeltaNet chunk on torch
  ops instead of flash-linear-attention's all-Triton chunk, because Triton 3.4's Intel
  backend lowers `tl.dot` to scalar FMA on Xe-LPG+ GPUs (zero DPAS in the emitted LLVM IR),
  making fla's chunk 30-340x slower than oneDNN GEMMs at the dot itself. The hybrid is
  **1.7-2.3x faster than the pure torch reference** on an Arc 130T (kev-0.8B, new and
  prefix-cached requests) with answers agreeing to bf16 noise (max |dp| ≈ 0.008).

## Prerequisites (Windows)

| Component | Why | How to get |
|---|---|---|
| Intel GPU | target | Arc dGPU or Core Ultra iGPU (Xe-LPG/LPG+ tested) |
| Windows + Python 3.10-3.13 | target | — |
| oneAPI Base Toolkit 2025.x | clang-cl + sycl8.dll for Triton | intel.com → oneAPI Base Toolkit |
| MSVC "Desktop development with C++" (VS 2022 or BuildTools) | C++ std lib + linker for Triton | Visual Studio Installer |
| Windows 10/11 SDK | ucrt/um/shared headers | ships with the workload above |
| torch (XPU build) + triton 3.4 | fla's elementwise kernels | `pip install torch --index-url https://download.pytorch.org/whl/xpu`, then `pip install triton==3.4.*` |

`kev.xpu_triton_env` locates all of the above automatically (oneAPI base dir + compiler
bin, MSVC via `vswhere`, newest Windows SDK) and prints what it found. To override a
probe, set the variable yourself before importing: `ONEAPI_ROOT`,
`LEVEL_ZERO_V1_SDK_PATH`, `CC`, `CXX`, `INCLUDE`, `LIB`.

## Setup

```bash
# in your venv
pip install -e ".[test]"
pip install "flash-linear-attention==0.5.2"
# or with uv:
uv sync --extra test
uv pip install "flash-linear-attention==0.5.2"
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
printed at startup:

```
xpu_triton_env: oneapi: ... | msvc: ... | winsdk: ... | sycl8: preloaded from ... | triton: 3.4.0 | patch: ... applied | patch: ... applied
```

## Switches

| Env var | Default | Meaning |
|---|---|---|
| `KEV_TORCH_DELTA` | `0` | `1` = serve the pure torch reference (fla blocked; simplest, most portable, slowest on XPU) |
| `KEV_TORCH_CHUNK` | `1` | `0` = put the DeltaNet chunk back on fla's Triton op (research only; slower on Arc) |
| `KEV_XPU_TRITON` | `1` | `0` = skip `xpu_triton_env` setup (only after rolling back the triton patches) |
| `KEV_FUSED` | auto | `0` / `1` = force fused layers off/on |
| `KEV_PREFIX_CACHE` | `4` | state-prefix entries kept across requests (0 disables) |

## Known issues / porting notes

- **Triton patches are version-guarded.** `xpu_triton_env` patches three spots in triton
  3.4's source (`build.py` flag splitting, `driver.py` `-fsycl` for every unit,
  `winmode=0` + sycl8 preload in the two DLL launchers). On a different triton version each
  patch reports `SKIPPED` in the startup banner — compilation will then likely fail, so pin
  triton 3.4 or expect to adapt the patterns.
- **`KEV_XPU_TRITON=0` after the patches have been applied crashes** the first DeltaNet
  forward (WinError 127). Reinstall triton (revert the site-packages edits) first.
- `KEV_TORCH_DELTA=1` blocks `fla` via `sys.modules` in `python -m kev.serve` only; library
  imports and tests are never affected.
- Linux + XPU and macOS are out of scope for this branch (the Windows-only path is tested).
- CUDA graph capture stays CUDA-only (`torch.xpu` has no capture API).

## Benchmarks

`python scripts/serving_bench.py --device xpu` (or `kev.serve` + the API). kev-0.8B on an
Arc 130T, median model time, new state / cached-state hit:

| request shape | pure torch reference | hybrid (default) |
|---|---|---|
| 89 tokens, 2 questions | 470 / 230 ms | 245 / 133 ms |
| 253 tokens, 6 questions | 1059 / 792 ms | 465 / 349 ms |
| 567 tokens, 5 questions | 1147 / 634 ms | 524 / 304 ms |
| 2392 tokens, 5 questions | 3654 / 685 ms | 1720 / 334 ms |

> AI生成
