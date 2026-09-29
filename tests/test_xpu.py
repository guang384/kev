"""The XPU DeltaNet chunk op (kev.fused_qwen35._chunk_gated_delta_rule_xpu) against transformers'
torch reference: parity on random and coherent (near-identical keys) states, state continuation, and
the regression that killed the first version — the UT solve must not go NaN on multi-chunk coherent
inputs (the explicit 64x64 inverse does; the fp32 blocked substitution does not).

The op is a pure torch implementation, so these tests run on any backend (CPU is fine and fast);
XPU is exercised too when available. No weights, no server.
Run: uv run --extra serve python -m pytest tests/test_xpu.py -q
"""
import sys

import pytest
import torch

# transformers' hub wrapper prefers the installed fla for torch_chunk_gated_delta_rule (import time
# resolution); the pure-torch reference we must match is only reached when fla is absent. Block fla for
# this import, exactly as kev.serve's KEV_TORCH_DELTA=1 does, then restore it so kev.fused_qwen35
# (which imports fla's elementwise kernels) loads normally.
for _name in tuple(sys.modules):
    if _name == "fla" or _name.startswith("fla."):
        del sys.modules[_name]
sys.modules["fla"] = None
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
for _name in tuple(sys.modules):
    if _name == "fla" or _name.startswith("fla."):
        del sys.modules[_name]

from kev.fused_qwen35 import _chunk_gated_delta_rule_xpu

DEVICES = ["cpu"] + (["xpu"] if getattr(torch, "xpu", None) and torch.xpu.is_available() else [])
CHUNK = 64
B, H, DK, DV = 1, 2, 32, 32   # DK 32 ~ real head dim (128), but fast; random 32-d keys are near-orthogonal like the real ones


def _inputs(device, T, coherent=False, seed=0):
    """bf16 q/k/v [B, T, H, D] with fp32 log-decay and beta in (0, 1), in the real model's regime:
    weak decay (per-token ~0.95-1.0, like trained A_log/dt_bias) and Dk=32 keys.
    coherent=True makes the keys nearly identical (inner product ~0.999, the measured layer-8 regime:
    |L| near 1 and row sums >> 1, where an explicit inverse overflows into NaN)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, DK, generator=g)
    k = torch.randn(B, T, H, DK, generator=g)
    if coherent:
        q[..., 0] = 1.0
        k = torch.ones_like(k) + 0.1 * torch.randn_like(k)   # l2-normalized similarity ~0.997
    v = torch.randn(B, T, H, DV, generator=g)
    decay = -torch.rand(B, T, H, generator=g) * (0.05 if not coherent else 0.002)
    beta = torch.rand(B, T, H, generator=g) if not coherent else (0.9 + 0.1 * torch.rand(B, T, H, generator=g))
    to_d = lambda x, dtype: x.to(device).to(dtype)
    return to_d(q, torch.bfloat16), to_d(k, torch.bfloat16), to_d(v, torch.bfloat16), \
        to_d(decay, torch.float32), to_d(beta, torch.float32)


def _run(device, T, coherent=False, seed=0):
    """(our out, our state, reference out, reference state) for one pass on bf16 inputs."""
    q, k, v, g, beta = _inputs(device, T, coherent=coherent, seed=seed)
    ref = torch_chunk_gated_delta_rule(q.float(), k.float(), v.float(), g, beta,
                                       chunk_size=CHUNK, output_final_state=True,
                                       use_qk_l2norm_in_kernel=True)   # kev.fused_qwen35's serving call does the same
    ours = _chunk_gated_delta_rule_xpu(q, k, v, g, beta, chunk_size=CHUNK, output_final_state=True)
    return *ours, *ref


def _relerr(a, b):
    return (a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("T", [1, 63, 64, 65, 192])
def test_matches_reference(device, T):
    """The solve is exact (blocked substitution); bf16 GEMM rounding (fp32 accumulation in oneDNN) is
    the only noise, so 5e-2 is a comfortable bound for the bf16 serving domain vs the fp32 reference."""
    out, S, ref_out, ref = _run(device, T)
    assert out.shape == ref_out.shape and S.shape == ref.shape
    assert torch.isfinite(out).all() and torch.isfinite(S).all()
    assert _relerr(out, ref_out) < 5e-2 and _relerr(S, ref) < 5e-2


@pytest.mark.parametrize("device", DEVICES)
def test_state_continuation_matches_single_pass(device):
    """A state filled by pass 1 and read by pass 2 equals one pass over the concatenation."""
    T1, T2, T = 65, 127, 192
    q, k, v, g, beta = _inputs(device, T, seed=7)
    _, S_full, ref_out, _ = _run(device, T, seed=7)
    cut = lambda x: x[:, :T1]     # [B, T1, ...] slices the time axis for 4D q/k/v and 3D g/beta alike
    cut2 = lambda x: x[:, T1:]
    _, S1 = _chunk_gated_delta_rule_xpu(*map(cut, (q, k, v, g, beta)), chunk_size=CHUNK, output_final_state=True)
    out2, S2 = _chunk_gated_delta_rule_xpu(*map(cut2, (q, k, v, g, beta)), chunk_size=CHUNK,
                                           initial_state=S1, output_final_state=True)
    assert torch.isfinite(out2).all() and torch.isfinite(S2).all()
    assert _relerr(S2, S_full) < 5e-2 and _relerr(out2, ref_out[:, T1:]) < 5e-2


@pytest.mark.parametrize("device", DEVICES)
def test_no_nan_on_coherent_long_state(device):
    """Regression: the first solve (bf16 Newton-Schulz explicit inverse) returned x ~ 1e13 on coherent
    keys at depth (reference: 6.2), poisoning the recurrent state to NaN within a few chunks. The
    blocked substitution must stay finite on coherent keys (the oneDNN GEMMs on near-1 values do not
    get to test its precision — the reference is used only to bound the scale)."""
    out, S, ref_out, ref = _run(device, 370, coherent=True)
    assert torch.isfinite(out).all() and torch.isfinite(S).all()
    assert torch.isfinite(ref_out).all() and torch.isfinite(ref).all()
    assert _relerr(S, ref) < 1.0


def test_no_output_state_when_not_requested():
    args = _inputs("cpu", 70)
    _, S = _chunk_gated_delta_rule_xpu(*args, chunk_size=CHUNK, output_final_state=False)
    assert S is None