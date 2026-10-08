"""Explicitly opt-in H20/GPU kernel tests; no CUDA access during default collection.

RUN_GDN_CUDA_TESTS=1 python -m pytest tests/test_gdn_precision_cuda.py -q
Requires vLLM 0.18.1 and this checkout installed/importable. Not a full-model test.
"""

import importlib
import os
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GDN_CUDA_TESTS") != "1",
    reason="GPU execution requires explicit RUN_GDN_CUDA_TESTS=1",
)


@pytest.fixture(scope="module")
def plugin():
    import vllm

    assert vllm.__version__.split("+")[0] == "0.18.1"
    module = importlib.import_module("vllm_qwen3_next_plugin.vllm_qwen3_next_0_18_1")
    expected = Path(__file__).resolve().parents[1] / "vllm_qwen3_next_plugin"
    assert Path(module.__file__).resolve().parent == expected, "Wrong plugin checkout loaded"
    return module


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("tokens", (1, 17, 65))
def test_prefill_gate_matches_fp32_reference(plugin, dtype, tokens):
    torch.manual_seed(123)
    a = torch.randn(tokens, 32, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.linspace(-3.1, 1.2, 32, device="cuda", dtype=torch.float32)
    dt_bias = torch.linspace(-4.2, -1.1, 32, device="cuda", dtype=torch.float32)
    g, beta = plugin.fused_gdn_gating(A_log, a, b, dt_bias)
    assert g.dtype == beta.dtype == torch.float32
    expected_g = -A_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    torch.testing.assert_close(g.squeeze(0), expected_g, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(beta.squeeze(0), b.float().sigmoid(), atol=2e-7, rtol=2e-6)


def test_packed_decode_keeps_beta_in_fp32(plugin):
    # With zero initial state and one-hot K, the updated state's first column
    # is exactly sigmoid(0.5); a BF16 round-trip is directly observable.
    heads, value_heads, dim = 1, 4, 64
    q = torch.zeros(1, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    q[..., 0] = k[..., 0] = 1
    v = torch.ones(1, value_heads, dim, device="cuda", dtype=torch.bfloat16)
    mixed = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=-1)
    a = torch.zeros(1, value_heads, device="cuda", dtype=torch.bfloat16)
    b = torch.full_like(a, 0.5)
    # Use slot 1: compatible with kernels reserving slot 0 for CUDA-graph padding.
    state = torch.zeros(2, value_heads, dim, dim, device="cuda", dtype=torch.float32)
    out = torch.empty(1, 1, value_heads, dim, device="cuda", dtype=torch.bfloat16)
    plugin.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=a,
        b=b.float(),
        A_log=torch.zeros(value_heads, device="cuda", dtype=torch.float32),
        dt_bias=torch.zeros(value_heads, device="cuda", dtype=torch.float32),
        scale=dim**-0.5,
        initial_state=state,
        out=out,
        ssm_state_indices=torch.ones(1, device="cuda", dtype=torch.int32),
        use_qk_l2norm_in_kernel=False,
    )
    expected = torch.full_like(state[1, :, :, 0], 0.5).sigmoid()
    torch.testing.assert_close(state[1, :, :, 0], expected, atol=2e-7, rtol=2e-6)
    assert torch.count_nonzero(state[1, :, :, 1:]) == 0
    assert torch.isfinite(out).all()
    assert not torch.equal(state[1, :, :, 0], expected.bfloat16().float())
