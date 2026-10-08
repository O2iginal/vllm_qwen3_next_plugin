"""CPU contracts for all variants, without importing vLLM or initializing CUDA.

Execute the actual parameter-init statements / gate wrappers extracted by AST.
GPU math is tested separately in test_gdn_precision_cuda.py (explicit opt-in).
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / "vllm_qwen3_next_plugin"
VERSIONS = ("0_10_2", "0_11_0", "0_15_1", "0_18_1")


def source_tree(version):
    return ast.parse((ROOT / f"vllm_qwen3_next_{version}.py").read_text())


def execute(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "<plugin precision contract>", "exec"), namespace)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("model_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("tp", (1, 2, 4, 8))
def test_decay_parameters_preserve_fp32_checkpoint(version, model_dtype, tp):
    cls = next(
        n
        for n in source_tree(version).body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3NextGatedDeltaNet"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    nodes = [
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr in ("A_log", "dt_bias") for t in n.targets)
    ]
    assert len(nodes) == 2
    obj = SimpleNamespace(num_v_heads=32, tp_size=tp)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(model_dtype)
        execute(nodes, {"self": obj, "torch": torch, "nn": torch.nn, "divide": lambda n, d: n // d})
    finally:
        torch.set_default_dtype(old_dtype)
    checkpoint = torch.linspace(-2.7182818, 1.2345678, 32, dtype=torch.float32)
    for rank in range(tp):
        shard = checkpoint.chunk(tp)[rank]
        for param in (obj.A_log, obj.dt_bias):
            assert param.dtype == torch.float32
            with torch.no_grad():
                param.copy_(shard)  # Same destination-copy semantics as vLLM's loader.
            torch.testing.assert_close(param, shard, rtol=0, atol=0)


class CpuGateKernel:
    """Stand-in for a Triton launch; verify the wrapper's actual output buffers."""

    def __getitem__(self, grid):
        def launch(g, beta_out, A_log, a, b, dt_bias, *args, **kwargs):
            g.copy_(
                (
                    -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
                ).unsqueeze(0)
            )
            beta_out.copy_(b.float().sigmoid().unsqueeze(0))

        return launch


@pytest.mark.parametrize("version", ("0_15_1", "0_18_1"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_prefill_beta_buffer_does_not_round_through_model_dtype(version, dtype):
    fn = next(
        n
        for n in source_tree(version).body
        if isinstance(n, ast.FunctionDef) and n.name == "fused_gdn_gating"
    )
    ns = {
        "torch": torch,
        "triton": SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
        "fused_gdn_gating_kernel": CpuGateKernel(),
    }
    execute([fn], ns)
    b = torch.full((3, 8), 0.5, dtype=dtype)
    g, beta = ns[fn.name](torch.zeros(8), torch.zeros_like(b), b, torch.zeros(8))
    assert g.dtype == beta.dtype == torch.float32
    torch.testing.assert_close(beta.squeeze(0), b.float().sigmoid(), rtol=0, atol=0)
    assert not torch.equal(beta, b.sigmoid().float().unsqueeze(0))


@pytest.mark.parametrize("version", ("0_10_2", "0_11_0"))
def test_legacy_beta_is_computed_in_fp32(version):
    nodes = [
        n
        for n in ast.walk(source_tree(version))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "beta" for t in n.targets)
        and isinstance(n.value, ast.Call)
    ]
    assert len(nodes) == 1
    b = torch.tensor([0.5], dtype=torch.bfloat16)
    ns = {"b": b}
    execute(nodes, ns)
    assert ns["beta"].dtype == torch.float32
    torch.testing.assert_close(ns["beta"], b.float().sigmoid(), rtol=0, atol=0)


def test_packed_decode_promotes_only_raw_beta_gate():
    cls = next(
        n
        for n in source_tree("0_18_1").body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3NextGatedDeltaNet"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_forward_core_decode_non_spec"
    )
    captured = {}
    ns = {
        "torch": torch,
        "GDNAttentionMetadata": object,
        "causal_conv1d_update": lambda x, *args, **kwargs: x,
        "fused_recurrent_gated_delta_rule_packed_decode": lambda **kwargs: captured.update(kwargs),
    }
    execute([fn], ns)
    state = torch.zeros(2, 4, 2, 2)
    obj = SimpleNamespace(
        kv_cache=[(torch.zeros(2, 3, 4), state)],
        conv1d=SimpleNamespace(weight=torch.ones(4, 1, 4), bias=None),
        activation="silu",
        A_log=torch.zeros(4),
        dt_bias=torch.zeros(4),
        head_k_dim=2,
    )
    metadata = SimpleNamespace(non_spec_state_indices_tensor=torch.arange(2), num_actual_tokens=2)
    qkv = torch.zeros(2, 16, dtype=torch.bfloat16)
    b = torch.full((2, 4), 0.5, dtype=torch.bfloat16)
    a = torch.zeros_like(b)
    out = torch.zeros(2, 4, 2, dtype=torch.bfloat16)
    ns[fn.name](obj, qkv, b, a, out, metadata, 0)
    assert captured["b"].dtype == torch.float32
    assert captured["a"].dtype == captured["mixed_qkv"].dtype == torch.bfloat16
    assert captured["out"].data_ptr() == out.data_ptr()
    assert captured["initial_state"] is state
    # Emulate the 0.18.1 packed kernel's dtype round-trip on its actual argument.
    beta = captured["b"].float().sigmoid().to(captured["b"].dtype).float()
    torch.testing.assert_close(beta, b.float().sigmoid(), rtol=0, atol=0)
    assert not torch.equal(beta, b.sigmoid().float())


def test_no_cuda_was_initialized():
    assert not torch.cuda.is_initialized()
