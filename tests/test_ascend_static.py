import re
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
QWEN3_NEXT_011 = (
    PLUGIN_ROOT / "vllm_qwen3_next_plugin" / "variants" / "vllm_0_11_0.py"
)
PLUGIN_INIT = PLUGIN_ROOT / "vllm_qwen3_next_plugin" / "__init__.py"
ASCEND_COMPAT = PLUGIN_ROOT / "vllm_qwen3_next_plugin" / "compat" / "ascend.py"


def test_011_gdn_prefill_uses_ascend_safe_varlen_conv_call() -> None:
    source = QWEN3_NEXT_011.read_text(encoding="utf-8")

    assert "metadata=attn_metadata" not in source
    assert "cache_indices=non_spec_state_indices_tensor" in source
    assert "query_start_loc=non_spec_query_start_loc" in source


def test_011_gdn_counts_actual_tokens_from_ascend_metadata_fields() -> None:
    source = QWEN3_NEXT_011.read_text(encoding="utf-8")

    assert re.search(
        r"num_actual_tokens\s*=\s*\(\s*"
        r"attn_metadata\.num_prefill_tokens\s*\+\s*"
        r"attn_metadata\.num_decode_tokens\s*\+\s*"
        r"attn_metadata\.num_spec_decode_tokens\s*\)",
        source,
    )
    assert "num_actual_tokens = attn_metadata.num_actual_tokens" not in source


def test_plugin_init_keeps_model_import_lazy_for_vllm_discovery() -> None:
    source = PLUGIN_INIT.read_text(encoding="utf-8")

    assert "def __getattr__(" in source
    assert "from .vllm_qwen3_next_0_11_0 import Qwen3NextForCausalLM" not in source


def test_plugin_init_applies_ascend_gdn_aclgraph_patch() -> None:
    source = PLUGIN_INIT.read_text(encoding="utf-8")

    assert "patch_gdn_aclgraph_support()" in source
    assert "patch_plugin_causal_conv1d_ops(Qwen3NextForCausalLM)" in source
    assert "patch_plugin_fla_ops(Qwen3NextForCausalLM)" in source
    assert "VLLM_PLUGINS=ascend,register_qwen3_next_model" in source


def test_ascend_compat_patches_gdn_aclgraph_support() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")

    assert "def patch_gdn_aclgraph_support()" in source
    assert "GDNAttentionMetadataBuilder.aclgraph_support" in source
    assert "AttentionCGSupport.UNIFORM_BATCH" in source


def test_ascend_compat_rebinds_plugin_causal_conv1d_ops() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")

    assert "def patch_plugin_causal_conv1d_ops(" in source
    assert "from vllm_ascend.ops.casual_conv1d import" in source
    assert "causal_conv1d_fn = _torch_causal_conv1d_fn" in source
    assert "causal_conv1d_update_npu = _torch_causal_conv1d_update" in source
    assert "variant_module.causal_conv1d_fn = causal_conv1d_fn" in source
    assert "variant_module.causal_conv1d_update = causal_conv1d_update_npu" in source


def test_ascend_compat_rebinds_plugin_fla_chunk_op() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")

    assert "def patch_plugin_fla_ops(" in source
    assert "from vllm_ascend.ops.fla import torch_chunk_gated_delta_rule" in source
    assert "def _wrap_ascend_chunk_gated_delta_rule(" in source
    assert "class _TorchLayerNormFn" in source
    assert "layernorm_guard.LayerNormFn = _TorchLayerNormFn" in source
    assert "variant_module.chunk_gated_delta_rule" in source
    assert "cu_seqlens" in source


def test_ascend_fla_chunk_wrapper_splits_varlen_prefill() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import (
        _wrap_ascend_chunk_gated_delta_rule,
    )

    calls: list[tuple[int, int]] = []

    def fake_chunk(query, key, value, g, beta, initial_state, output_final_state,
                   use_qk_l2norm_in_kernel):
        del key, g, beta, output_final_state, use_qk_l2norm_in_kernel
        calls.append((query.shape[1], initial_state.shape[0]))
        return value + 1, initial_state + len(calls)

    wrapped = _wrap_ascend_chunk_gated_delta_rule(fake_chunk)
    query = torch.zeros((1, 5, 2, 3))
    value = torch.zeros((1, 5, 2, 4))
    initial_state = torch.zeros((2, 2, 3, 4))

    out, state = wrapped(
        q=query,
        k=query,
        v=value,
        g=torch.zeros((1, 5, 2)),
        beta=torch.zeros((1, 5, 2)),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 2, 5]),
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert calls == [(2, 1), (3, 1)]
    assert out.shape == value.shape
    assert state.shape == initial_state.shape
    assert torch.all(out == 1)
    assert torch.all(state[0] == 1)
    assert torch.all(state[1] == 2)


def test_ascend_grouped_head_fla_fallback_expands_key_heads() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import _torch_chunk_gated_delta_rule

    out, state = _torch_chunk_gated_delta_rule(
        query=torch.randn((1, 3, 2, 4)),
        key=torch.randn((1, 3, 2, 4)),
        value=torch.randn((1, 3, 4, 5)),
        g=torch.randn((1, 3, 4)),
        beta=torch.sigmoid(torch.randn((1, 3, 4))),
        initial_state=torch.zeros((1, 4, 4, 5)),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == (1, 3, 4, 5)
    assert state.shape == (1, 4, 4, 5)


def test_ascend_torch_layernorm_fallback_applies_rms_gate() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import _TorchLayerNormFn

    x = torch.randn((3, 5), dtype=torch.float32)
    z = torch.randn((3, 5), dtype=torch.float32)
    weight = torch.ones((5,), dtype=torch.float32)

    out = _TorchLayerNormFn.apply(x, weight, None, z, 1e-6, None, True, True)
    expected_norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected_norm * torch.nn.functional.silu(z)

    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_ascend_recurrent_fallback_updates_selected_state() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import (
        _torch_fused_recurrent_gated_delta_rule,
    )

    state = torch.zeros((3, 4, 2, 5), dtype=torch.float32)
    out, final_state = _torch_fused_recurrent_gated_delta_rule(
        q=torch.randn((1, 1, 2, 2)),
        k=torch.randn((1, 1, 2, 2)),
        v=torch.randn((1, 1, 4, 5)),
        g=torch.zeros((1, 1, 4)),
        beta=torch.ones((1, 1, 4)),
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 1]),
        ssm_state_indices=torch.tensor([2]),
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == (1, 1, 4, 5)
    assert final_state is state
    assert torch.all(final_state[0] == 0)
    assert torch.all(final_state[1] == 0)
    assert torch.any(final_state[2] != 0)


def test_ascend_recurrent_fallback_uses_2d_spec_state_indices() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import (
        _torch_fused_recurrent_gated_delta_rule,
    )

    state = torch.zeros((5, 2, 2, 3), dtype=torch.float32)
    out, final_state = _torch_fused_recurrent_gated_delta_rule(
        q=torch.randn((1, 2, 2, 2)),
        k=torch.randn((1, 2, 2, 2)),
        v=torch.randn((1, 2, 2, 3)),
        g=torch.zeros((1, 2, 2)),
        beta=torch.ones((1, 2, 2)),
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 2]),
        ssm_state_indices=torch.tensor([[3, 4]]),
        num_accepted_tokens=torch.tensor([2]),
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == (1, 2, 2, 3)
    assert final_state is state
    assert torch.all(final_state[:3] == 0)
    assert torch.any(final_state[3] != 0)
    assert torch.any(final_state[4] != 0)


def test_ascend_prefill_conv_fallback_preserves_pad_slot_shape() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import (
        PAD_SLOT_ID,
        _torch_causal_conv1d_fn,
    )

    x = torch.randn((2, 5))
    out = _torch_causal_conv1d_fn(
        x=x,
        weight=torch.randn((2, 3)),
        bias=None,
        query_start_loc=torch.tensor([0, 2, 5]),
        cache_indices=torch.tensor([0, PAD_SLOT_ID]),
        has_initial_state=torch.tensor([False, False]),
        conv_states=torch.zeros((1, 2, 2)),
        activation=None,
    )

    assert out.shape == x.shape
    assert torch.all(out[:, 2:] == 0)


def test_ascend_causal_conv_patch_does_not_rebind_without_ascend(monkeypatch) -> None:
    import builtins
    import sys
    import types

    from vllm_qwen3_next_plugin.compat.ascend import patch_plugin_causal_conv1d_ops

    module_name = "test_variant_without_ascend"
    variant = types.ModuleType(module_name)
    original_fn = object()
    original_update = object()
    variant.causal_conv1d_fn = original_fn
    variant.causal_conv1d_update = original_update
    monkeypatch.setitem(sys.modules, module_name, variant)

    class FakeModel:
        __module__ = module_name

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert patch_plugin_causal_conv1d_ops(FakeModel) is False
    assert variant.causal_conv1d_fn is original_fn
    assert variant.causal_conv1d_update is original_update


def test_ascend_compat_patches_triton_python_utility_symbols() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")

    assert "next_power_of_2" in source
    assert "cdiv" in source
    assert "from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton" in source


def test_011_gdn_gating_has_torch_fallback_when_triton_exp_is_unavailable() -> None:
    source = QWEN3_NEXT_011.read_text(encoding="utf-8")

    assert "from vllm.triton_utils import HAS_TRITON, tl, triton" in source
    assert "if HAS_TRITON:" in source
    assert "F.softplus" in source


def test_011_dense_mlp_uses_layer_prefix_for_ascend_static_context() -> None:
    source = QWEN3_NEXT_011.read_text(encoding="utf-8")

    assert re.search(
        r"self\.mlp\s*=\s*Qwen3NextMLP\(\s*"
        r"hidden_size=config\.hidden_size,\s*"
        r"intermediate_size=config\.intermediate_size,\s*"
        r"hidden_act=config\.hidden_act,\s*"
        r"quant_config=quant_config,\s*"
        r"prefix=f\"\{prefix\}\.mlp\",\s*"
        r"\)",
        source,
    )
