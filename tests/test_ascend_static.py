import re
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
QWEN3_NEXT_011 = (
    PLUGIN_ROOT / "vllm_qwen3_next_plugin" / "variants" / "vllm_0_11_0.py"
)
PLUGIN_INIT = PLUGIN_ROOT / "vllm_qwen3_next_plugin" / "__init__.py"
ASCEND_COMPAT = PLUGIN_ROOT / "vllm_qwen3_next_plugin" / "compat" / "ascend.py"
QWEN3_NEXT_0202 = (
    PLUGIN_ROOT / "vllm_qwen3_next_plugin" / "variants" / "vllm_0_20_2.py"
)


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


def test_011_gdn_state_debug_is_opt_in_and_prefix_filterable() -> None:
    source = QWEN3_NEXT_011.read_text(encoding="utf-8")

    assert "VLLM_QWEN3_NEXT_DEBUG_GDN_STATE" in source
    assert "VLLM_QWEN3_NEXT_DEBUG_GDN_STATE_PREFIX" in source
    assert source.count("_maybe_log_gdn_state(") >= 4
    assert '"before_conv"' in source
    assert '"after_conv"' in source
    assert '"before_recurrent"' in source
    assert '"after_recurrent"' in source


def test_011_gdn_decode_can_force_chunk_recurrent_path() -> None:
    source = QWEN3_NEXT_011.read_text(encoding="utf-8")

    assert "VLLM_QWEN3_NEXT_FORCE_CHUNK_GDN_DECODE" in source
    assert "decode_initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()" in source
    assert "ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to" in source
    assert re.search(
        r"elif attn_metadata\.num_decodes > 0:.*?"
        r"chunk_gated_delta_rule\(",
        source,
        re.DOTALL,
    )


def test_011_decoder_layer_state_debug_is_opt_in_and_layer_filterable() -> None:
    source = QWEN3_NEXT_011.read_text(encoding="utf-8")

    assert "VLLM_QWEN3_NEXT_DEBUG_LAYER_STATE" in source
    assert "VLLM_QWEN3_NEXT_DEBUG_LAYER_STATE_LAYERS" in source
    assert "_maybe_log_layer_state(" in source
    assert '"after_attention"' in source
    assert '"after_mlp"' in source
    assert '"attn_state"' in source


def test_011_attention_io_debug_is_opt_in_and_layer_filterable() -> None:
    source = QWEN3_NEXT_011.read_text(encoding="utf-8")

    assert "VLLM_QWEN3_NEXT_DEBUG_ATTENTION_IO" in source
    assert "VLLM_QWEN3_NEXT_DEBUG_ATTENTION_IO_LAYERS" in source
    assert "_maybe_log_attention_io(" in source
    assert '"before_attention"' in source
    assert '"after_attention"' in source
    assert '"query"' in source
    assert '"key"' in source
    assert '"value"' in source
    assert '"attn_output"' in source


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


def test_0202_variant_contains_hybrid_gdn_structure_fixes() -> None:
    source = QWEN3_NEXT_0202.read_text(encoding="utf-8")

    assert "RMSNorm as Qwen3NextPlainRMSNorm" in source
    assert "def _block_rms_norm_cls(config)" in source
    assert "rms_norm_add_unit_offset" in source
    assert "getattr(config, \"num_experts\", None) == 0" in source
    assert "getattr(config, \"enable_qk_norm\", True) is False" in source
    assert "attention_bias" in source
    assert "enable_qk_norm" in source
    assert "self.q_norm = None" in source
    assert "self.k_norm = None" in source
    assert "self.num_moe_layers = 0" in source
    assert "self.num_routed_experts = 0" in source


def test_ascend_compat_installs_hd64_gdn_runtime_workarounds() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")
    init_source = PLUGIN_INIT.read_text(encoding="utf-8")

    assert "def patch_ascend_gdn_hd64_runtime_workarounds()" in source
    assert "VLLM_QWEN3_NEXT_DISABLE_GDN_HD64_FIX" in source
    assert "_qwen3_next_hd64_padded_chunk_fwd" in source
    assert "head_k_dim >= 128" in source
    assert "head_v_dim >= 128" in source
    assert "chunk_module.chunk_gated_delta_rule_fwd =" in source
    assert "GatedDeltaNetAttention._forward_core =" in source
    assert "AscendGatedDeltaNetAttention._forward_core =" in source
    assert "_torch_recurrent_gated_delta_rule_decode" in source
    assert "patch_ascend_gdn_hd64_runtime_workarounds()" in init_source


def test_plugin_init_registers_ascend_oot_ops_before_model_construction() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")
    init_source = PLUGIN_INIT.read_text(encoding="utf-8")

    assert "def ensure_ascend_custom_ops_registered()" in source
    assert "from vllm_ascend.utils import register_ascend_customop" in source
    assert "register_ascend_customop()" in source
    assert "ensure_ascend_custom_ops_registered()" in init_source
    assert (
        init_source.index("ensure_ascend_custom_ops_registered()")
        < init_source.index("from .qwen3_next import Qwen3NextForCausalLM")
    )


def test_ascend_compat_patches_plugin_qwen3_next_attention() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")
    init_source = PLUGIN_INIT.read_text(encoding="utf-8")

    assert "def patch_plugin_qwen3_next_attention(" in source
    assert "Qwen3NextAttention" in source
    assert "triton_split_qkv_rmsnorm_mrope" in source
    assert "getattr(self, \"enable_qk_norm\", True)" in source
    assert "attention_cls.forward = forward" in source
    assert "patch_plugin_qwen3_next_attention(Qwen3NextForCausalLM)" in init_source


def test_ascend_compat_can_force_direct_attention_trace_flag_false() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")
    init_source = PLUGIN_INIT.read_text(encoding="utf-8")

    assert "def patch_ascend_attention_direct_call_trace_flag()" in source
    assert "VLLM_QWEN3_NEXT_FORCE_DIRECT_ASCEND_ATTENTION" in source
    assert "trace_flag=False" in source
    assert "Attention.forward = forward" in source
    assert "patch_ascend_attention_direct_call_trace_flag()" in init_source


def test_ascend_compat_can_force_torch_decode_attention() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")
    init_source = PLUGIN_INIT.read_text(encoding="utf-8")

    assert "def patch_ascend_torch_decode_attention()" in source
    assert "VLLM_QWEN3_NEXT_FORCE_TORCH_DECODE_ATTENTION" in source
    assert "VLLM_QWEN3_NEXT_DEBUG_TORCH_DECODE_ATTENTION" in source
    assert "Qwen3Next torch decode attention" in source
    assert "gathered_key_abs_sum" in source
    assert "slot_mapping_head" in source
    assert "def _derive_slot_mapping_from_block_tables(" in source
    assert "attn_metadata.slot_mapping = corrected_slot_mapping" in source
    assert "def _write_kv_cache_by_slot_mapping(" in source
    assert "_write_kv_cache_by_slot_mapping(" in source
    assert "_torch_decode_attention(" in source
    assert "original_forward" in source
    assert "_qwen3_next_torch_decode_attention_forward" in source
    assert "AscendAttentionBackendImpl._forward_decode_only" in source
    assert "patch_ascend_torch_decode_attention()" in init_source


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


def test_ascend_recurrent_fallback_matches_chunk_for_single_token_decode() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import (
        _torch_chunk_gated_delta_rule,
        _torch_fused_recurrent_gated_delta_rule,
    )

    torch.manual_seed(7)
    q = torch.randn((1, 1, 2, 3), dtype=torch.float32)
    k = torch.randn((1, 1, 2, 3), dtype=torch.float32)
    v = torch.randn((1, 1, 4, 5), dtype=torch.float32)
    g = -torch.rand((1, 1, 4), dtype=torch.float32)
    beta = torch.sigmoid(torch.randn((1, 1, 4), dtype=torch.float32))
    state = torch.randn((3, 4, 3, 5), dtype=torch.float32)

    expected_out, expected_state = _torch_chunk_gated_delta_rule(
        query=q,
        key=k,
        value=v,
        g=g,
        beta=beta,
        initial_state=state[2].unsqueeze(0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    actual_out, actual_state = _torch_fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state.clone(),
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 1]),
        ssm_state_indices=torch.tensor([2]),
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.allclose(actual_out, expected_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_state[2], expected_state[0], atol=1e-5, rtol=1e-5)


def test_ascend_recurrent_fallback_matches_chunk_for_two_token_decode() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import (
        _torch_chunk_gated_delta_rule,
        _torch_fused_recurrent_gated_delta_rule,
    )

    torch.manual_seed(11)
    q = torch.randn((1, 2, 2, 3), dtype=torch.float32)
    k = torch.randn((1, 2, 2, 3), dtype=torch.float32)
    v = torch.randn((1, 2, 4, 5), dtype=torch.float32)
    g = -torch.rand((1, 2, 4), dtype=torch.float32)
    beta = torch.sigmoid(torch.randn((1, 2, 4), dtype=torch.float32))
    state = torch.randn((3, 4, 3, 5), dtype=torch.float32)

    expected_out, expected_state = _torch_chunk_gated_delta_rule(
        query=q,
        key=k,
        value=v,
        g=g,
        beta=beta,
        initial_state=state[2].unsqueeze(0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    actual_out, actual_state = _torch_fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state.clone(),
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 2]),
        ssm_state_indices=torch.tensor([2]),
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.allclose(actual_out, expected_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_state[2], expected_state[0], atol=1e-5, rtol=1e-5)


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


def test_ascend_causal_conv_patch_can_force_torch_fallback(monkeypatch) -> None:
    import sys
    import types

    import vllm_qwen3_next_plugin.compat.ascend as ascend

    module_name = "test_variant_force_torch_conv"
    variant = types.ModuleType(module_name)
    variant.causal_conv1d_fn = object()
    variant.causal_conv1d_update = object()
    monkeypatch.setitem(sys.modules, module_name, variant)
    fake_ascend = types.ModuleType("vllm_ascend")
    fake_ops = types.ModuleType("vllm_ascend.ops")
    fake_conv = types.ModuleType("vllm_ascend.ops.casual_conv1d")
    fake_conv.causal_conv1d_fn = object()
    fake_conv.causal_conv1d_update_npu = object()
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.casual_conv1d", fake_conv)
    monkeypatch.setattr(ascend, "find_spec", lambda name: object())
    monkeypatch.setenv("VLLM_QWEN3_NEXT_FORCE_TORCH_CAUSAL_CONV", "1")

    class FakeModel:
        __module__ = module_name

    assert ascend.patch_plugin_causal_conv1d_ops(FakeModel) is True
    assert variant.causal_conv1d_fn is ascend._torch_causal_conv1d_fn
    assert variant.causal_conv1d_update is ascend._torch_causal_conv1d_update


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


def test_torch_recurrent_decode_fallback_expands_gdn_gqa_heads() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import (
        _torch_recurrent_gated_delta_rule_decode,
    )

    out = _torch_recurrent_gated_delta_rule_decode(
        query=torch.zeros((1, 8, 64), dtype=torch.bfloat16),
        key=torch.zeros((1, 8, 64), dtype=torch.bfloat16),
        value=torch.ones((1, 32, 64), dtype=torch.bfloat16),
        g=torch.zeros((1, 32), dtype=torch.float32),
        beta=torch.ones((1, 32), dtype=torch.float32),
        state=torch.zeros((2, 32, 64, 64), dtype=torch.float32),
        scale=64**-0.5,
        ssm_state_indices=torch.tensor([0]),
    )

    assert out.shape == (1, 32, 64)


def test_torch_recurrent_decode_no_host_sync_matches_token_loop_with_pad() -> None:
    from vllm_qwen3_next_plugin.compat.ascend import (
        PAD_SLOT_ID,
        _torch_recurrent_gated_delta_rule_decode,
    )

    torch.manual_seed(17)
    query = torch.randn((3, 2, 3), dtype=torch.float32)
    key = torch.randn((3, 2, 3), dtype=torch.float32)
    value = torch.randn((3, 4, 5), dtype=torch.float32)
    g = -torch.rand((3, 4), dtype=torch.float32)
    beta = torch.sigmoid(torch.randn((3, 4), dtype=torch.float32))
    state = torch.randn((4, 4, 3, 5), dtype=torch.float32)
    state_indices = torch.tensor([2, PAD_SLOT_ID, 0])
    scale = 3**-0.5

    expected_output = torch.zeros_like(value)
    expected_state = state.clone()
    expanded_query = query.repeat_interleave(2, dim=1)
    expanded_key = key.repeat_interleave(2, dim=1)
    for token_idx, cache_idx in ((0, 2), (2, 0)):
        state_t = expected_state[cache_idx].float()
        q_t = expanded_query[token_idx].float()
        k_t = expanded_key[token_idx].float()
        v_t = value[token_idx].float()
        g_t = g[token_idx].float()
        beta_t = beta[token_idx].float()
        decay_t = g_t.exp()
        v_new = v_t * beta_t.unsqueeze(-1) - torch.einsum(
            "hk,hkv->hv", k_t * beta_t.unsqueeze(-1) * decay_t.unsqueeze(-1), state_t
        )
        expected_output[token_idx] = (
            torch.einsum(
                "hk,hkv->hv", q_t * scale * decay_t.unsqueeze(-1), state_t
            )
            + (q_t * scale * k_t).sum(dim=-1, keepdim=True) * v_new
        )
        expected_state[cache_idx] = state_t * decay_t.view(-1, 1, 1) + torch.einsum(
            "hk,hv->hkv", k_t, v_new
        )

    actual_state = state.clone()
    actual_output = _torch_recurrent_gated_delta_rule_decode(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        state=actual_state,
        scale=scale,
        ssm_state_indices=state_indices,
    )

    assert torch.equal(actual_output, expected_output)
    assert torch.equal(actual_state, expected_state)


def test_torch_recurrent_decode_avoids_scalar_item_sync() -> None:
    source = ASCEND_COMPAT.read_text(encoding="utf-8")
    function_source = source.split(
        "def _torch_recurrent_gated_delta_rule_decode(", 1
    )[1].split("\ndef _wrap_ascend_chunk_gated_delta_rule", 1)[0]

    assert ".item(" not in function_source
    assert "state.index_select(0, safe_cache_idx)" in function_source
    assert "state.index_copy_(0, safe_cache_idx" in function_source
