from types import SimpleNamespace

import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm_qwen3_next_plugin.variants.config_vllm_0_18_1 import Qwen3NextConfig
from vllm_qwen3_next_plugin.variants import mtp_vllm_0_18_1 as mtp_module
from vllm_qwen3_next_plugin.variants.mtp_vllm_0_18_1 import Qwen3NextMultiTokenPredictor
import pytest


def test_config_0_18_1_accepts_custom_architecture_fields() -> None:
    config = Qwen3NextConfig(
        attn_position_embedding_type="nope",
        rnn_position_embedding_type="rope",
        ffn_token_shift="conv",
        ffn_intermediate_token_shift="conv",
        attn_token_shift="conv",
        attn_q_token_shift="conv",
        attn_k_token_shift="conv",
        attn_v_token_shift="conv",
        token_shift_conv_size=8,
        token_shift_conv_init="default",
        attn_logits_scaling="logn 256",
        attn_output_gate=False,
        norm_type="gated_rms",
        gated_norm_rank=32,
        gated_norm_gate_scale=0.5,
        attn_qk_norm=True,
    )

    assert config.attn_position_embedding_type == "nope"
    assert config.rnn_position_embedding_type == "rope"
    assert config.ffn_token_shift == "conv"
    assert config.ffn_intermediate_token_shift == "conv"
    assert config.attn_token_shift == "conv"
    assert config.attn_q_token_shift == "conv"
    assert config.attn_k_token_shift == "conv"
    assert config.attn_v_token_shift == "conv"
    assert config.token_shift_conv_size == 8
    assert config.token_shift_conv_init == "default"
    assert config.attn_logits_scaling == "logn 256"
    assert config.attn_output_gate is False
    assert config.norm_type == "gated_rms"
    assert config.gated_norm_rank == 32
    assert config.gated_norm_gate_scale == 0.5
    assert config.attn_qk_norm is True


def test_config_0_18_1_rejects_invalid_custom_field_values() -> None:
    with pytest.raises(AssertionError):
        Qwen3NextConfig(attn_position_embedding_type="bad")

    with pytest.raises(AssertionError):
        Qwen3NextConfig(rnn_position_embedding_type="bad")

    with pytest.raises(AssertionError):
        Qwen3NextConfig(norm_type="bad")


def test_mtp_predictor_uses_configured_layer_types(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEmbedding(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class DummyLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class DummyNorm(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class DummyDecoderLayer(torch.nn.Module):
        def __init__(self, vllm_config, layer_type: str, prefix: str) -> None:
            super().__init__()
            self.layer_type = layer_type
            self.layer_idx = int(prefix.rsplit(".", 1)[-1])
            if layer_type == "linear_attention":
                self.linear_attn = torch.nn.Identity()
            else:
                self.self_attn = torch.nn.Identity()

    monkeypatch.setattr(mtp_module, "VocabParallelEmbedding", DummyEmbedding)
    monkeypatch.setattr(mtp_module, "ColumnParallelLinear", DummyLinear)
    monkeypatch.setattr(mtp_module, "Qwen3NextRMSNorm", DummyNorm)
    monkeypatch.setattr(mtp_module, "Qwen3NextDecoderLayer", DummyDecoderLayer)

    hf_config = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        num_nextn_predict_layers=1,
        mtp_layer_types=["linear_attention"],
        num_experts=0,
        num_experts_per_tok=0,
        shared_expert_intermediate_size=0,
        moe_intermediate_size=0,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        dtype=torch.bfloat16,
    )
    current_vllm_config = VllmConfig(cache_config=CacheConfig())
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=CacheConfig(),
        quant_config=None,
        speculative_config=None,
        compilation_config=current_vllm_config.compilation_config,
    )

    with set_current_vllm_config(current_vllm_config):
        predictor = Qwen3NextMultiTokenPredictor(vllm_config=vllm_config, prefix="mtp")

    assert isinstance(predictor.layers, torch.nn.ModuleDict)
    assert str(hf_config.num_hidden_layers) in predictor.layers
    layer = predictor.layers[str(hf_config.num_hidden_layers)]
    assert hasattr(layer, "linear_attn")
    assert not hasattr(layer, "self_attn")
    assert layer.layer_idx == hf_config.num_hidden_layers
