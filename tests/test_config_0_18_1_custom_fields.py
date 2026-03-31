from vllm_qwen3_next_plugin.variants.config_vllm_0_18_1 import Qwen3NextConfig
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
