# vLLM 0.20.2 Hybrid GDN Plugin Integration Design

## Goal

Move the validated CANN9 hybrid-GDN fixes from manual `site-packages` patches into the plugin registration path, so serving only requires enabling `register_qwen3_next_model`.

Current status: model-structure fixes, Ascend OOT registration, and prefill workarounds are in the plugin. The head_dim<128 GDN long-decode path is still under service-level validation and is not complete.

## Architecture

Model-structure differences live in `vllm_qwen3_next_plugin/variants/vllm_0_20_2.py`. Ascend runtime workarounds live in `vllm_qwen3_next_plugin/compat/ascend.py` and are installed by `register()` before model registration.

## Scope

- `rms_norm_add_unit_offset=False` selects plain decoder/final RMSNorm.
- `attention_bias` is honored as the QKV bias fallback.
- `enable_qk_norm=False` skips full-attention q/k norms and their weights.
- `num_experts=0` marks the model as dense instead of raising in MoE metadata setup.
- Ascend head_dim<128 GDN prefill pads K/V state dimensions to 128 for the triton path, then slices outputs/final state back.
- Ascend head_dim<128 decode fallback is being integrated in `compat/ascend.py`; current service-level GSM8K validation is still failing.

## Runtime Controls

- Enable plugin: `VLLM_PLUGINS=ascend,register_qwen3_next_model`.
- Preserve the Ascend runtime `PYTHONPATH` when adding the plugin path:
  `PYTHONPATH=/path/to/plugin:${PYTHONPATH}`.
- Disable the head_dim<128 Ascend workaround for debugging:
  `VLLM_QWEN3_NEXT_DISABLE_GDN_HD64_FIX=1`.
- The trusted checkpoint still passes:
  `--hf-overrides '{"rms_norm_add_unit_offset": false}'`.

## Validation

The final validation target is a real `vllm serve` request on the trusted hybrid-GDN checkpoint:

- `/health` returns ready.
- `The capital of Japan is` produces readable English.
- Janet GSM8K prompt produces `$18`.

As of the latest run, `/health` and readable short prompts have been observed, but Janet GSM8K long decode has not passed.
