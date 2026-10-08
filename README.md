modified from [vllm/model_executor/models/qwen3_next.py](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/qwen3_next.py)

## GDN precision

The plugin preserves `A_log`, `dt_bias`, and computed GDN beta in FP32 while
leaving the main model dtype unchanged. For a09, use vLLM **0.18.1** and enable
`moe_router_sqrt_gate` in the checkpoint config.

See [2026-10-08 precision fix and H20 validation](docs/2026-10-08-gdn-precision.md)
for the exact scope, CPU test results, opt-in GPU tests, and serving baseline.
GPU / end-to-end accuracy validation is still required; CPU tests do not certify it.
