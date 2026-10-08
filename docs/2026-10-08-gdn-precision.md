# 2026-10-08 GDN precision fix

## Scope and baseline

Baseline: master `28239e63b1f12fad1e6bbce53bf633ed7dcf90f4`.
This is a small precision patch on the master's flat layout, not a vLLM upgrade.
The user authorized replacing `dev/gyzp`; the old refactored dev implementation
is not merged into this patch. Its previous tip `f40a83ac0c73c9ba397b7141232a32fd82c0bd54`
is preserved locally as `archive/dev-gyzp-before-precision-20261008`.
The existing dev worktree and its uncommitted changes are left untouched.

Primary target: a09 materialized HF, 49 GDN layers, 32 value heads per layer,
key/value head dim 64, vLLM 0.18.1, BF16 model, H20. No model weights are changed.
The external a09 `config.json` must have `moe_router_sqrt_gate: true`.
Its bundled HF modeling does not implement that switch and is not a valid
sqrt-gate reference without its own fix. vLLM must actually load this plugin.

## Changes

1. Explicit FP32 `A_log` and `dt_bias` allocation in all four existing model
   implementations. vLLM loads weights into these destination parameters;
   casting a parameter back to FP32 after loading would be too late.
   TP sharding and checkpoint names are unchanged. The 0.10.2 / 0.11.0 variants
   already used FP32 for `A_log`; those only need the `dt_bias` change.
2. Keep computed beta in FP32. The 0.10.2 / 0.11.0 paths use `b.float().sigmoid()`;
   the 0.15.1 / 0.18.1 fused prefill wrappers allocate an FP32 beta output.
   Their Triton kernels already compute sigmoid in FP32. The 0.18.1 prefill
   warmup uses FP32 g/beta to warm the same dtype specialization as real inputs.
3. For 0.18.1 packed decode, pass `b.float()` to the upstream kernel. That kernel
   otherwise executes `sigmoid(b_fp32).to(b.dtype).to(float32)` and loses precision.
   Promoting the small raw gate buffer avoids this round-trip without copying
   Q/K/V, disabling packed decode, vendoring a kernel, or modifying site-packages.
   This adds a small per-decode conversion/allocation; latency has not been measured.

FP32 preservation of a09's 3,136 decay-parameter elements adds only 6,272 bytes
over BF16, before TP sharding. Keeping `dt_bias` in FP32 preserves this custom
checkpoint, not a claim that upstream requires every model's dt_bias to be FP32.

Not changed: routing/sqrt-gate semantics, model/activation dtype, recurrent-cache
dtype, clamp behavior, version dispatch, MTP configuration, quantization policy.
The historical 0.10.2 variant still does not support a09's sqrt gate; use 0.18.1.

## Evidence and limitations

- Original a09 has 98 FP32 `A_log` / `dt_bias` tensors. CPU inspection found
  3,135 / 3,136 values change on BF16 conversion (maximum absolute error 0.00777).
  This is parameter rounding, **not** a measured logits / evaluation-score delta.
- Upstream 0.18.1 does not explicitly preserve these two parameter dtypes:
  https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/qwen3_next.py
- Upstream 0.31.0 explicitly preserves `A_log`, but not `dt_bias`:
  https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- Upstream packed-beta precision fix:
  https://github.com/vllm-project/vllm/pull/53877

No GPU, server, or full-model accuracy run was performed on the busy local cluster.
No environment packages were installed or modified. The new CPU tests execute
actual source AST fragments with stubbed kernel launches: they test parameter
allocation, TP-shaped checkpoint copies, wrapper buffers, and packed-call dtype
contracts, **not CUDA kernel correctness or whole-model loading**.

CPU validation (56 passed, 7 explicitly skipped GPU tests):

Additional CPU-only checks passed in the existing vLLM 0.18.1 environment:
importing this checkout, calling `register()`, loading the local a09 config
(sqrt gate true, 49 GDN layers, dim 64, plain RMSNorm, clamps disabled), parsing
all plugin Python files, and `git diff --check`. CUDA remained uninitialized.
The pre-existing version-specific GatedRMSNorm test suite was not run as a
multi-version integration matrix; this record covers the targeted precision tests.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python -B -m pytest tests/test_gdn_precision.py tests/test_gdn_precision_cuda.py \
  -q -p no:cacheprovider
```

## H20 validation before evaluation

Use a dedicated vLLM 0.18.1 environment. After obtaining this branch, install the
plugin without changing the environment's vLLM dependencies:

```bash
python -m pip install -e . --no-deps
```

First run the opt-in GPU tests on an idle allocated GPU, from the repo root:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_GDN_CUDA_TESTS=1 \
  python -m pytest tests/test_gdn_precision_cuda.py -q
```

These check real fused prefill gates against FP32 math and packed decode's
single-step recurrent state against unrounded sigmoid(0.5), with head dim 64.
They check the imported plugin path to avoid silently testing another checkout.
They do not load a09 or measure end-to-end quality.

Then use a fresh process and a conservative serving baseline on idle allocated GPUs:

```bash
MODEL=/path/on/h20/to/a09/hf
CUDA_VISIBLE_DEVICES=0,1 VLLM_PLUGINS=register_qwen3_next_model \
  vllm serve "$MODEL" \
  --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 4096 --max-num-seqs 4 \
  --mamba-ssm-cache-dtype float32 \
  --moe-backend triton \
  --additional-config '{"gdn_prefill_backend":"triton"}' \
  --enforce-eager
```

This is a proposed baseline, not a locally validated launch. Confirm the startup
banner registers the custom `Qwen3NextForCausalLM`. Use plain completion prompts
unless this base model has an appropriate chat template. Do not enable MTP for
the drop-MTP checkpoint. The SSM-cache flag does not control parameter precision.

Before formal evaluation compare fixed-token logits / log probabilities, not
only readable text: short and near-4096-token inputs; prefill versus token decode;
packed decode on/off (`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0`); single versus
batched requests. Compare against a reference with sqrt gate actually implemented.
Keep all routing, tokenizer, context, and sampling settings fixed. Differences in
reduction order mean bitwise full-model equality is not an appropriate requirement.

If you need an A/B measurement of this patch, compare it to master `28239e6`
using the **same corrected sqrt-gate config** and identical serving options.
Do not attribute a combined sqrt-gate/config change to the precision patch.
