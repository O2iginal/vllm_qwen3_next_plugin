# vLLM 0.20.2 Hybrid GDN Plugin Implementation Plan

**Commit snapshot:** 2026-06-15 dev/gyzp_ascend staged commit. Correctness integrated and validated for serve/GSM8K. Efficiency (OpenCompass batch) is the active follow-up.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the validated CANN9 hybrid-GDN fixes into the plugin instead of relying on manual runtime patches.

**Architecture:** Keep checkpoint model structure changes in the `vllm_0_20_2` variant. Keep Ascend-specific kernel workarounds in `compat/ascend.py`, installed automatically during plugin registration.

**Tech Stack:** Python, vLLM plugin registry, vLLM-Ascend runtime monkey patches, pytest, real `vllm serve` validation.

---

### Task 1: Static Regression Tests

**Files:**
- Modify: `tests/test_ascend_static.py`

- [x] **Step 1: Add failing tests**

Add tests asserting that the 0.20.2 variant contains plain RMS selection, `attention_bias`, optional QK norm, and dense `num_experts=0` metadata. Add tests asserting that `compat/ascend.py` exposes and registers the head_dim<128 GDN runtime workaround.

- [x] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/test_ascend_static.py::test_0202_variant_contains_hybrid_gdn_structure_fixes tests/test_ascend_static.py::test_ascend_compat_installs_hd64_gdn_runtime_workarounds -q
```

Expected before implementation: both tests fail.

### Task 2: 0.20.2 Variant Structure Fixes

**Files:**
- Modify: `vllm_qwen3_next_plugin/variants/vllm_0_20_2.py`

- [x] **Step 1: Add norm selector**

Add `_block_rms_norm_cls(config)` and import `RMSNorm as Qwen3NextPlainRMSNorm`.

- [x] **Step 2: Apply attention compatibility**

Use `attention_bias` as QKV bias fallback and create/apply q/k norms only when `enable_qk_norm` is true.

- [x] **Step 3: Apply block/final norm compatibility**

Use `_block_rms_norm_cls(config)` for decoder input/post-attention norms and final model norm.

- [x] **Step 4: Apply dense MoE metadata compatibility**

If no MoE layers exist, set all MoE metadata counts to zero and return.

### Task 3: Ascend Runtime Workarounds

**Files:**
- Modify: `vllm_qwen3_next_plugin/compat/ascend.py`
- Modify: `vllm_qwen3_next_plugin/__init__.py`

- [x] **Step 1: Add prefill workaround**

Patch `vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule_fwd` so head_dim<128 and PCP world size 1 pads K/Q/W/U/state to 128, calls the triton h/o kernels, then slices output and final state back.

- [x] **Step 2: Add decode workaround**

Patch `GatedDeltaNetAttention._forward_core` so head_dim<128 temporarily disables `enable_packed_recurrent_decode`, leaving 128-dim official Qwen3-Next unchanged.

- [x] **Step 3: Register workaround**

Call `patch_ascend_gdn_hd64_runtime_workarounds()` from lazy model import and `register()`.

### Task 4: Documentation and Validation

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `/mnt/yulan-pretrain/gaoyanzipeng/progress.md`

- [ ] **Step 1: Document usage and controls**

Record `VLLM_PLUGINS=ascend,register_qwen3_next_model`, `--hf-overrides '{"rms_norm_add_unit_offset": false}'`, and `VLLM_QWEN3_NEXT_DISABLE_GDN_HD64_FIX=1`.
Also record that the plugin path must be prepended to the existing Ascend `PYTHONPATH`, not assigned as the only value.

- [ ] **Step 2: Run pytest**

Run:

```bash
pytest tests/test_ascend_static.py tests/test_versioning.py -q
```

Latest evidence: `PYTHONPATH=. pytest tests/test_ascend_static.py tests/test_versioning.py -q` passed with `39 passed`.

- [x] **Step 3: Run trusted serve validation** (correctness)

Start trusted hybrid-GDN serve with plugin enabled and verify `/health`, Japan prompt, and Janet GSM8K `$18`.

Latest status (as of this commit): 
- Service starts cleanly with `PYTHONPATH=/path/to/vllm_qwen3_next_plugin:${PYTHONPATH} VLLM_PLUGINS=ascend,register_qwen3_next_model` + `--hf-overrides '{"rms_norm_add_unit_offset": false}'`.
- `/health` OK; short factual prompts (capital of Japan/France) and full Janet GSM8K produce correct coherent output + `$18` matching native HF baseline.
- Three root causes addressed via plugin: plain RMSNorm, prefill cross-chunk (pad-to-128 triton), packed recurrent decode guard (hd<128 falls back to normal branch).

Remaining open (efficiency phase): OpenCompass GSM8K full batch eval shows very low NPU AICore util (~0-1%) and high EngineCore CPU because hd=64 decode lands in torch fallback after the correctness guards. Next work: converge decode to efficient NPU or batched torch path inside `compat/ascend.py`. See gaoyanzipeng/progress.md for full profile history.

- [ ] **Step 4 (new)**: Profile + implement efficient hd<128 GDN decode path for OpenCompass throughput.
