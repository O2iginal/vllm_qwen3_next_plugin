# Qwen3-Next 自定义修改记录 (Custom Modifications)

## 2026-10-08：GDN 精度修复

- 所有现有版本实现显式保留 `A_log` / `dt_bias` 为 FP32，避免 BF16 加载舍入。
- prefill 的 sigmoid(beta) 计算与输出保持 FP32；0.18.1 warmup dtype 同步。
- 0.18.1 packed decode 仅将原始门控 `b` 提升为 FP32，绕过上游内核的
  sigmoid→BF16→FP32 舍入，不修改 site-packages，不关闭 packed 快速路径。
- 不改变主权重、激活和 SSM cache 的默认 dtype；后者仍用启动参数独立控制。
- 增加 CPU 回归测试及必须显式启用的 GPU 内核测试。
- 详细验证边界与 H20 命令见 [精度修复记录](docs/2026-10-08-gdn-precision.md)。

---

基于 vLLM 官方 Qwen3-Next 实现，进行自定义修改，按照修改顺序记录如下修改清单，便于升级 vLLM 版本时复用到新代码。

涉及文件：
- `vllm_qwen3_next_plugin/vllm_qwen3_next_0_11_0.py`（主模型，vLLM 0.11.0）
- `vllm_qwen3_next_plugin/vllm_qwen3_next_0_10_2.py`（主模型，vLLM 0.10.2，与 0_11_0 修改点对应）
- `vllm_qwen3_next_plugin/vllm_qwen3_next_config_0_11_0.py`（Config）
- `vllm_qwen3_next_plugin/vllm_qwen3_next_mtp_0_11_0.py`（MTP 多 token 预测）

---

## 1. num_moe = 0 时仍能正确执行（无 MoE 时的 fallback）

**意图**：不强制存在 MoE 层即可跑通；当 `num_experts == 0` 时使用默认元信息，无需从 MoE 层获取。

**修改要点**：

- **Config**  
  - 无新增字段；保证 `num_experts` 可设为 `0`（若上游有校验需放宽或跳过）。0_11_0 Config 中 `num_experts=512` 为默认，由调用方传入 0 即可。

- **主模型 `vllm_qwen3_next_0_11_0.py`**  
  - **DecoderLayer 选 MLP/MoE**：条件为 `(self.layer_idx not in mlp_only_layers) and (config.num_experts > 0 and (self.layer_idx + 1) % config.decoder_sparse_step == 0)` 时才用 `Qwen3NextSparseMoeBlock`；否则若 `use_mlp_token_shift` 用 `Qwen3NextMLPWithTokenShift`，否则用 `Qwen3NextMLP`。即 **必须** `num_experts > 0` 才会建 MoE 层。  
  - **Qwen3Next 主类**：若 `config.num_experts == 0`，不遍历 `model.layers` 找 MoE，直接设 `num_moe_layers = 0` 以及 `num_expert_groups`、`num_shared_experts`、`num_logical_experts`、`num_physical_experts`、`num_local_physical_experts`、`num_routed_experts`、`num_redundant_experts` 均为 0。  
  - **set_eplb_state**：若 `config.num_experts == 0` 则直接 `return`。  
  - **update_physical_experts_metadata**：若 `config.num_experts == 0` 则直接 `return`。  
  - **load_weights**：通过 `get_expert_mapping()` 获取 expert 映射；`get_expert_mapping()` 在 `config.num_experts == 0` 时返回空列表 `[]`，故 MoE 相关权重加载自然被跳过。

复用时在新版本中保留上述 `num_experts == 0` 分支与默认值设置即可。

---

## 2. Attention 中 QKV 有 bias、O 无 bias

**意图**：QKV 缺省为 True（与实现一致）；当 config 中 `attention_bias=True` 时 Q/K/V 投影带 bias；output 投影（o_proj）**始终无 bias**。

**修改要点**：

- **Config**（0_11_0）  
  - `attention_bias=False`（`__init__` 默认）；主模型缺省用 True，由 `getattr(config, "attention_bias", True)` 控制 QKV 是否带 bias。

- **主模型 `Qwen3NextAttention`**  
  - **qkv_proj**：`QKVParallelLinear(..., bias=getattr(config, "attention_bias", True))`。  
  - **o_proj**：`RowParallelLinear(..., bias=False)`，固定无 bias。

复用时检查新版本是否把 o_proj 做成了可选的 bias，若是则改为固定 `bias=False`，并让 qkv 的 bias 由 `config.attention_bias` 控制。

---

## 3. 默认不开启 attention gate

**意图**：默认不启用 attention output gate（缺省 False），仅当设置 `attn_output_gate = True` 时才启用；与实现已对齐。

**修改要点**：

- **Config（0_11_0）**  
  - 已增加 `attn_output_gate=False`（`__init__` 及 `self.attn_output_gate`）。

- **主模型 `Qwen3NextAttention`**  
  - `self.attn_output_gate = getattr(config, "attn_output_gate", False)`。  
  - qkv_proj 的 output 维度为 `self.total_num_heads * (1 + self.attn_output_gate)`；forward 中仅当 `self.attn_output_gate` 为 True 时做 `q_gate, k, v` 的 split 以及 `attn_output = attn_output * gate`。

复用时确认默认值为 `False`，且 gate 相关维度与分支与现有一致。

---

## 4. Norm 使用 RMSNorm（非 GemmaRMSNorm）

**意图**：在 config 中设置 `norm_type`，默认为普通 RMSNorm（`"rms"`）；可选 `"gemma_rms"`；主模型与 MTP 均按此选择。

**修改要点**：

- **Config（0_11_0）**  
  - 已增加 `norm_type="rms"`（`"rms"` | `"gemma_rms"`），并写入 `self.norm_type`。

- **主模型 `vllm_qwen3_next_0_11_0.py`**  
  - 导入 `RMSNorm`、`GemmaRMSNorm`；提供 `_get_qwen3_next_norm_cls(config)`，按 `config.norm_type` 返回 RMSNorm 或 GemmaRMSNorm。  
  - `input_layernorm`、`post_attention_layernorm`、最终 `norm` 均用 `_get_qwen3_next_norm_cls(config)(...)` 构造。  
  - GDN 内部的 `RMSNormGated` 不变。

- **MTP `vllm_qwen3_next_mtp_0_11_0.py`**  
  - 使用 `_get_qwen3_next_norm_cls(config)` 构造 `norm`、`pre_fc_norm_hidden`、`pre_fc_norm_embedding`，与主模型对齐 norm_type。

复用时在新版中按 config.norm_type 选择 norm 类即可。

---

## 5. 默认关闭 QK Norm（支持 config 开启）

**意图**：full-attention 中对 Q/K 不做 QK Norm（默认关闭）；可通过 config 选项开启，开启后对 q、k 分别做 RMSNorm（与 norm_type 一致）后再参与 attention。

**修改要点**：

- **Config（0_11_0）**  
  - 已增加 `attn_qk_norm=False`（`__init__` 及 `self.attn_qk_norm`）。

- **主模型 `Qwen3NextAttention`**  
  - `self.attn_qk_norm = getattr(config, "attn_qk_norm", False)`。  
  - 若为 True：创建 `self.q_norm`、`self.k_norm`（用 `_get_qwen3_next_norm_cls(config)`，head_dim，rms_norm_eps）；forward 在 split 出 q/k 且 token shift 之后、RoPE 之前，对 q/k 分别做 `q_norm`/`k_norm`（view 成 [..., head_dim] 再 norm 再 view 回）。  
  - 若为 False：`self.q_norm`、`self.k_norm` 为 None，forward 中不应用，保持原样。

复用时在新版 Attention 中按 config.attn_qk_norm 分支创建与应用 q_norm/k_norm 即可。

---

## 6. 支持 Cannon Layer（Token Shift）

**意图**：在 attn 入口、GDN 输出、MLP 入口/中间、以及 MoE 入口等处支持 token shift（cannon layer）；实现方式包括 `conv`（因果 depthwise conv）等。

**修改要点**：

- **Config（0_11_0）**  
  - 已增加：`ffn_token_shift`, `ffn_intermediate_token_shift`, `attn_token_shift`, `attn_q_token_shift`, `attn_k_token_shift`, `attn_v_token_shift`（均为 `None` | `"cat"` | `"conv"`），`token_shift_conv_size=4`, `token_shift_conv_init="default"`。

- **主模型 0_11_0 实际实现**  
  - **工具函数**：`_apply_causal_depthwise_conv1d_bcl(x_bcl, weight_c1w)`，用 `F.pad` + `F.conv1d`，无状态、因果，便于 torch.compile。  
  - **MLP**：`Qwen3NextMLPWithTokenShift`：入口与 down_proj 前（intermediate 上）可选 conv token shift；`ffn_token_shift`/`ffn_intermediate_token_shift == "conv"` 时注册对应 `nn.Conv1d`，forward 中调用 `_apply_causal_depthwise_conv1d_bcl`。DecoderLayer 在 `use_mlp_token_shift` 为 True 时选用此类。  
  - **MoE**：`Qwen3NextSparseMoeBlock` 根据 `ffn_token_shift == "conv"` 注册 `token_shift_conv`，forward 入口对输入做一次因果 depthwise conv。  
  - **Attention 入口**：`Qwen3NextDecoderLayer` 在调用 `self_attn` 前，若 `attn_token_shift == "conv"`，对 `hidden_states` 做一次因果 depthwise conv 再送入 attention。  
  - **Q/K/V 单独 shift**：`Qwen3NextAttention` 中按 `attn_q_token_shift` / `attn_k_token_shift` / `attn_v_token_shift == "conv"` 注册对应 Conv1d，在 forward 中 split 出 q/k/v 后、RoPE 前对 q、k、v 分别做 `_apply_causal_depthwise_conv1d_bcl`。

复用时在新版中找回对应 MLP/MoE/DecoderLayer/Attention 位置，把上述 token_shift 的注册与 forward 分支按 config 重新接上；注意张量格式（如 `[B,C,L]`）与 causal 约束。

---

## 7. 支持单独设置 Attn 与 RNN（GDN）的 RoPE

**意图**：Attention 与 RNN（GatedDeltaNet）可分别配置是否使用 RoPE（如 attn 用 rope，rnn 用 nope，或反之）。

**修改要点**：

- **Config（0_11_0）**  
  - 已增加（约 154–161、246–248、314–316 行）：`attn_position_embedding_type="rope"`, `rnn_position_embedding_type="nope"`；`__init__` 中 assert 仅允许 `"rope"` 或 `"nope"`（约 254–259 行）。

- **主模型 0_11_0 实现**  
  - **Qwen3NextAttention**：`self.attn_position_embedding_type = getattr(config, "attn_position_embedding_type", "rope")`；仅当 `== "rope"` 时创建 `self.rotary_emb`，否则 `None`；forward 中仅当 `self.rotary_emb is not None` 时对 q,k 应用 RoPE。  
  - **Qwen3NextGatedDeltaNet**：
    - `self.rnn_position_embedding_type = getattr(config, "rnn_position_embedding_type", "nope")`；仅当 `== "rope"` 时创建 `self.rotary_emb`。  
    - **positions 传递**：`forward(hidden_states, output, positions=None)` 中，若 `positions is not None` 则通过 `setattr`/`getattr` 在 context 上动态读写 `gdn_positions`（ForwardContext 为 dataclass，用 getattr/setattr 读写，避免 .get/.setdefault 以兼容 Dynamo 编译），再调用 `torch.ops.vllm.gdn_attention`；在 `_forward` 内读取并应用 RoPE 至 q、k 后再进入 recurrent。  
  - **Qwen3NextDecoderLayer**：调用 `self.linear_attn(hidden_states=..., output=..., positions=positions)`，将 positions 传入 GDN。

复用时在新版中定位 Attention 与 GDN 的 RoPE 创建与调用处，改为按上述两个 config 分别分支；GDN 的 positions 通过 forward 参数传入，在 GDN 内用 **getattr/setattr 在 ForwardContext 上读写 `gdn_positions`**（勿用 `.get`/`.setdefault`，因 ForwardContext 为 dataclass）。

---

## 8. 支持 Attn 的 logits scaling

**意图**：在 attention 中对 query 做可选的 logits scaling（用于长度外推等），再送入 attention 计算。

**修改要点**：

- **Config（0_11_0）**  
  - 已增加：`attn_logits_scaling=None`。含义：`None` 不缩放；`float` 常数缩放 `q = q * scale`；`str` 如 `"log"` 或 `"log <a>"` 表示 `scale = log(position+a)/log(a)`，`a` 缺省 362.0。

- **主模型 `Qwen3NextAttention`**  
  - `self.attn_logits_scaling = getattr(config, "attn_logits_scaling", None)`。  
  - 在应用 RoPE 之后、`self.attn(q, k, v)` 之前：若为 int/float 则 `q = q * self.attn_logits_scaling`；若为 str 则 `parts = self.attn_logits_scaling.split()`，`a = float(parts[1]) if len(parts) > 1 else 362.0`，`scale = (torch.log(pos_f + a) / math.log(a)).unsqueeze(-1)`，`q = q * scale`。  
  - 然后执行 `attn_output = self.attn(q, k, v)`。

复用时在新版 Attention 的 forward 中，在 RoPE 之后、attention 计算之前插入上述分支即可。

---

## 9. GatedRMSNorm（norm_type="gated_rms"）

**意图**：在 decoder 的 input_layernorm / post_attention_layernorm（及最终 norm）上支持 GatedRMSNorm：先 RMSNorm，再经低秩门控 `gate = sigmoid(W_up(swish(W_down(y)))) * gate_scale`，输出 `gate * y`。与 HF modeling 中 `Qwen3NextGatedRMSNorm` 及训练侧 GatedRMSNorm 对齐，便于加载转换后的 gated_rms 权重。GDN 内部 norm 仍为普通 RMS，不受 norm_type 影响。

**修改要点**：

- **Config（0_15_1）**  
  - `norm_type` 增加可选值 `"gated_rms"`（`"rms"` | `"gemma_rms"` | `"gated_rms"`），并增加 `gated_norm_rank=16`、`gated_norm_gate_scale=1.0`，写入 `self.gated_norm_rank`、`self.gated_norm_gate_scale`。  
  - `__init__` 中对 `norm_type` 做 assert，仅允许上述三值。

- **主模型 `vllm_qwen3_next_0_15_1.py`**  
  - 新增 `GatedRMSNorm` 类：内部为 `RMSNorm` + `nn.Linear` 的 `w_down`（hidden_size -> rank）、`w_up`（rank -> hidden_size），零初始化；forward 为 `y = rms_norm(x)`，`gate = sigmoid(w_up(swish(w_down(y)))) * gate_scale`，`return gate * y`。  
  - 支持与现有 norm 一致的 fused residual 调用：`forward(x, residual=None)`，当 `residual is not None` 时返回 `(out, residual)`，否则仅返回 `out`。  
  - `_get_qwen3_next_norm_cls(config)`：当 `config.norm_type == "gated_rms"` 时返回一工厂函数 `_factory(hidden_size, eps)`，内部根据 `config.gated_norm_rank`、`config.gated_norm_gate_scale` 构造并返回 `GatedRMSNorm` 实例；其余与 custom 4 相同，仍返回 RMSNorm / GemmaRMSNorm 类。  
  - `input_layernorm`、`post_attention_layernorm`、最终 `norm` 的构造方式不变，仍为 `_nc(config.hidden_size, eps=config.rms_norm_eps)`，故对 gated_rms 会得到 `GatedRMSNorm` 实例。

- **权重加载**  
  - HF 转换后权重名为 `input_layernorm.rms_norm.weight`、`input_layernorm.w_down.weight`、`input_layernorm.w_up.weight`（及 post_attention_layernorm、norm 的对应 key），与 `GatedRMSNorm` 的 `named_parameters()` 一致，无需额外 remap。

复用时在新版中保留 `GatedRMSNorm` 类、`_get_qwen3_next_norm_cls` 对 `gated_rms` 的分支及 config 三字段即可。

---

## 其他注意事项

- **Config 默认值**：升级 vLLM 后请对照本 CHANGELOG 与 `vllm_qwen3_next_config_0_11_0.py` 的 `__init__` 默认值，避免遗漏或冲突。
- **0_10_2 与 0_11_0**：上述修改在 `vllm_qwen3_next_0_10_2.py` 中均有对应实现，复用时两处需同步更新（尤其 GDN positions 传递、Token shift、RoPE 等）。
- **ForwardContext**：vLLM 的 `ForwardContext` 为 dataclass，无 `.get`/`.setdefault`；自定义字段请用 `getattr`/`setattr` 读写，否则在 Dynamo 编译下会触发 AttributeError。

## 10. Final gated norm and up-projection bias

**Intent**: support checkpoints whose final norm is `gated_rms` and whose `w_up` projection has a bias. Previously, the causal LM always created a plain final `RMSNorm`, so loading failed on `model.norm.weight`; additionally, `model.norm.w_up.bias` was ignored even when `gated_norm_up_bias=true`.

**Changes**:

- `GatedRMSNorm` accepts `up_bias` and creates `w_up.bias` when required.
- The norm factory reads `gated_norm_up_bias`.
- The final norm uses the gated norm when `final_gated_norm=true`; otherwise it remains plain RMS norm.

---

## 0_11_0 实现状态小结（便于复用核对）

| 条目 | Config 0_11_0 | 主模型 / MTP 0_11_0 | 备注 |
|------|----------------|----------------------|------|
| 1 num_moe=0 | 无新增 | 已实现 fallback | — |
| 2 QKV bias / O 无 bias | attention_bias=False | qkv 缺省 True，o_proj bias=False；意图已对齐 | — |
| 3 attention gate | attn_output_gate=False | getattr(..., False)；意图已对齐 | — |
| 4 Norm | norm_type="rms" | 主模型与 MTP 均按 _get_qwen3_next_norm_cls(config) | — |
| 5 QK Norm | attn_qk_norm=False | 默认关闭；True 时创建并应用 q_norm/k_norm | — |
| 6 Token shift | 已加全部字段 | 已实现 ffn/attn 入口、MoE 入口、**attn q/k/v 分别 shift** | — |
| 7 Attn/RNN RoPE | 已加并校验 | 已实现；GDN 通过 gdn_positions 传位置（getattr/setattr 读写） | — |
| 8 logits scaling | 已加 | 已实现 | — |
| 9 GatedRMSNorm | norm_type 支持 gated_rms；gated_norm_rank / gated_norm_gate_scale | GatedRMSNorm 类 + _get_qwen3_next_norm_cls 工厂 | 0_11_0 / 0_15_1 |


# 版本适配记录

## 0.11.0

创建本 `CHANGELOG.md` 时，vLLM 版本为 0.11.0.

## 0.15.1

## 0.18.1

- 新增 `vllm_qwen3_next_0_18_1.py` / `vllm_qwen3_next_config_0_18_1.py` / `vllm_qwen3_next_mtp_0_18_1.py`：基于 stock 0.18.1 重放全部自定义 delta（GatedRMSNorm 含 `gated_norm_up_bias`/`final_gated_norm`、token-shift/cannon 卷积、rope/nope、qk_norm、`num_experts==0` fallback 等）；stock config 在 0.15.1 与 0.18.1 之间逐字节一致，config delta 1:1 迁移。GDN RoPE 逻辑挂入重构后的 `_forward_core`；启用 GDN RoPE 时绕过不解体 q/k 的 packed-decode 快路径。保留 stock 0.18 的 FlashInfer GDN prefill、kernel warmup 等优化。
- `__init__.py` 分发新增 `0.18` 分支。

## 10. MoE router sqrt gate（moe_router_sqrt_gate）

**意图**：对齐 Megatron `megatron/core/transformer/moe/router.py` 的 sqrt-gate：softmax → top-k → L1 归一化之后，将选中专家的组合权重替换为 `sqrt(w)`（未选中保持 0，不再二次归一化；sum-to-1 仅保留给上游 aux-loss 语义）。

**修改要点**：

- 通过 `FusedMoE`/`SharedFusedMoE` 的 `custom_routing_function` 钩子实现 `_sqrt_gate_routing_function`（内部调用各版本自带的 `fused_topk`，返回 `topk_weights.sqrt()` 与原始 ids）。
- 0_11_0 / 0_15_1 / 0_18_1 均已接入；config 新增 `moe_router_sqrt_gate=False`（默认关闭）。

## 11. MLP gate/up 软钳制（activation_func_clamp_value）

**意图**：对齐 HF 参考 `modeling_qwen3_next.py`：`gate = c*tanh(gate/c)`、`up = c*tanh(up/c)` 后再做 `silu(gate)*up`（c=7.0）。

**修改要点**：

- 以 `_SiluAndMulClampShim` 包装类包裹 `torch.ops._C.silu_and_mul`：未激活时透传原 `OpOverloadPacket`（`__getattr__` 委托，`.default` 等属性完整，不影响 torch.compile fusion 的 pattern matching）；激活后按 HF 公式逐元素计算。MoE 各后端（fused_experts 内联调用与 modular activation）与 dense MLP/shared expert 的 `SiluAndMul` 均覆盖。
- config 新增 `activation_func_clamp_value=None`、`activation_func_clamp_mode="soft"`；非 soft 模式或多模型值冲突会显式报错。
- 已知边界：FlashInfer CUTLASS MoE 内核路径（0.11 需 `VLLM_USE_FLASHINFER_MOE_FP16=1`+EP；0.18+DP）在 kernel 内部做激活，会绕过钳制；启用钳制时请勿走该路径。

## 12. logits 硬钳制（final_logits_clamp_value）

**意图**：对齐 HF 参考：`lm_head(...).clamp(-v, v)`（v=30.0）。

**修改要点**：

- 各版本 `Qwen3NextForCausalLM.compute_logits`（及 MTP）在 `final_logits_clamp_value` 设置时对 logits 做硬钳制；config 默认 `None` 不启用。
