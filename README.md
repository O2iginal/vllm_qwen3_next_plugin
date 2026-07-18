# vLLM Qwen3 Next Plugin

这是一个用于替换 `Qwen3NextForCausalLM` 实现的 vLLM 外部插件仓库。目标不是只做可导入的 demo，而是让我们自定义架构的 Qwen3 Next checkpoint 能在不同版本的 `vllm` 中稳定完成真实推理。

当前仓库仍在重构与多版本适配过程中，但已经确认：

- `0.11.0` 可完成真实服务级推理
- `0.15.1` 可完成真实服务级推理
- `0.15.1` 已通过更严格的 `hybrid gdn + moe` checkpoint 服务级验证
- `0.18.1` 已完成 quick checkpoint 的真实服务级验证
- `0.18.1 + hybrid gdn + moe` 已完成严格 `vllm serve` 启动与请求返回验证
- `0.18.1 + hybrid gdn + MTP` 已完成插件侧 speculative decoding 兼容适配，当前开发分支为 `dev/gyzp_mtp`
- `0.20.2 + CANN9 + trusted hybrid GDN` 已完成真实TP4+EP服务和完整GSM8K验证；plugin负责NoPE、sqrt-gate等模型语义，head_dim=64 recurrent正确性还依赖下述vLLM-Ascend `native_compact`运行时提交

## 当前已验证基线

- 已验证环境：`vllm 0.11.0`
- 已验证环境：`vllm 0.15.1`
- 已验证环境：`vllm 0.18.1`
- 已验证环境：`vllm 0.20.2`（CANN9 / vLLM-Ascend trusted hybrid GDN）
- 已验证入口：`VLLM_PLUGINS=register_qwen3_next_model`
- 已验证能力：
  - 插件导入成功
  - `register()` 成功接管 `Qwen3NextForCausalLM`、`Qwen3NextMTP`、`Qwen3NextConfig`
  - `register()` 会安装 Qwen3Next MTP speculative runtime patches
  - 能真实启动 `python -m vllm.entrypoints.openai.api_server`
  - 能对测试 checkpoint 返回可读自然语言
- 能对更严格的 `hybrid gdn + moe` checkpoint 返回可读自然语言
- CANN9 trusted hybrid GDN checkpoint 已通过Janet长decode、20题门和GSM8K全集1319题验证；最终为1129/1319=85.60%，与历史同配置86.884%相差1.28pp，按“约86%”口径判定PASS

## vLLM 0.20.2 / CANN9 trusted hybrid GDN

`0.20.2` 的 trusted hybrid GDN模型语义已经收敛到插件；CANN9 head_dim=64 recurrent运行时还需使用
vLLM-Ascend commit `eeb96a64ba97a85e319a2da0fccb2420971a04f9`或其后继版本。不要手工编辑单个
`site-packages`文件：应安装/checkout该vLLM-Ascend版本，并确认运行时`gdn.py`与源码一致。启动时启用Ascend插件和
本插件，保留Ascend环境已有的`PYTHONPATH`：

```bash
PYTHONPATH=/path/to/vllm_qwen3_next_plugin:${PYTHONPATH} \
VLLM_PLUGINS=ascend,register_qwen3_next_model \
VLLM_QWEN3_NEXT_DISABLE_GDN_HD64_FIX=1 \
VLLM_ASCEND_GDN_HD64_DECODE_BACKEND=native_compact \
vllm serve /path/to/trusted-hybrid-gdn \
  --trust-remote-code \
  --trust-request-chat-template \
  --enable-expert-parallel \
  --tensor-parallel-size 4 \
  --hf-overrides '{"rms_norm_add_unit_offset": false}'
```

不要用 `PYTHONPATH=/path/to/plugin` 覆盖原值；否则 CANN9 的 `acl` Python 模块不会传入 EngineCore spawn 子进程。

本路径包含以下兼容逻辑：

- `rms_norm_add_unit_offset=false` 时 decoder block norm 与 final norm 使用普通 `RMSNorm(w)`，官方 Qwen3-Next 默认仍使用 Gemma-style `GemmaRMSNorm(1+w)`。
- `attention_bias` 可作为 QKV bias 配置来源；`o_proj` 仍固定无 bias。
- `enable_qk_norm=false` 时跳过 full-attention q/k norm，避免加载不存在的 q_norm/k_norm 权重。
- `num_experts=0` 被视为 dense 模型，MoE metadata 计数置零。
- 本轮验证的head_dim=64 decode由vLLM-Ascend `native_compact`把活动SSM状态映射到连续slot，再调用原生NPU recurrent算子并写回side cache；它不是旧的Python/Torch decode fallback。
- 插件环境变量`VLLM_QWEN3_NEXT_DISABLE_GDN_HD64_FIX=1`只关闭插件内较旧的hd64 monkey patch，不能同时关闭vLLM-Ascend的修复；后者使用不同的`VLLM_ASCEND_DISABLE_GDN_HD64_WORKAROUND`开关，正式运行不要设置为1。
- 当前hd64 prefill为正确性保留Torch chunk fallback。完整GSM8K八副本运行中，动态加入一个prompt会让已有decode停约20--40秒，全集耗时约2小时32分；下一步应先验证fused chunk prefill在连续长请求和slot复用下的正确性，再进行性能切换。
- 2026-07-18全集验证使用修正版550B roundtrip HF、TP4+EP、八个服务副本和OpenCompass `qwen3_reason_32k`口径。温度0.6会产生逐样本采样翻转，因此85.60%不表示逐字复现历史86.884%，但已通过任务级质量门、完整输出门和真实NPU递推回归。

调试插件自身的旧hd64 patch时可使用：

```bash
VLLM_QWEN3_NEXT_DISABLE_GDN_HD64_FIX=1
```

本轮重构的最低要求不是“还能 import”，而是至少回到这个服务级基线。

## 已知自定义方向

当前仓库的自定义修改主要围绕以下 checkpoint 结构差异展开，详细清单见 `CHANGELOG.md`：

- `num_experts=0` 的非 MoE fallback
- `qkv` 带 bias、`o_proj` 固定无 bias
- 默认关闭 `attn_output_gate`
- `norm_type` 扩展，包括 `rms` / `gemma_rms` / `gated_rms`
- 默认关闭 `attn_qk_norm`
- cannon layer / token shift
- attention 与 GDN 分离位置编码
- attention logits scaling
- MTP draft layers 使用全局 layer index，并按 `mtp_layer_types` 支持 full attention / linear attention 混排
- vLLM speculative decoding 中对 Qwen3Next MTP draft attention layer names、metadata builder、attention group 做运行时兼容修补

## 重构目标

参考 `sglang_qwen3_next_plugin` 的已完成工程化结构，本仓库后续将重构为：

- `upstream/`
  - 保存不同 `vllm` 版本的上游基线快照
- `variants/`
  - 保存各版本实际插件实现
- `compat/`
  - 仅保存某个 `vllm` 版本特有的运行时兼容补丁
- `versioning.py`
  - 管理版本键、分发逻辑、路径和模块映射
- 项目级 `.codex/skills/`
  - 固化未来新增 `vllm` 版本适配流程

## 验收原则

后续所有重构和升版都至少需要通过两级验证：

1. 轻量验证
   - 包导入
   - `register()` 分发
   - 当前版本命中正确实现
2. 服务级验证
   - 能启动 OpenAI API server
   - 能加载真实 checkpoint
   - 能发送请求并返回自然语言

`sglang` 仓库里的“生成和 logprob 验收”方法会迁移到这里，作为统一验收链路。

当前已经确认的两类 checkpoint：

- 快速回归 checkpoint
  - 用于快速判断结构改动是否破坏服务路径
- 最终严格 checkpoint
  - `hybrid gdn + moe`
  - 用于覆盖 `num_experts > 0`、`enable_expert_parallel`、shared-expert 相关路径

## 真实运行模板

除了最小 `api_server` 验证外，后续还需要对齐更贴近实际使用的 `vllm serve` 路径。

当前用户提供的参考命令为：

```bash
source /mnt/ssd/yulan/pretrain-linear-moe/evaluation/opencompass/.venv-vllm0.11.0/bin/activate
CUDA_VISIBLE_DEVICES=5,7 \
VLLM_ATTENTION_BACKEND=FLASHINFER \
VLLM_USE_V1=1 \
VLLM_ENGINE=1 \
vllm serve /mnt/ssd/yulan/pretrain-linear-moe-dev-worktree/YuLan-Pretrain-gyzp_mom/cache/exports/yulan_hybrid_gdn_maskmoe_to_moe_iter17163/iter_release-hf \
  --port 30115 \
  --max-num-seqs 1024 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --tensor-parallel-size 2
```

后续阶段性 serve 验证会优先向这条命令靠拢，而不是只用最小参数路径。

## 当前阶段结论

到目前为止：

- `0.11.0`：导入、注册、服务、生成均已通过
- `0.15.1`：导入、注册、服务、生成均已通过
- `0.15.1 + hybrid gdn + moe`：严格 `vllm serve` 路径已通过
- `0.18.1`：quick checkpoint 的 `serve -> /health -> completion` 已通过
- `0.18.1 + hybrid gdn + moe`：严格 `vllm serve` 路径已能启动并返回文本
- `0.18.1 + hybrid gdn + MTP`：已支持 MTP checkpoint 的 draft model 加载、spec step 轮转、linear-attention draft layer attention metadata 兼容；对应分支 `dev/gyzp_mtp`

## MTP speculative 适配要点

当前 `dev/gyzp_mtp` 分支补齐了 vLLM `0.18.1` 上 Qwen3Next MTP 作为 draft model 时的关键兼容路径：

- `vllm_qwen3_next_plugin/variants/mtp_vllm_0_18_1.py`
  - `Qwen3NextMultiTokenPredictor.layers` 改为按全局 layer index 存储的 `ModuleDict`
  - MTP layer index 从 `config.num_hidden_layers` 后继续编号，匹配 HF/MCore 转换后的权重命名
  - 每个 MTP head 按 `config.mtp_layer_types[idx]` 选择 `full_attention` 或 `linear_attention`
  - speculative step 通过 `_qwen3_next_mtp_spec_step_idx` 轮转到对应 MTP head
- `vllm_qwen3_next_plugin/compat/speculative.py`
  - 在 `register()` 阶段安装 runtime patches
  - 对 Qwen3Next linear-attention MTP draft layers 补齐 `_draft_attn_layer_names`
  - 为 draft linear-attention layer 找到正确 metadata builder
  - 重建或切换 draft attention groups，避免 vLLM 只按 full-attention draft layer 组织 metadata
- `0.11.0 / 0.15.1 / 0.18.1` 的 MTP variant 均已同步全局 layer index 与 `mtp_layer_types` 逻辑，便于后续版本对齐。

注意：这部分是针对 Qwen3Next MTP speculative decoding 的插件兼容层，不等价于主模型普通 greedy/sampling 路径。普通非 MTP 推理仍走 `Qwen3NextForCausalLM` 主模型。

## 当前 `0.18.1` 排查结论

`0.18.1` 适配过程中，一开始出现过“服务能起但输出语义明显异常”的现象。通过与 `0.11.0 / 0.15.1` 在同一 quick checkpoint、同一 prompt 下做对照，最终定位并修复了两类关键缺口：

- GDN 路径缺少 `rnn_position_embedding_type`、`gdn_positions` 透传和 rotary 应用
- 最终 `norm` 需要和 `0.15.1` 一样固定回 `RMSNorm`

修复后，`0.18.1` 在 quick checkpoint 上的输出已经和 `0.15.1` 对齐：

- `temperature=0`：可稳定返回 `2 + 3 = 5`
- `temperature=0.7`：采样行为与 `0.15.1` 一致

这说明当前 `0.18.1` 的 quick checkpoint 主路径已经恢复到正确语义，不再只是“能返回文本”。

另外，本轮还补做了两组版本对照：

- `0.11.0 vs 0.15.1`
  - quick checkpoint 下行为一致
- `0.15.1 vs 0.18.1`
  - quick checkpoint 下行为一致
  - strict `hybrid gdn + moe` checkpoint 下也已回到同一行为层级

因此当前没有证据表明 `0.18.1` 还存在独有的 strict MoE/GDN 语义退化；此前最明显的 `0.18.1` 异常，已经被定位并修复为 quick checkpoint 主路径上的 GDN 位置编码 / final norm 迁移缺口。

## 当前目录状态

目前仓库仍是旧结构：

- `vllm_qwen3_next_plugin/__init__.py`
- `vllm_qwen3_next_plugin/vllm_qwen3_next_0_11_0.py`
- `vllm_qwen3_next_plugin/vllm_qwen3_next_0_15_1.py`
- 对应的 `config` / `mtp` 版本文件

这也是本轮重构的起点，而不是目标终态。
