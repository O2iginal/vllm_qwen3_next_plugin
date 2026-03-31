# vLLM Qwen3 Next Plugin

这是一个用于替换 `Qwen3NextForCausalLM` 实现的 vLLM 外部插件仓库。目标不是只做可导入的 demo，而是让我们自定义架构的 Qwen3 Next checkpoint 能在不同版本的 `vllm` 中稳定完成真实推理。

当前仓库仍在重构与多版本适配过程中，但已经确认：

- `0.11.0` 可完成真实服务级推理
- `0.15.1` 可完成真实服务级推理
- `0.15.1` 已通过更严格的 `hybrid gdn + moe` checkpoint 服务级验证
- `0.18.1` 已完成 quick checkpoint 的真实服务级验证
- `0.18.1 + hybrid gdn + moe` 已完成严格 `vllm serve` 启动与请求返回验证

## 当前已验证基线

- 已验证环境：`vllm 0.11.0`
- 已验证环境：`vllm 0.15.1`
- 已验证环境：`vllm 0.18.1`
- 已验证入口：`VLLM_PLUGINS=register_qwen3_next_model`
- 已验证能力：
  - 插件导入成功
  - `register()` 成功接管 `Qwen3NextForCausalLM`、`Qwen3NextMTP`、`Qwen3NextConfig`
  - 能真实启动 `python -m vllm.entrypoints.openai.api_server`
  - 能对测试 checkpoint 返回可读自然语言
  - 能对更严格的 `hybrid gdn + moe` checkpoint 返回可读自然语言

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
