---
name: vllm-plugin-version-adaptation
description: Use when adding or reviewing support for a new vLLM version in this repository, especially when you need to separate checkpoint-specific customizations from version-specific compat, extract upstream baselines, and require real server-level validation with natural-language generation.
---

# vLLM Plugin Version Adaptation

## Overview

这是本仓库给 `vllm` 新版本做插件适配的标准流程。核心原则只有两个：

- 以真实上游代码和真实服务结果为准，不以历史记忆为准
- 先区分“checkpoint 公共 custom”和“版本 compat”，再写代码

## 何时使用

在以下情况必须使用本 skill：

- 仓库要新增一个 `vllm` 版本支持
- 需要判断旧 patch 是否仍然需要
- 需要抽取某个新版本的上游 `qwen3_next` 基线
- 需要把当前仓库从旧平铺结构迁到 `upstream / variants / compat / versioning`
- 需要做真实 `api_server` 级别验收

以下情况不要单独使用本 skill：

- 只是修改无关文档
- 只是改一条 prompt 或注释
- 只是做与版本适配无关的小修复

## 固定流程

### 1. 先采集新环境事实

必须先记录：

- `vllm.__version__`
- `vllm.entrypoints.openai.api_server` 实际路径
- 上游模型文件路径
- 上游 config 文件路径
- 相关 registry / plugin 入口行为
- 关键运行时栈版本：
  - `torch`
  - `transformers`
  - `flashinfer-python`

如果新环境还没有装目标 `vllm`，先克隆现有 venv，再安装目标版本，不要污染已经验证过的旧基线环境。

### 2. 新版本环境一律从 clone 出来

本仓库的约定：

1. 先复制一个已有可用 venv
2. 在复制出来的环境中卸载旧 `vllm`
3. 再安装目标新版本

用户当前指定的安装方式为：

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
version=0.18.1
uv pip install vllm==$version --extra-index-url https://wheels.vllm.ai/$version/ --torch-backend=auto
```

### 3. 抽取上游基线

每新增一个版本，至少复制这些文件到仓库：

- `upstream/vllm_x_y_z/qwen3_next.py`
- `upstream/vllm_x_y_z/qwen3_next_mtp.py`
- `upstream/vllm_x_y_z/qwen3_next_config.py`

不允许只看 site-packages 后直接改插件实现，而不保留上游快照。

### 4. 先判定哪些是公共 custom

优先按 checkpoint 结构归类的候选项：

- `num_experts=0`
- `qkv` bias / `o_proj` 无 bias
- `attn_output_gate`
- `norm_type`
- `attn_qk_norm`
- token shift
- attention / GDN 分离位置编码
- attention logits scaling
- `gated_rms`

这些逻辑只有在新上游已经原生满足 checkpoint 结构时才能删除。

### 5. 再判定哪些是版本 compat

只有在真实新版本存在缺口时，才允许写 `compat/`：

- 配置解析差异
- registry / plugin 入口变化
- runtime API 变化
- compile / hybrid / linear attention 路径差异
- 必须额外 patch 才能跑通的版本问题

compat 必须单独落在：

- `vllm_qwen3_next_plugin/compat/`

不能重新塞回单一大模型文件。

### 6. 保持轻量分发器

仓库角色分工固定为：

- `versioning.py`
  - 版本键、模块映射、路径映射
- `upstream/`
  - 各版本上游快照
- `variants/`
  - 各版本实际插件实现
- `compat/`
  - 各版本特有 compat
- `qwen3_next.py`
  - 主模型分发器
- `mtp.py`
  - MTP 分发器
- `config.py`
  - config 分发器

### 7. 每完成一阶段就更新文档

必须同步更新：

- `README.md`
- `progress.md`
- `CHANGELOG.md`
- `docs/superpowers/specs/...`
- `docs/superpowers/plans/...`

至少记录：

- 当前目标版本
- 当前环境事实
- 哪些旧 patch 属于公共 custom
- 哪些旧 patch 被判定为 compat
- 当前是否已经通过真实 serve 验收

## 最小验收

新增版本支持后，至少要通过以下检查：

1. 轻量检查
   - `python -m pytest tests -q`
   - 导入成功
   - 分发命中正确版本
   - `register()` 冒烟通过
2. 服务级检查
   - 启动 `python -m vllm.entrypoints.openai.api_server`
   - `/health` 正常
   - 对真实 checkpoint 发送请求
   - 返回可读自然语言

至少要能通过：

- 一条中文短 prompt
- 一条英文短 prompt

如果没有真实服务级验证，不得宣布新版本适配完成。

## 当前仓库的已知事实

- 当前旧结构已经在 `vllm 0.11.0` 上通过真实服务级验证
- 当前仓库也已经在 `vllm 0.15.1` 上通过真实服务级验证
- 验证入口为：
  - `VLLM_PLUGINS=register_qwen3_next_model`
  - `python -m vllm.entrypoints.openai.api_server`
- 当前测试时使用：
  - `CUDA_VISIBLE_DEVICES=4,5`
- 当前已验证 checkpoint 具备：
  - `architectures = ["Qwen3NextForCausalLM"]`
  - `layer_types`
  - `num_experts = 0`
- FLA 路径可能出现 `head_first / tensor shape format mismatch` warning
  - 当前用户已明确说明该 warning 可忽略
  - 原因是 prompt 太短导致 `seq len` 维度过小
- 当前后续目标版本为：
  - `vllm 0.18.1`
- 当前更严格的最终测试 checkpoint 为：
  - `hybrid gdn + moe`
  - 已在 `0.15.1` 下通过 `vllm serve` 服务级验证

## 当前仓库的运行时经验

- 新版本环境不要直接 `uv pip install -e .`
  - 应优先使用 `uv pip install -e . --no-deps`
  - 否则当前 `pyproject.toml` 的依赖范围可能把新环境错误回滚到旧版 `vllm`
- clone 得到的环境不一定带齐 API server 依赖
  - 在 `0.15.1` 适配中，曾缺：
    - `fastapi`
    - `pydantic`
    - `ray`
    - `model-hosting-container-standards`
- `config.dtype` 不能当成真实运行 dtype
  - 在 `0.15.1` 中，checkpoint `config.dtype=float32`
  - 但 vLLM 运行时实际已经下采样到 `bfloat16`
  - 若自定义模块直接使用 `config.dtype` 初始化 norm 或线性路径，容易触发 `float != bfloat16`
- MoE 路径里不要无条件创建 shared expert 相关模块
  - 若 `shared_expert_intermediate_size == 0`，不应要求 `shared_expert_gate.weight`
- `SharedFusedMoE` 返回值不能想当然
  - 在 `0.15.1` 的 expert-parallel 路径中，返回值可能是 `(shared_out, fused_out)`
  - 若直接把它当 Tensor 去 `all_reduce`，会在 Dynamo / fake tensor 路径崩溃
- `0.18.1` 的 import path 已经变化
  - 旧实现里的 `vllm.attention...` 不再成立
  - 必须先读取真实 upstream imports，再决定 variant 的移植方式
- `0.18.1` clone 环境里若残留上游 `flash-attn`
  - 即使 `vllm 0.18.1` 自带 `vllm.vllm_flash_attn`
  - `rotary_embedding/common.py` 也可能因 `find_spec("flash_attn")` 误走坏路径
  - 若出现 `flash_attn_2_cuda ... undefined symbol`
    - 优先考虑直接 `uv pip uninstall flash-attn`
    - 不要先急着调整 `torch`
- 若 `0.18.1` quick checkpoint 表现为：
  - `serve` 成功
  - `/health` 正常
  - 但同 prompt 下输出语义明显偏离 `0.11/0.15`
  - 优先检查 GDN 迁移是否缺少：
    - `rnn_position_embedding_type`
    - GDN 自身的 `rotary_emb`
    - `forward_context.gdn_positions` 的写入
    - `_forward_core()` 中对 spec / non-spec GDN 分支应用 rotary
    - `Qwen3NextDecoderLayer.forward()` 是否把 `positions` 传给 `linear_attn`
- `0.18.1` 的最终 `norm` 也要重点核对
  - 当前这套 checkpoint 路径下，行为对齐 `0.15.1` 时仍需要固定回 `RMSNorm`
  - 不能想当然沿用上游默认 final norm
- 当怀疑“新版本语义异常”时，不要只看单版本单次输出
  - 应优先做两组对照：
    - quick checkpoint：`0.11 / 0.15 / 新版本`
    - strict checkpoint：`0.15 / 新版本`
  - 且至少比较：
    - `temperature=0`
    - `temperature=0.7` 多次采样
  - 这样才能区分：
    - 新版本独有回归
    - checkpoint 本身在高温采样下就会发散

## 常见错误

- 还没重构结构，就直接往旧平铺文件里继续叠 patch
- 不保留上游快照，只在插件文件里硬改
- 只做导入验证，不做真实 server 验证
- 只看 `/health`，不发真实请求
- 新版本一失败就默认把旧 patch 全量搬过去
- 修改代码后不更新 `progress.md` 和设计文档
