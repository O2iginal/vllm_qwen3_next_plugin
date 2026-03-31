# vLLM Qwen3 Next 多版本插件适配设计文档

## 目标

将当前仍采用旧式平铺结构的 `vllm_qwen3_next_plugin` 重构为可维护的多版本插件工程，并在此基础上继续适配更新的 `vllm` 版本。

本轮设计目标有三个：

- 把当前仓库整理成与 `sglang_qwen3_next_plugin` 对齐的工程结构
- 显式拆分“checkpoint 公共 custom”和“版本 compat”
- 为后续 `vllm 0.18.1` 适配建立稳定的升级流程与项目级 skill

## 已知前提

- 当前仓库的旧结构已经在 `vllm 0.11.0` 上完成真实服务级验证
- 该验证不是只看导入，而是：
  - 启动 `python -m vllm.entrypoints.openai.api_server`
  - 加载真实 checkpoint
  - 发送请求并返回可读自然语言
- 用户已明确说明：
  - 当前 FLA 的 `head_first / tensor shape format mismatch` warning 可忽略
  - 原因是 prompt 太短导致 `seq len` 维度过小
- 用户已指定后续目标新版本为：
  - `vllm 0.18.1`
- 用户后续明确：
  - 放弃 `0.10.2`
  - 最终仅支持 `0.11.0 / 0.15.1 / 0.18.1`
- 用户要求：
  - 先重构，再适配新版本

## 范围

本轮工作包括：

- 建立多版本插件工程结构
- 增补中文工程文档
- 增补测试与验收脚本结构
- 新增项目级 skill
- 以重构后的结构为基础，准备后续 `vllm 0.18.1` 适配

本轮工作暂不包括：

- 立即把所有旧版实现完全迁移完毕
- 立即切换生产默认环境到 `vllm 0.18.1`
- 在没有真实 serve 验证之前宣称新版本适配完成

## 总体方案

### 1. 仓库分层

目标结构：

```text
vllm_qwen3_next_plugin/
├── __init__.py
├── versioning.py
├── qwen3_next.py
├── mtp.py
├── config.py
├── compat/
├── upstream/
│   ├── vllm_0_11_0/
│   ├── vllm_0_15_1/
│   └── vllm_0_18_1/
└── variants/
    ├── vllm_0_11_0.py
    ├── vllm_0_15_1.py
    ├── vllm_0_18_1.py
    ├── mtp_vllm_0_11_0.py
    ├── mtp_vllm_0_15_1.py
    ├── mtp_vllm_0_18_1.py
    ├── config_vllm_0_11_0.py
    ├── config_vllm_0_15_1.py
    └── config_vllm_0_18_1.py
```

说明：

- `upstream/`
  - 保存真实上游快照，作为升级对照基线
- `variants/`
  - 保存版本化的插件实现
- `compat/`
  - 保存某个 `vllm` 版本独有的运行时兼容补丁
- `versioning.py`
  - 统一维护版本键、模块映射、路径映射与支持矩阵
- `qwen3_next.py / mtp.py / config.py`
  - 作为轻量分发器

### 2. 补丁分类原则

当前仓库里的修改不能继续按“都塞在版本文件里”管理，必须重分成两类。

#### 公共 custom

这类逻辑应优先视为与 checkpoint 结构绑定，跨版本都可能保留：

- `num_experts=0` 的 fallback
- `qkv` bias / `o_proj` 无 bias
- `attn_output_gate=False`
- `norm_type`
- `attn_qk_norm`
- token shift
- attention / GDN 分离位置编码
- attention logits scaling
- `gated_rms`

#### 版本 compat

这类逻辑只在某个 `vllm` 版本存在真实缺口时才允许保留：

- 上游 config 没有正确消费 checkpoint 字段
- 某个 runtime API 或 registry 发生变化
- 某个版本的 FLA / hybrid / compile / config 装配路径与旧版不兼容
- 某个版本需要额外注入或 monkey patch 才能跑通

原则：

- 先证明必要性，再保留 compat
- 不能因为旧版本有 patch，就默认新版本也继承

### 3. 新版本适配策略

后续 `vllm 0.18.1` 适配按以下顺序推进：

1. 克隆一份独立虚拟环境
2. 卸载旧 `vllm`
3. 使用用户指定方式安装 `vllm==0.18.1`
4. 采集真实环境事实：
   - `vllm.__version__`
   - 上游模型文件路径
   - 上游 config 文件路径
   - API server 路径
   - 相关接口签名
5. 抽取 `0.18.1` 上游基线
6. 先只迁移公共 custom
7. 仅在真实 serve 失败时再补最小 compat
8. 以真实服务级生成结果作为最终验收依据

### 4. 文档与验收组织

后续仓库至少应包含：

- `README.md`
  - 当前支持状态
  - 目录结构
  - 使用方式
  - 验收方法
- `progress.md`
  - 每轮关键事实和阶段结论
- `CHANGELOG.md`
  - 当前插件相对上游的偏差清单
- `docs/superpowers/specs/`
  - 设计文档
- `docs/superpowers/plans/`
  - 实施计划

### 5. 验收设计

验收必须分两级。

#### 轻量验收

- 包导入
- 版本分发
- `register()` 行为
- config 注入

#### 服务级验收

- 能启动 OpenAI API server
- 能加载真实 checkpoint
- `/health` 正常
- 至少一条中文请求返回可读自然语言
- 至少一条短英文请求返回合理文本

在此基础上，再补：

- 更系统的生成验收脚本
- 可选的 logprob / 质量验收脚本

## 风险

- `vllm 0.18.1` 的模型文件布局可能已经变化
- 当前旧补丁里有一部分可能其实是历史版本绕错逻辑，而不是 checkpoint 公共 custom
- 如果直接把旧 `0.11.0` patch 全量搬到 `0.18.1`，很容易引入无谓 compat
- 当前仓库没有成型的测试与文档骨架，若不先补齐，后续迭代会继续失控

## 原则

- 以真实上游代码和真实服务结果为准
- 先重构工程，再升级版本
- 公共 custom 与版本 compat 必须显式分层
- 每完成一个主要阶段都更新中文文档
- 项目级 skill 必须记录真实工作流，不写空泛原则
