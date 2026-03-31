# vLLM Qwen3 Next 多版本插件适配实施计划

**Goal:** 将当前旧式平铺结构的 `vllm_qwen3_next_plugin` 重构为单仓库多版本插件工程，并为后续 `vllm 0.15.1` 与 `vllm 0.18.1` 适配建立文档、验收与 skill 基线。

**Architecture:** 先把仓库重构为 `upstream / variants / compat / versioning / dispatcher / docs / tests / scripts` 的工程结构，再逐项迁移旧实现，最后进入 `0.18.1` 的事实上游采集与兼容判断。

**Tech Stack:** Python 3.12、vLLM、PyTorch、Transformers、`uv pip`、editable install、本地 OpenAI API server 验收

---

### Task 1: 固化重构前服务基线

**Files:**
- Modify: `README.md`
- Create: `progress.md`

- [ ] Step 1: 将 `0.11.0` serve 验证结果写入文档
- [ ] Step 2: 明确后续重构的最低回归标准是“仍能 serve 并返回自然语言”
- [ ] Step 3: 记录当前测试 checkpoint 路径、入口环境变量和 GPU 使用约定

### Task 2: 建立多版本设计与计划文档

**Files:**
- Create: `docs/superpowers/specs/2026-03-31-vllm-multi-version-plugin-design.md`
- Create: `docs/superpowers/plans/2026-03-31-vllm-multi-version-plugin.md`

- [ ] Step 1: 明确仓库分层目标
- [ ] Step 2: 明确公共 custom 与版本 compat 的分类原则
- [ ] Step 3: 明确 `vllm 0.18.1` 的后续适配顺序

### Task 3: 建立工程目录骨架

**Files:**
- Create: `vllm_qwen3_next_plugin/versioning.py`
- Create: `vllm_qwen3_next_plugin/qwen3_next.py`
- Create: `vllm_qwen3_next_plugin/mtp.py`
- Create: `vllm_qwen3_next_plugin/config.py`
- Create: `vllm_qwen3_next_plugin/upstream/__init__.py`
- Create: `vllm_qwen3_next_plugin/variants/__init__.py`
- Create: `vllm_qwen3_next_plugin/compat/__init__.py`

- [ ] Step 1: 建立轻量分发器
- [ ] Step 2: 建立版本键和模块映射
- [ ] Step 3: 保持旧入口兼容，避免一开始就破坏 `register()`

### Task 4: 迁移旧版本实现到新结构

**Files:**
- Create: `vllm_qwen3_next_plugin/upstream/vllm_0_11_0/...`
- Create: `vllm_qwen3_next_plugin/upstream/vllm_0_15_1/...`
- Create: `vllm_qwen3_next_plugin/variants/vllm_0_11_0.py`
- Create: `vllm_qwen3_next_plugin/variants/vllm_0_15_1.py`
- Create: `vllm_qwen3_next_plugin/variants/mtp_vllm_0_11_0.py`
- Create: `vllm_qwen3_next_plugin/variants/mtp_vllm_0_15_1.py`
- Create: `vllm_qwen3_next_plugin/variants/config_vllm_0_11_0.py`
- Create: `vllm_qwen3_next_plugin/variants/config_vllm_0_15_1.py`

- [ ] Step 1: 先搬迁文件，不先改逻辑
- [ ] Step 2: 确保 `0.11.0` 分发行为不变
- [ ] Step 3: 对照 `CHANGELOG.md` 开始标记公共 custom

### Task 5: 增补 tests 与 scripts 结构

**Files:**
- Create: `tests/test_versioning.py`
- Create: `tests/test_dispatcher_exports.py`
- Create: `tests/test_register_smoke.py`
- Create: `scripts/check_plugin_import.py`
- Create: `scripts/run_acceptance.py`
- Create: `scripts/validate_generation.py`

- [ ] Step 1: 建立轻量导入和分发测试
- [ ] Step 2: 建立 `register()` 冒烟测试
- [ ] Step 3: 建立真实服务生成脚本骨架

### Task 6: 新增项目级 skill

**Files:**
- Create: `.codex/skills/vllm-plugin-version-adaptation/SKILL.md`

- [ ] Step 1: 记录新增 `vllm` 版本前必须采集的环境事实
- [ ] Step 2: 记录如何抽取上游基线与区分 custom / compat
- [ ] Step 3: 记录必须通过的轻量验收与服务级验收

### Task 7: 准备 `vllm 0.15.1` 与 `vllm 0.18.1` 适配

**Files:**
- Modify: `progress.md`
- Modify: `CHANGELOG.md`

- [ ] Step 1: clone `0.15.1` 本地测试环境
- [ ] Step 2: 采集 `0.15.1` 上游文件路径和接口事实
- [ ] Step 3: clone 独立测试环境并安装 `vllm==0.18.1`
- [ ] Step 4: 采集 `0.18.1` 上游文件路径和接口事实
- [ ] Step 5: 在重构后的结构上开始新增 `0.18.1` 版本目录
