# vLLM Qwen3 Next Plugin 进度记录

## 2026-03-31

### Phase 0: 当前背景恢复

- 用户目标已经明确：
  - 参考 `sglang_qwen3_next_plugin` 的重构结果，对当前 `vllm` 插件仓库做工程化重构
  - 在重构完成后再考虑适配更新的 `vllm` 版本
  - 为未来新增 `vllm` 版本支持补齐项目级 skill
- 用户后续确认：
  - 不再保留 `vllm 0.10.2`
  - 最终目标支持版本为 `0.11.0 / 0.15.1 / 0.18.1`
- 当前 `vllm` 仓库仍是旧式平铺结构：
  - 模型、config、MTP 都直接放在包根
  - 还没有 `versioning / upstream / variants / compat / tests / docs / skill` 这些分层
- 当前最重要的前提不是“代码能 import”，而是“原插件确实能真实服务并返回自然语言”

### Phase 1: 重构前服务基线验证

- 已新建仓库内测试环境：
  - `/mnt/ssd/yulan/cache/packages/vllm_qwen3_next_plugin-dev/.venv-vllm0.11.0`
- 环境处理方式：
  - 使用 `virtualenv-clone` 从既有 `0.11.0` 环境复制
  - 使用 `uv pip install -e .` 将插件重新绑定到当前 dev 仓库
- 已验证基础契约：
  - `import vllm_qwen3_next_plugin` 成功
  - 当前命中版本为 `vllm 0.11.0`
  - `register()` 会将：
    - `Qwen3NextForCausalLM`
    - `Qwen3NextMTP`
    - `Qwen3NextConfig`
    注册到当前插件实现

### Phase 2: 真实 serve 验收

- 验证 GPU：
  - `CUDA_VISIBLE_DEVICES=4,5`
- 验证入口：
  - `VLLM_PLUGINS=register_qwen3_next_model`
  - `python -m vllm.entrypoints.openai.api_server`
- 验证 checkpoint：
  - `/mnt/ssd/yulan/cache/models/O2iginal/all_mcore_checkpoints/Dist-mathcode10b-s1randg-sch1-CPT-200b-stage2-r640k-GDN2.9b-A7-12_20_21_23_46_48_49-sl32768bs128lr2e5-2e5/merged_10ckpts_iter_38144-hf_to_iter_47683-hf_mean`
- checkpoint 关键事实：
  - `architectures = ["Qwen3NextForCausalLM"]`
  - 显式提供 `layer_types`
  - `num_experts = 0`
  - 含 `attention_bias=true`
  - 含 `attn_output_gate=false`
- 实测结论：
  - 服务成功启动
  - `/health` 返回 `200`
  - 中文请求可返回可读自然语言
  - 英文短问题 `What is 2 + 3? Answer briefly.` 返回 `2 + 3 = 5`
- 当前已知现象：
  - FLA 路径会打印 `head_first / tensor shape format mismatch` warning
  - 用户已明确说明：该 warning 可忽略，原因是 prompt 太短导致 `seq len` 维度过小，不视为当前阻塞项

### 当前阶段结论

- 当前旧仓库虽然结构不规范，但已经具备真实服务级可用性
- 后续重构不能退化到“只剩导入成功”
- 重构后的最小回归目标应至少包括：
  - 插件导入成功
  - `register()` 接管成功
  - API server 启动成功
  - 真实 checkpoint 生成自然语言成功

### 下一步

- 参考 `sglang_qwen3_next_plugin`，先建立中文工程文档与多版本重构设计
- 再把仓库改造成：
  - `upstream/`
  - `variants/`
  - `compat/`
  - `versioning.py`
- 文档和 skill 成型后，再开始准备新版 `vllm` 适配事实采集
- 用户已指定后续目标版本为：
  - `vllm 0.18.1`
  - 该版本将在重构完成后再进入正式适配
- 用户补充说明：
  - `0.15.1` 环境可直接从 opencompass 路径下 clone
  - 当前已确认源环境：
    - `/mnt/ssd/yulan/pretrain-linear-moe/evaluation/opencompass/.venv-vllm0.15.1`
  - 当前更贴近真实使用的 serve 模板为：
    - `vllm serve`
    - `CUDA_VISIBLE_DEVICES=5,7`
    - `VLLM_ATTENTION_BACKEND=FLASHINFER`
    - `VLLM_USE_V1=1`
    - `VLLM_ENGINE=1`
    - `--trust-remote-code`
    - `--enable-expert-parallel`
    - `--tensor-parallel-size 2`

### Phase 3: 文档、skill 与验收骨架落地

- 已新增中文工程文档：
  - `README.md`
  - `progress.md`
  - `docs/superpowers/specs/2026-03-31-vllm-multi-version-plugin-design.md`
  - `docs/superpowers/plans/2026-03-31-vllm-multi-version-plugin.md`
- 已新增项目级 skill：
  - `.codex/skills/vllm-plugin-version-adaptation/SKILL.md`
- 已新增最小验收脚本：
  - `scripts/check_plugin_import.py`
  - `scripts/validate_generation.py`
  - `scripts/run_acceptance.py`
- 已新增最小测试：
  - `tests/test_plugin_contract.py`
  - `tests/test_versioning.py`
  - `tests/test_dispatcher_exports.py`
  - `tests/test_skill_layout.py`
- 当前本地验证结果：
  - `python -m pytest tests -q`
  - 结果：`12 passed`
  - `python scripts/check_plugin_import.py`
  - 结果：通过

### Phase 4: 非破坏性结构迁移启动

- 已新增轻量分发与版本元数据文件：
  - `vllm_qwen3_next_plugin/versioning.py`
  - `vllm_qwen3_next_plugin/qwen3_next.py`
  - `vllm_qwen3_next_plugin/mtp.py`
  - `vllm_qwen3_next_plugin/config.py`
- 已建立目标目录骨架：
  - `vllm_qwen3_next_plugin/variants/`
  - `vllm_qwen3_next_plugin/compat/`
  - `vllm_qwen3_next_plugin/upstream/`
- 当前策略：
  - 暂时不直接搬动旧模型实现文件
  - 先通过 `variants/` wrapper 复用原有实现
  - 先把分发层与目录结构稳定下来，再继续做文件迁移
- 已验证结论：
  - `0.11.0` 下的 `register()` 行为仍保持正常
  - 新增 dispatcher/versioning 层没有破坏当前旧基线

### 当前阶段结论

- 现在仓库已经从“只有旧平铺实现”推进到“有文档、有 skill、有 tests、有 scripts、有 dispatcher 骨架”
- 这一步仍然保住了当前 `0.11.0` 的轻量回归基线
- 下一阶段可以开始真正把旧实现向 `variants/` 和 `upstream/` 迁移

### Phase 5: `0.11.0` 真正迁入 `variants/`

- 已完成：
  - `vllm_qwen3_next_plugin/variants/vllm_0_11_0.py`
  - `vllm_qwen3_next_plugin/variants/mtp_vllm_0_11_0.py`
  - `vllm_qwen3_next_plugin/variants/config_vllm_0_11_0.py`
  现在不再只是 wrapper，而是承载了真实 `0.11.0` 实现
- 包根旧文件现在改为兼容 shim：
  - `vllm_qwen3_next_0_11_0.py`
  - `vllm_qwen3_next_mtp_0_11_0.py`
  - `vllm_qwen3_next_config_0_11_0.py`
- 已新增 `0.11.0` upstream 基线：
  - `vllm_qwen3_next_plugin/upstream/vllm_0_11_0/qwen3_next.py`
  - `vllm_qwen3_next_plugin/upstream/vllm_0_11_0/qwen3_next_mtp.py`
  - `vllm_qwen3_next_plugin/upstream/vllm_0_11_0/qwen3_next_config.py`
- 轻量回归结果：
  - `python -m pytest tests -q`
  - 结果：`12 passed`
  - `python scripts/check_plugin_import.py`
  - 结果：通过
- 阶段性 serve 回归结果：
  - 服务成功启动
  - `/health` 返回 `200`
  - 请求 `What is 2 + 3? Answer briefly.`
  - 返回 `2 + 3 = 5`

### 当前策略更新

- 后续不再为 `0.10.2` 投入重构和验证成本
- 后续重构和环境准备仅围绕：
  - `0.11.0`
  - `0.15.1`
  - `0.18.1`

### Phase 6: `0.15.1` 环境恢复与结构迁移

- 已确认 opencompass 源环境：
  - `/mnt/ssd/yulan/pretrain-linear-moe/evaluation/opencompass/.venv-vllm0.15.1`
- 已克隆本地测试环境：
  - `/mnt/ssd/yulan/cache/packages/vllm_qwen3_next_plugin-dev/.venv-vllm0.15.1`
- 环境事实：
  - `vllm 0.15.1`
  - `torch 2.9.1+cu128`
  - 插件 editable 已重新绑定到当前 dev 仓库
- 中途发现的关键问题：
  - 原 `pyproject.toml` 依赖仍写成 `vllm>=0.10.2,<=0.11.0`
  - 这会导致在新环境执行 editable 安装时，把 `0.15.1` 错误降回 `0.11.0`
- 已实施修复：
  - 将依赖范围调整为：
    - `vllm>=0.11.0,<0.19`
  - 后续在新版本环境里统一使用：
    - `uv pip install -e . --no-deps`
    避免 editable 安装再次改写 `vllm` 主版本
- 已完成 `0.15.1` upstream 基线抽取：
  - `vllm_qwen3_next_plugin/upstream/vllm_0_15_1/qwen3_next.py`
  - `vllm_qwen3_next_plugin/upstream/vllm_0_15_1/qwen3_next_mtp.py`
  - `vllm_qwen3_next_plugin/upstream/vllm_0_15_1/qwen3_next_config.py`
- 已完成 `0.15.1` 实现迁入 `variants/`：
  - `vllm_qwen3_next_plugin/variants/vllm_0_15_1.py`
  - `vllm_qwen3_next_plugin/variants/mtp_vllm_0_15_1.py`
  - `vllm_qwen3_next_plugin/variants/config_vllm_0_15_1.py`
- 当前状态：
  - `0.15.1` 已具备独立测试环境
  - 已具备真实上游基线
  - 已具备结构迁移后的 variant 文件
  - 下一步进入 `0.15.1` 轻量验证与阶段性 serve 验证

### Phase 7: `0.15.1` 轻量验证通过，服务级验证暴露真实运行时问题

- `0.15.1` 轻量验证结果：
  - `import vllm_qwen3_next_plugin` 成功
  - 当前命中：
    - `vllm_qwen3_next_plugin.variants.vllm_0_15_1`
    - `vllm_qwen3_next_plugin.variants.mtp_vllm_0_15_1`
    - `vllm_qwen3_next_plugin.variants.config_vllm_0_15_1`
  - `python scripts/check_plugin_import.py`
    - 结果：通过
- `0.15.1` 环境问题排查：
  - 初次 serve 失败不是插件逻辑，而是本地 clone 环境缺少 API server 依赖
  - 缺失包包括：
    - `fastapi`
    - `pydantic`
    - `ray`
    - `model-hosting-container-standards`
  - 已按 opencompass 源环境版本补齐：
    - `fastapi==0.120.3`
    - `pydantic==2.12.5`
    - `ray==2.51.0`
    - `model-hosting-container-standards==0.1.13`
- 修复依赖后，`0.15.1` 服务已能继续向前推进到：
  - API server 正常启动流程
  - engine core 初始化
  - worker 加载 checkpoint
  - profile run / dummy run
- 当前真实阻塞点：
  - 服务级启动最终失败于 dtype mismatch
  - 错误信息：
    - `expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`
  - 堆栈命中：
    - `vllm_qwen3_next_plugin.variants.vllm_0_15_1`
- 当前结论：
  - `0.15.1` 的障碍已经从“环境缺包”收敛为“插件在该版本上的真实运行时适配问题”
  - 这正是下一步需要处理的核心适配项
  - 但后续复核还需要进一步向用户提供的真实 `vllm serve` 命令模板靠拢，避免只在最小 `api_server` 路径上做判断

### Phase 8: `0.15.1` runtime dtype 修复并恢复服务级验证

- 已定位的关键问题：
  - `0.15.1` 在服务级 profile run 中出现：
    - `expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`
  - 结合堆栈和源码排查，发现 `Qwen3NextGatedDeltaNet.norm` 使用的是：
    - `dtype=config.dtype`
  - 但当前 checkpoint 中：
    - `config.dtype = float32`
  - 实际运行时模型 dtype 则已被 vLLM 下采样到：
    - `torch.bfloat16`
- 已实施修复：
  - `Qwen3NextGatedDeltaNet.norm` 改为使用真实运行 dtype：
    - `dtype=model_config.dtype if model_config is not None else config.dtype`
  - 同时在 `0.15.1` 的 `Qwen3NextModel` / `Qwen3NextDecoderLayer` 中补了激活 dtype 对齐逻辑，确保 norm 后与运行时 dtype 保持一致
- 修复后 `0.15.1` 服务级验证结果：
  - API server 成功启动
  - `/health` 返回 `200`
  - 请求：
    - `What is 2 + 3? Answer briefly.`
  - 返回：
    - `2 + 3 = 5`
- 当前阶段结论：
  - `0.15.1` 已恢复到和 `0.11.0` 相同等级的阶段性 serve 基线
  - 后续可继续用旧 checkpoint 做快速回归
  - 再切换到用户给出的更严格 `hybrid gdn + moe` checkpoint 做最终测试

### Phase 9: `hybrid gdn + moe` 严格测试通过

- 最终严格测试使用的 checkpoint：
  - `/mnt/ssd/yulan/pretrain-linear-moe-dev-worktree/YuLan-Pretrain-gyzp_mom/cache/exports/yulan_hybrid_gdn_maskmoe_to_moe_iter17163/iter_release-hf`
- 关键配置特征：
  - `num_experts = 192`
  - `num_experts_per_tok = 2`
  - `mlp_only_layers = [0, 1]`
  - 显式 `layer_types`
- 严格测试命令模板对齐用户真实使用方式：
  - `vllm serve`
  - `CUDA_VISIBLE_DEVICES=4,5`
  - `VLLM_ATTENTION_BACKEND=FLASHINFER`
  - `VLLM_USE_V1=1`
  - `VLLM_ENGINE=1`
  - `--enable-expert-parallel`
  - `--tensor-parallel-size 2`
- 本阶段先后暴露并修复的真实问题：
  1. `shared_expert_gate.weight` 不应在 `shared_expert_intermediate_size == 0` 的 checkpoint 上强制创建和加载
  2. `SharedFusedMoE` 在当前 `0.15.1` 路径下返回 `(shared_out, fused_out)` 二元组，`Qwen3NextSparseMoeBlock.forward()` 不能把它当成单一 `Tensor`
  3. `Qwen3NextGatedDeltaNet.norm` 的 dtype 必须使用真实运行时 `model_config.dtype`，不能直接使用 checkpoint 中的 `config.dtype=float32`
- 修复后最终严格测试结果：
  - `vllm serve` 成功启动
  - `/health` 返回 `200`
  - completion 请求成功
  - 返回可读自然语言片段，例如：
    - `To find the value of (2 + 3 ...`

### 当前阶段结论

- 当前仓库已经完成：
  - `0.11.0` 快速 checkpoint 服务级验证
  - `0.15.1` 快速 checkpoint 服务级验证
  - `0.15.1` 严格 `hybrid gdn + moe` 服务级验证
- 这说明本轮重构没有退化，且已经覆盖：
  - `num_experts = 0`
  - `num_experts > 0`
  - `enable_expert_parallel`
  - `hybrid gdn + moe`
- 下一步可以把重心切换到：
  - 清理和固化当前重构结果
  - 准备 `0.18.1` 环境与适配

### Phase 10: `0.18.1` 环境与事实上游采集

- 已完成本地环境准备：
  - `/mnt/ssd/yulan/cache/packages/vllm_qwen3_next_plugin-dev/.venv-vllm0.18.1`
- 安装方式：
  - 使用用户指定命令：
    - `uv pip install vllm==0.18.1 --extra-index-url https://wheels.vllm.ai/0.18.1/ --torch-backend=auto`
  - 再执行：
    - `uv pip install -e . --no-deps`
- 当前环境事实：
  - `torch 2.10.0+cu128`
  - `vllm 0.18.1`
- 已采集上游路径：
  - `vllm/model_executor/models/qwen3_next.py`
  - `vllm/model_executor/models/qwen3_next_mtp.py`
  - `vllm/transformers_utils/configs/qwen3_next.py`
  - `vllm/entrypoints/openai/api_server.py`
- 已建立 `0.18.1` upstream 基线：
  - `vllm_qwen3_next_plugin/upstream/vllm_0_18_1/qwen3_next.py`
  - `vllm_qwen3_next_plugin/upstream/vllm_0_18_1/qwen3_next_mtp.py`
  - `vllm_qwen3_next_plugin/upstream/vllm_0_18_1/qwen3_next_config.py`
- 已确认的第一层适配结论：
  - 不能直接复用 `0.15.1` 实现
  - 原因是 `0.18.1` 中旧实现依赖的模块路径已经变化
  - 当前直接导入会报：
    - `No module named 'vllm.attention'`
  - 说明必须基于 `0.18.1` 的真实 upstream 重新构建 variant，而不是简单 wrapper 到 `0.15.1`

### Phase 11: `0.18.1` 第一层入口打通

- 已完成：
  - 将 `0.18.1` 纳入 `versioning.py`
  - 建立：
    - `variants/vllm_0_18_1.py`
    - `variants/mtp_vllm_0_18_1.py`
    - `variants/config_vllm_0_18_1.py`
  - 将上游文件中的 `.interfaces` / `.utils` 相对导入改为 `vllm` 当前真实模块路径
  - 将 `mtp_vllm_0_18_1.py` 改为引用仓库内 `variants.vllm_0_18_1`
- 当前结果：
  - `.venv-vllm0.18.1` 中：
    - `import vllm_qwen3_next_plugin` 成功
    - `python scripts/check_plugin_import.py` 成功
  - 当前命中：
    - `vllm_qwen3_next_plugin.variants.vllm_0_18_1`
    - `vllm_qwen3_next_plugin.variants.mtp_vllm_0_18_1`
    - `vllm_qwen3_next_plugin.variants.config_vllm_0_18_1`
- 当前结论：
  - `0.18.1` 已从“完全 unsupported”推进到“最小导入 / register 闭环可用”
  - 下一步进入真实 custom 迁移，而不再是入口兼容修补

### 经验总结

- 新版本环境安装后，editable 安装必须优先使用：
  - `uv pip install -e . --no-deps`
  - 否则当前仓库的依赖约束会把新环境错误回滚到旧版 `vllm`
- 从旧环境 clone 出来的新环境不一定拥有完整的 API server 依赖集
  - `0.15.1` 就曾缺少：
    - `fastapi`
    - `pydantic`
    - `ray`
    - `model-hosting-container-standards`
- `config.dtype` 不能等价于真实运行 dtype
  - `0.15.1` 的 `Qwen3NextGatedDeltaNet.norm` 必须使用 `model_config.dtype`
- 在 MoE 路径里，shared expert 相关模块不能无条件创建
  - `shared_expert_intermediate_size == 0` 时不应要求 `shared_expert_gate.weight`
- `SharedFusedMoE` 在不同版本 / 路径下可能返回 tuple，而不是单一 Tensor
  - 不能默认把返回值直接喂给 `all_reduce`
- 升到 `0.18.1` 之后，模块导入路径已经发生变化
  - 不能再把 `0.15.1` 实现简单 wrapper 到 `0.18.1`

### Phase 12: `0.18.1` 公共 custom 开始迁移

- 本轮先完成了不依赖空闲 GPU 的迁移与校验：
  - `variants/config_vllm_0_18_1.py`
    - 补回 checkpoint 侧自定义字段：
      - `ffn_token_shift`
      - `ffn_intermediate_token_shift`
      - `attn_token_shift`
      - `attn_q/k/v_token_shift`
      - `token_shift_conv_size`
      - `token_shift_conv_init`
      - `attn_position_embedding_type`
      - `rnn_position_embedding_type`
      - `attn_logits_scaling`
      - `attn_output_gate`
      - `norm_type`
      - `gated_norm_rank`
      - `gated_norm_gate_scale`
      - `attn_qk_norm`
    - 补回对应校验逻辑
  - `variants/vllm_0_18_1.py`
    - 新增：
      - `GatedRMSNorm`
      - `_get_qwen3_next_norm_cls`
      - `_apply_causal_depthwise_conv1d_bcl`
      - `Qwen3NextMLPWithTokenShift`
    - 已迁移的公共 custom：
      - attention bias / `attn_output_gate`
      - attention token shift
      - attention qk norm
      - attention logits scaling
      - attention rope/nope 开关
      - MLP token shift
      - decoder/model 的 runtime dtype 对齐
      - MoE shared expert 按 `shared_expert_intermediate_size > 0` 条件创建
      - `SharedFusedMoE` tuple 返回值解包
- 本轮非 GPU 校验结果：
  - `python -m pytest tests/test_config_0_18_1_custom_fields.py tests/test_versioning.py -q`
    - 结果：通过
  - `.venv-vllm0.18.1` 中 `python scripts/check_plugin_import.py`
    - 结果：通过
  - `.venv-vllm0.18.1` 中 `python -m py_compile .../vllm_0_18_1.py .../mtp_vllm_0_18_1.py .../config_vllm_0_18_1.py`
    - 结果：通过
- 当前明确未完成项：
  - `0.18.1` 还没有做真实 `serve` 验收
  - 由于当前机器没有空闲卡，本轮到此为止，后续需换机继续
- 换机后的第一优先级：
  - 在 `.venv-vllm0.18.1` 中先用旧 quick checkpoint 做阶段性 `vllm serve` 冒烟
  - 若 quick checkpoint 通过，再切到最终 `hybrid gdn + moe` checkpoint 做严格验收

### Phase 13: `0.18.1` 服务级验证恢复，并定位语义偏差根因

- 已确认的环境问题：
  - clone 出来的 `.venv-vllm0.18.1` 中残留了上游 `flash-attn==2.8.3+cu12torch2.9`
  - 当前环境实际为：
    - `torch 2.10.0+cu128`
    - `vllm 0.18.1`
  - `rotary_embedding/common.py` 会因为 `find_spec("flash_attn")` 命中这个残留包而误走坏路径
  - 具体报错为：
    - `flash_attn_2_cuda ... undefined symbol`
- 已实施修复：
  - 在 `.venv-vllm0.18.1` 中执行：
    - `uv pip uninstall flash-attn`
  - 结论：
    - 该问题是环境污染，不是 `torch 2.10.0` 本身不兼容
    - 当前无需先调整 `torch`
- 随后暴露的 `0.18.1` 插件级问题：
  - quick checkpoint 为 `num_experts=0`
  - `QwenNextMixtureOfExperts.set_moe_parameters()` 还未迁入 `0.15.1` 里的无 MoE 短路逻辑
  - 错误：
    - `RuntimeError: No Qwen3Next layer found in the model.layers.`
- 已实施修复：
  - 在 `variants/vllm_0_18_1.py` 中补回：
    - `config.num_experts == 0` 时直接把各类 MoE 统计字段归零并返回
  - 并新增测试：
    - `tests/test_moe_parameters_0_18_1.py`
- 服务级结果：
  - quick checkpoint 下，`0.18.1` 已可在 `--enforce-eager` 模式成功：
    - `vllm serve`
    - `/health -> 200`
    - completion 请求返回文本
  - 严格 `hybrid gdn + moe` checkpoint 下，`0.18.1` 也已可在：
    - `--enable-expert-parallel`
    - `--enforce-eager`
    模式成功启动服务
    - `/health -> 200`
    - completion / chat completion 请求返回文本
- 但同一阶段发现：
  - `0.18.1` quick checkpoint 在相同 prompt 下返回的文本语义明显偏离 `0.11.0`
  - 这说明问题已经从“服务起不来”推进到“语义正确性有残留偏差”

### Phase 14: 通过 `0.11 / 0.15 / 0.18` 同 prompt 对比定位剩余缺口

- 对比方法：
  - checkpoint：同一个 quick checkpoint
  - prompt：`What is 2 + 3? Answer briefly.`
  - 比较版本：
    - `0.11.0`
    - `0.15.1`
    - `0.18.1`
  - 比较采样：
    - `temperature=0`
    - `temperature=0.7`
- 初始现象：
  - `0.11.0` 在 `temperature=0` 下可返回：
    - `2 + 3 = 5`
  - `0.18.1` 在 `temperature=0` 下却返回明显异常的编号 / 列表结构
  - 说明问题不是单纯的采样随机性，而是 `0.18.1` 适配仍缺少关键语义路径
- 对照 `0.15.1` 后定位出的高优先级缺口：
  - `Qwen3NextGatedDeltaNet` 缺少：
    - `rnn_position_embedding_type`
    - GDN 自身的 `rotary_emb`
  - `Qwen3NextGatedDeltaNet.forward()` 没有把 `positions` 写入 `forward_context.gdn_positions`
  - `_forward_core()` 没有在 GDN 路径上按 positions 应用 rotary
  - `Qwen3NextDecoderLayer.forward()` 调用 `linear_attn` 时没把 `positions` 传进去
  - `Qwen3NextModel` 最终 `norm` 仍是 `Qwen3NextRMSNorm`，而 `0.15.1` 中为了 checkpoint 一致性固定回 `RMSNorm`
- 已实施修复：
  - `variants/vllm_0_18_1.py`
    - 补回 GDN 的 `rnn_position_embedding_type` / `rotary_emb`
    - 补回 `gdn_positions` 的 `getattr/setattr` 透传
    - 补回 `_forward_core()` 中对 spec / non-spec GDN 分支的 rotary 应用
    - `Qwen3NextDecoderLayer.forward()` 调用 `linear_attn` 时补传 `positions`
    - 最终 `norm` 调整为 `RMSNorm`
- 修复后的同 prompt 对比结果：
  - `0.15.1`
    - `temperature=0`：`2 + 3 = 5`
    - `temperature=0.7`：3 次采样结果与历史 quick checkpoint 行为一致
  - `0.18.1`
    - `temperature=0`：现在与 `0.15.1` 对齐，返回 `2 + 3 = 5`
    - `temperature=0.7`：3 次采样也与 `0.15.1` 对齐
- 当前阶段结论：
  - `0.18.1` quick checkpoint 的“语义不通”问题已经被拉回到与 `0.15.1` 同一行为水平
  - 这说明剩余缺口确实在 GDN 位置编码和最终 norm 迁移，而不是模型随机性
  - 后续如果继续追严格 `hybrid gdn + moe` 的输出质量，应在当前修复基础上继续验证 strict checkpoint 的语义，而不必再怀疑 quick checkpoint 的 `0.18.1` 主路径

### Phase 15: 版本两两对照结果固化

- `quick checkpoint`，prompt 固定为：
  - `What is 2 + 3? Answer briefly.`
- `0.11.0` vs `0.15.1`
  - `temperature=0`
    - 两者都返回：
      - `2 + 3 = 5`
  - `temperature=0.7`
    - 两者都出现相同风格的发散采样
    - 典型包括：
      - 正确答到 `5`
      - 漂到其它数学/语言题目
  - 结论：
    - quick checkpoint 下，`0.15.1` 与 `0.11.0` 行为是一致的
- `quick checkpoint`，`0.15.1` vs 修补后的 `0.18.1`
  - `temperature=0`
    - 两者都返回：
      - `2 + 3 = 5`
  - `temperature=0.7`
    - 两者 3 次采样结果对齐
  - 结论：
    - quick checkpoint 下，`0.18.1` 已被拉回到 `0.15.1` 行为水平
- `strict hybrid gdn + moe checkpoint`
  - 比较版本：
    - `0.15.1`
    - 修补后的 `0.18.1`
  - `temperature=0`
    - 两者都返回相同风格文本：
      - 会先写出 `To find the value of (2 + 3)...`
      - 不是最短答案，但语义是正常的数学解释开头
  - `temperature=0.7`
    - 两者也基本对齐
    - sample 2 / sample 3 已一致
    - sample 1 虽然细节文本不同，但都属于采样发散，不再表现出“只有 `0.18.1` 异常”的模式
- 当前阶段结论：
  - `strict checkpoint` 上，修补后的 `0.18.1` 与 `0.15.1` 也已经回到同一行为层级
  - 因此目前没有证据表明：
    - `0.18.1` 还存在独有的 strict MoE/GDN 语义退化
  - 当前更合理的判断是：
    - strict checkpoint 本身在这个 prompt 下就不擅长给“极简直接答案”
    - 先前 `0.18.1` 独有的异常，确实主要来自 quick checkpoint 主路径上的 GDN / final norm 迁移缺口
