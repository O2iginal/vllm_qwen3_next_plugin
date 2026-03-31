#!/usr/bin/env python3
"""
验证当前插件的最小导入与 registry 接管行为。
"""

from __future__ import annotations

import sys


def main() -> int:
    import vllm
    from vllm import ModelRegistry
    import vllm.transformers_utils.configs as configs
    import vllm_qwen3_next_plugin as plugin

    print(f"vllm_version: {vllm.__version__}")
    print(f"plugin_file: {plugin.__file__}")
    print(f"entry_model_module: {plugin.Qwen3NextForCausalLM.__module__}")

    plugin.register()

    registry = getattr(ModelRegistry, "models", None)
    if registry is None:
        registry = getattr(ModelRegistry, "_models", None)
    assert registry is not None, "无法读取 ModelRegistry 中的已注册模型"

    qwen_entry = registry.get("Qwen3NextForCausalLM")
    mtp_entry = registry.get("Qwen3NextMTP")

    assert qwen_entry is not None, "registry 中缺少 Qwen3NextForCausalLM"
    assert mtp_entry is not None, "registry 中缺少 Qwen3NextMTP"

    qwen_cls = getattr(qwen_entry, "model_cls", qwen_entry)
    mtp_cls = getattr(mtp_entry, "model_cls", mtp_entry)

    print(f"registered_model: {qwen_cls}")
    print(f"registered_mtp: {mtp_cls}")
    print(f"config_module: {configs.Qwen3NextConfig.__module__}")

    assert qwen_cls.__module__.startswith("vllm_qwen3_next_plugin"), (
        "Qwen3NextForCausalLM 未指向插件实现"
    )
    assert mtp_cls.__module__.startswith("vllm_qwen3_next_plugin"), (
        "Qwen3NextMTP 未指向插件实现"
    )
    assert configs.Qwen3NextConfig.__module__.startswith("vllm_qwen3_next_plugin"), (
        "Qwen3NextConfig 未指向插件实现"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"ASSERTION FAILED: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {exc}")
        raise SystemExit(1)
