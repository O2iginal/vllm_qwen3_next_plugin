import importlib


def test_package_exports_plugin_entrypoints() -> None:
    plugin = importlib.import_module("vllm_qwen3_next_plugin")

    assert plugin.__all__ == ["Qwen3NextForCausalLM", "register"]
    assert plugin.Qwen3NextForCausalLM.__module__.startswith(
        "vllm_qwen3_next_plugin."
    )
    assert callable(plugin.register)


def test_register_applies_runtime_compat_patches(monkeypatch) -> None:
    plugin = importlib.import_module("vllm_qwen3_next_plugin")

    called = {"count": 0}

    def fake_apply_runtime_patches() -> None:
        called["count"] += 1

    monkeypatch.setattr(plugin, "Qwen3NextForCausalLM", object())
    monkeypatch.setattr(plugin, "apply_runtime_patches", fake_apply_runtime_patches)

    class _FakeRegistry:
        calls = []

        @classmethod
        def register_model(cls, name, model):
            cls.calls.append((name, model))

    class _FakeConfigsModule:
        Qwen3NextConfig = None

    import sys
    import types

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.18.1"
    fake_vllm.ModelRegistry = _FakeRegistry

    fake_config_module = types.ModuleType("vllm_qwen3_next_plugin.config")
    fake_config_module.Qwen3NextConfig = object()

    fake_mtp_module = types.ModuleType("vllm_qwen3_next_plugin.mtp")
    fake_mtp_module.Qwen3NextMTP = object()

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.configs", _FakeConfigsModule)
    monkeypatch.setitem(sys.modules, "vllm_qwen3_next_plugin.config", fake_config_module)
    monkeypatch.setitem(sys.modules, "vllm_qwen3_next_plugin.mtp", fake_mtp_module)

    plugin.register()

    assert called["count"] == 1
