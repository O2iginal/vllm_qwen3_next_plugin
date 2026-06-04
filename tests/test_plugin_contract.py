import importlib
import sys
import types


def test_package_exports_plugin_entrypoints_without_eager_model_import() -> None:
    plugin = importlib.import_module("vllm_qwen3_next_plugin")

    assert plugin.__all__ == ["Qwen3NextForCausalLM", "register"]
    assert callable(plugin.register)
    assert "vllm_qwen3_next_plugin.qwen3_next" not in sys.modules


def test_register_applies_runtime_compat_patches(monkeypatch) -> None:
    plugin = importlib.import_module("vllm_qwen3_next_plugin")

    called = {"count": 0}

    def fake_apply_runtime_patches() -> None:
        called["count"] += 1

    monkeypatch.setattr(plugin, "apply_runtime_patches", fake_apply_runtime_patches)

    class _FakeRegistry:
        calls = []

        @classmethod
        def register_model(cls, name, model):
            cls.calls.append((name, model))

    class _FakeConfigsModule:
        Qwen3NextConfig = None

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.18.1"
    fake_vllm.ModelRegistry = _FakeRegistry
    fake_transformers_utils = types.ModuleType("vllm.transformers_utils")
    fake_transformers_utils.configs = _FakeConfigsModule

    fake_qwen3_next_module = types.ModuleType("vllm_qwen3_next_plugin.qwen3_next")
    fake_qwen3_next_module.Qwen3NextForCausalLM = object()

    fake_config_module = types.ModuleType("vllm_qwen3_next_plugin.config")
    fake_config_module.Qwen3NextConfig = object()

    fake_mtp_module = types.ModuleType("vllm_qwen3_next_plugin.mtp")
    fake_mtp_module.Qwen3NextMTP = object()

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils", fake_transformers_utils)
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.configs", _FakeConfigsModule)
    monkeypatch.setitem(sys.modules, "vllm_qwen3_next_plugin.qwen3_next", fake_qwen3_next_module)
    monkeypatch.setitem(sys.modules, "vllm_qwen3_next_plugin.config", fake_config_module)
    monkeypatch.setitem(sys.modules, "vllm_qwen3_next_plugin.mtp", fake_mtp_module)

    plugin.register()

    assert called["count"] == 1
    assert _FakeRegistry.calls == [
        ("Qwen3NextForCausalLM", fake_qwen3_next_module.Qwen3NextForCausalLM),
        ("Qwen3NextMTP", fake_mtp_module.Qwen3NextMTP),
    ]
