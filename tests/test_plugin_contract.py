import importlib


def test_package_exports_plugin_entrypoints() -> None:
    plugin = importlib.import_module("vllm_qwen3_next_plugin")

    assert plugin.__all__ == ["Qwen3NextForCausalLM", "register"]
    assert plugin.Qwen3NextForCausalLM.__module__.startswith(
        "vllm_qwen3_next_plugin."
    )
    assert callable(plugin.register)
