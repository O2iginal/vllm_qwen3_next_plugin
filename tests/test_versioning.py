import pathlib

from vllm_qwen3_next_plugin import versioning


def test_version_key_is_normalized() -> None:
    assert versioning.normalize_version_key("0.11.0") == "vllm_0_11_0"
    assert versioning.normalize_version_key("0.15.1") == "vllm_0_15_1"
    assert versioning.normalize_version_key("0.18.1") == "vllm_0_18_1"


def test_version_key_uses_numeric_prefix_only() -> None:
    assert versioning.normalize_version_key("0.15.1.post1") == "vllm_0_15_1"
    assert versioning.normalize_version_key("0.16.0rc2") == "vllm_0_16_0"


def test_supported_version_key_handles_declared_prefixes() -> None:
    assert versioning.get_supported_version_key("0.11.0") == "vllm_0_11_0"
    assert versioning.get_supported_version_key("0.15.9") == "vllm_0_15_1"
    assert versioning.get_supported_version_key("0.18.7") == "vllm_0_18_1"


def test_supported_version_key_can_use_env_override_for_source_checkout(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_QWEN3_NEXT_PLUGIN_VERSION", "0.11.0")

    assert versioning.get_supported_version_key("dev") == "vllm_0_11_0"


def test_variant_module_names_are_declared() -> None:
    mapping = versioning.get_variant_module_names()

    assert mapping["vllm_0_11_0"] == "vllm_qwen3_next_plugin.variants.vllm_0_11_0"
    assert mapping["vllm_0_15_1"] == "vllm_qwen3_next_plugin.variants.vllm_0_15_1"
    assert mapping["vllm_0_18_1"] == "vllm_qwen3_next_plugin.variants.vllm_0_18_1"


def test_mtp_module_names_are_declared() -> None:
    mapping = versioning.get_mtp_module_names()

    assert (
        mapping["vllm_0_11_0"]
        == "vllm_qwen3_next_plugin.variants.mtp_vllm_0_11_0"
    )
    assert (
        mapping["vllm_0_15_1"]
        == "vllm_qwen3_next_plugin.variants.mtp_vllm_0_15_1"
    )
    assert (
        mapping["vllm_0_18_1"]
        == "vllm_qwen3_next_plugin.variants.mtp_vllm_0_18_1"
    )


def test_config_module_names_are_declared() -> None:
    mapping = versioning.get_config_module_names()

    assert (
        mapping["vllm_0_11_0"]
        == "vllm_qwen3_next_plugin.variants.config_vllm_0_11_0"
    )
    assert (
        mapping["vllm_0_15_1"]
        == "vllm_qwen3_next_plugin.variants.config_vllm_0_15_1"
    )
    assert (
        mapping["vllm_0_18_1"]
        == "vllm_qwen3_next_plugin.variants.config_vllm_0_18_1"
    )


def test_upstream_variant_paths_are_declared() -> None:
    mapping = versioning.get_upstream_variant_paths(pathlib.Path.cwd())

    assert mapping["vllm_0_15_1"].name == "qwen3_next.py"
    assert mapping["vllm_0_18_1"].name == "qwen3_next.py"
