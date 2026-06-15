from __future__ import annotations

import os
import pathlib
import re


SUPPORTED_VERSION_KEYS = {
        "0.20.2": "vllm_0_20_2",
    "0.11.0": "vllm_0_11_0",
    "0.15.1": "vllm_0_15_1",
    "0.18.1": "vllm_0_18_1",
}

SUPPORTED_VERSION_PREFIXES = {
        "0.20": "vllm_0_20_2",
    "0.11.0": "vllm_0_11_0",
    "0.15": "vllm_0_15_1",
    "0.18": "vllm_0_18_1",
}


def normalize_version_key(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"unsupported vllm version format: {version}")
    major, minor, patch = match.groups()
    return f"vllm_{major}_{minor}_{patch}"


def get_supported_version_key(version: str) -> str:
    override = os.getenv("VLLM_QWEN3_NEXT_PLUGIN_VERSION")
    if override:
        version = override
    for prefix, version_key in SUPPORTED_VERSION_PREFIXES.items():
        if version.startswith(prefix):
            return version_key
    raise ValueError(
        f"unsupported vllm version: {version}. "
        "Supported versions: 0.11.0, 0.15.x, 0.18.x"
    )


def get_variant_module_names() -> dict[str, str]:
    return {
        "vllm_0_20_2": "vllm_qwen3_next_plugin.variants.vllm_0_20_2",
        "vllm_0_11_0": "vllm_qwen3_next_plugin.variants.vllm_0_11_0",
        "vllm_0_15_1": "vllm_qwen3_next_plugin.variants.vllm_0_15_1",
        "vllm_0_18_1": "vllm_qwen3_next_plugin.variants.vllm_0_18_1",
    }


def get_mtp_module_names() -> dict[str, str]:
    return {
        "vllm_0_20_2": "vllm_qwen3_next_plugin.variants.mtp_vllm_0_20_2",
        "vllm_0_11_0": "vllm_qwen3_next_plugin.variants.mtp_vllm_0_11_0",
        "vllm_0_15_1": "vllm_qwen3_next_plugin.variants.mtp_vllm_0_15_1",
        "vllm_0_18_1": "vllm_qwen3_next_plugin.variants.mtp_vllm_0_18_1",
    }


def get_config_module_names() -> dict[str, str]:
    return {
        "vllm_0_20_2": "vllm_qwen3_next_plugin.variants.config_vllm_0_20_2",
        "vllm_0_11_0": "vllm_qwen3_next_plugin.variants.config_vllm_0_11_0",
        "vllm_0_15_1": "vllm_qwen3_next_plugin.variants.config_vllm_0_15_1",
        "vllm_0_18_1": "vllm_qwen3_next_plugin.variants.config_vllm_0_18_1",
    }


def get_active_variant_module_name(version: str) -> str:
    version_key = get_supported_version_key(version)
    return get_variant_module_names()[version_key]


def get_active_mtp_module_name(version: str) -> str:
    version_key = get_supported_version_key(version)
    return get_mtp_module_names()[version_key]


def get_active_config_module_name(version: str) -> str:
    version_key = get_supported_version_key(version)
    return get_config_module_names()[version_key]


def get_upstream_variant_paths(repo_root: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        "vllm_0_15_1": repo_root
        / "vllm_qwen3_next_plugin"
        / "upstream"
        / "vllm_0_15_1"
        / "qwen3_next.py",
        "vllm_0_18_1": repo_root
        / "vllm_qwen3_next_plugin"
        / "upstream"
        / "vllm_0_18_1"
        / "qwen3_next.py",
    }
