import importlib

import vllm

from vllm_qwen3_next_plugin import versioning


def _current_variant_suffix() -> str:
    return versioning.get_supported_version_key(vllm.__version__)


def test_qwen3_next_dispatcher_reexports_variant_symbols() -> None:
    dispatcher = importlib.import_module("vllm_qwen3_next_plugin.qwen3_next")
    variant = importlib.import_module(
        f"vllm_qwen3_next_plugin.variants.{_current_variant_suffix()}"
    )

    assert dispatcher.Qwen3NextForCausalLM is variant.Qwen3NextForCausalLM


def test_mtp_dispatcher_reexports_variant_symbols() -> None:
    dispatcher = importlib.import_module("vllm_qwen3_next_plugin.mtp")
    variant = importlib.import_module(
        f"vllm_qwen3_next_plugin.variants.mtp_{_current_variant_suffix()}"
    )

    assert dispatcher.Qwen3NextMTP is variant.Qwen3NextMTP


def test_config_dispatcher_reexports_variant_symbols() -> None:
    dispatcher = importlib.import_module("vllm_qwen3_next_plugin.config")
    variant = importlib.import_module(
        f"vllm_qwen3_next_plugin.variants.config_{_current_variant_suffix()}"
    )

    assert dispatcher.Qwen3NextConfig is variant.Qwen3NextConfig
