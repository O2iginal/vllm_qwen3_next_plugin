from __future__ import annotations

from importlib import import_module

from vllm_qwen3_next_plugin.versioning import get_active_variant_module_name


def _load_active_variant():
    import vllm

    module_name = get_active_variant_module_name(vllm.__version__)
    return import_module(module_name)


_active_variant = _load_active_variant()

Qwen3NextForCausalLM = _active_variant.Qwen3NextForCausalLM


def __getattr__(name: str):
    return getattr(_active_variant, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_active_variant)))
