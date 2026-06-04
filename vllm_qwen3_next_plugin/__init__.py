GREEN = "\033[32m"
RESET = "\033[0m"

from .compat.speculative import apply_runtime_patches
from .compat.ascend import (
    patch_ascend_qwen3_next_registration,
    patch_gdn_aclgraph_support,
    patch_plugin_causal_conv1d_ops,
    patch_plugin_fla_ops,
    patch_triton_placeholder_math_symbols,
)

__all__ = ["Qwen3NextForCausalLM", "register"]


def __getattr__(name: str):
    if name == "Qwen3NextForCausalLM":
        patch_triton_placeholder_math_symbols()
        patch_gdn_aclgraph_support()
        from .qwen3_next import Qwen3NextForCausalLM

        patch_plugin_causal_conv1d_ops(Qwen3NextForCausalLM)
        patch_plugin_fla_ops(Qwen3NextForCausalLM)
        return Qwen3NextForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register():
    from vllm import ModelRegistry
    import vllm

    patch_triton_placeholder_math_symbols()
    patch_gdn_aclgraph_support()

    import vllm.transformers_utils.configs as vllm_configs_module
    from .config import Qwen3NextConfig as PluginQwen3NextConfig
    from .mtp import Qwen3NextMTP
    from .qwen3_next import Qwen3NextForCausalLM
    from .versioning import get_supported_version_key

    vllm_version = vllm.__version__
    version_key = get_supported_version_key(vllm_version)
    vllm_configs_module.Qwen3NextConfig = PluginQwen3NextConfig
    apply_runtime_patches()
    patch_plugin_causal_conv1d_ops(Qwen3NextForCausalLM)
    patch_plugin_fla_ops(Qwen3NextForCausalLM)
    patch_ascend_qwen3_next_registration(Qwen3NextForCausalLM, Qwen3NextMTP)

    print(f"{GREEN}[vLLM Plugin] Loaded implementation for vLLM {vllm_version}{RESET}")
    ModelRegistry.register_model("Qwen3NextForCausalLM", Qwen3NextForCausalLM)
    print(
        f"{GREEN}[vLLM Plugin] Registered Qwen3NextForCausalLM with custom support{RESET}"
    )
    print(
        f"{GREEN}[vLLM Plugin] Registered Qwen3NextMTP ({version_key}) with custom support{RESET}"
    )
    ModelRegistry.register_model("Qwen3NextMTP", Qwen3NextMTP)

    print(
        f"{GREEN}[vLLM Plugin] Enable this plugin with `VLLM_PLUGINS=register_qwen3_next_model`; on Ascend use `VLLM_PLUGINS=ascend,register_qwen3_next_model` if filtering plugins{RESET}"
    )
