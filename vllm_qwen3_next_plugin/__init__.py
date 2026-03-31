GREEN = "\033[32m"
RESET = "\033[0m"

try:
    from .qwen3_next import Qwen3NextForCausalLM
except ImportError:
    from .vllm_qwen3_next_0_11_0 import Qwen3NextForCausalLM

__all__ = ["Qwen3NextForCausalLM", "register"]


def register():
    from vllm import ModelRegistry
    import vllm

    import vllm.transformers_utils.configs as vllm_configs_module
    from .config import Qwen3NextConfig as PluginQwen3NextConfig
    from .mtp import Qwen3NextMTP
    from .versioning import get_supported_version_key

    vllm_version = vllm.__version__
    version_key = get_supported_version_key(vllm_version)
    vllm_configs_module.Qwen3NextConfig = PluginQwen3NextConfig

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
        f"{GREEN}[vLLM Plugin] Enable this plugin by setting `os.environ['VLLM_PLUGINS'] = 'register_qwen3_next_model'` if this not working{RESET}"
    )
