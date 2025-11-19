GREEN = "\033[32m"
RESET = "\033[0m"

def register():
    from vllm import ModelRegistry
    import vllm
    
    # Get vLLM version and select appropriate implementation
    vllm_version = vllm.__version__
    
    if vllm_version.startswith("0.10.2") or vllm_version.startswith("0.11.0"):
        # Import Qwen3NextForCausalLM based on vLLM version
        if vllm_version.startswith("0.10.2"):
            from .vllm_qwen3_next_0_10_2 import Qwen3NextForCausalLM
        elif vllm_version.startswith("0.11.0"):
            from .vllm_qwen3_next_0_11_0 import Qwen3NextForCausalLM
        
        print(f"{GREEN}[vLLM Plugin] Loaded implementation for vLLM {vllm_version}{RESET}")
        
        # Register Qwen3NextForCausalLM
        print(f"{GREEN}[vLLM Plugin] Registered Qwen3NextForCausalLM with custom support{RESET}")
        ModelRegistry.register_model("Qwen3NextForCausalLM", Qwen3NextForCausalLM)
        
        # Register Qwen3NextMTP
        if vllm_version.startswith("0.10.2"):
            from .vllm_qwen3_next_mtp_0_10_2 import Qwen3NextMTP
            print(f"{GREEN}[vLLM Plugin] Registered Qwen3NextMTP (v0.10.2) with custom support{RESET}")
            ModelRegistry.register_model("Qwen3NextMTP", Qwen3NextMTP)
        elif vllm_version.startswith("0.11.0"):
            from .vllm_qwen3_next_mtp_0_11_0 import Qwen3NextMTP
            print(f"{GREEN}[vLLM Plugin] Registered Qwen3NextMTP (v0.11.0) with custom support{RESET}")
            ModelRegistry.register_model("Qwen3NextMTP", Qwen3NextMTP)
    else:
        raise ImportError(f"Unsupported vLLM version: {vllm_version}. Supported versions: 0.10.2, 0.11.0")
    
    print(f"{GREEN}[vLLM Plugin] Enable this plugin by setting `os.environ['VLLM_PLUGINS'] = 'register_qwen3_next_model'` if this not working{RESET}")