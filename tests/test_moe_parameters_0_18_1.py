from types import SimpleNamespace

from vllm_qwen3_next_plugin.variants.vllm_0_18_1 import QwenNextMixtureOfExperts


class _DummyMoEHost(QwenNextMixtureOfExperts):
    pass


def test_set_moe_parameters_handles_num_experts_zero() -> None:
    host = _DummyMoEHost()
    host.config = SimpleNamespace(num_experts=0)
    host.model = SimpleNamespace(layers=[])

    host.set_moe_parameters()

    assert host.moe_layers == []
    assert host.num_moe_layers == 0
    assert host.num_expert_groups == 0
    assert host.num_shared_experts == 0
    assert host.num_logical_experts == 0
    assert host.num_physical_experts == 0
    assert host.num_local_physical_experts == 0
    assert host.num_routed_experts == 0
    assert host.num_redundant_experts == 0
