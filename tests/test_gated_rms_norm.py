from types import SimpleNamespace
import unittest

from vllm_qwen3_next_plugin.vllm_qwen3_next_0_11_0 import (
    GatedRMSNorm,
    _get_qwen3_next_norm_cls,
)


class TestGatedRMSNorm(unittest.TestCase):
    def test_up_bias_follows_checkpoint_option(self):
        config = SimpleNamespace(
            norm_type="gated_rms",
            gated_norm_rank=4,
            gated_norm_gate_scale=1.0,
            gated_norm_up_bias=True,
        )
        make_norm = _get_qwen3_next_norm_cls(config)
        norm = make_norm(hidden_size=8, eps=1e-6)

        self.assertIsInstance(norm, GatedRMSNorm)
        self.assertIsNotNone(norm.w_up.bias)

    def test_norm_factory_uses_gated_rms(self):
        config = SimpleNamespace(
            norm_type="gated_rms",
            gated_norm_rank=4,
            gated_norm_gate_scale=1.0,
            gated_norm_up_bias=False,
        )

        self.assertIsInstance(_get_qwen3_next_norm_cls(config)(8), GatedRMSNorm)


if __name__ == "__main__":
    unittest.main()
