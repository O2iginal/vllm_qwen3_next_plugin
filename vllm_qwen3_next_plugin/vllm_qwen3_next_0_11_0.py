# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

import math
import os
from collections.abc import Iterable
from itertools import islice
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from vllm.attention import Attention, AttentionBackend, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops import (
    RMSNormGated,
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.fused_moe import FusedMoE

# yapf conflicts with isort for this block
# yapf: disable
from vllm.model_executor.layers.layernorm import RMSNorm, GemmaRMSNorm
# yapf: enable
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


class GatedRMSNorm(nn.Module):
    """
    RMSNorm followed by element-wise gating (low-rank self-gating).
    Forward: y = RMSNorm(x); gate = sigmoid(W_up(swish(W_down(y)))) * gate_scale; return gate * y.
    Supports fused residual path: forward(x, residual) -> (out, residual).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        rank: int = 16,
        gate_scale: float = 1.0,
        up_bias: bool = False,
    ) -> None:
        super().__init__()
        self.rms_norm = RMSNorm(hidden_size, eps=eps)
        self.w_down = nn.Linear(hidden_size, rank, bias=False)
        self.w_up = nn.Linear(rank, hidden_size, bias=up_bias)
        self.gate_scale = gate_scale
        self._init_gate_weights()

    def _init_gate_weights(self) -> None:
        nn.init.zeros_(self.w_down.weight)
        nn.init.zeros_(self.w_up.weight)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Fuse residual add like vLLM RMSNorm: norm(x + residual), return (out, x+residual).
        if residual is not None:
            x = x + residual
            residual = x
        y = self.rms_norm(x)
        down = self.w_down(y)
        activated = down * torch.sigmoid(down)  # swish
        up = self.w_up(activated)
        gate = torch.sigmoid(up) * self.gate_scale
        out = gate * y
        if residual is not None:
            return out, residual
        return out


def _get_qwen3_next_norm_cls(config):
    """Return RMSNorm, GemmaRMSNorm, or a factory for GatedRMSNorm by config.norm_type (default 'rms')."""
    t = getattr(config, "norm_type", "rms")
    if t in ("gemma_rms", "gemma"):
        return GemmaRMSNorm
    if t == "gated_rms":
        rank = getattr(config, "gated_norm_rank", 16)
        gate_scale = getattr(config, "gated_norm_gate_scale", 1.0)
        up_bias = getattr(config, "gated_norm_up_bias", False)

        def _factory(hidden_size: int, eps: float = 1e-6) -> GatedRMSNorm:
            return GatedRMSNorm(
                hidden_size,
                eps=eps,
                rank=rank,
                gate_scale=gate_scale,
                up_bias=up_bias,
            )

        return _factory
    return RMSNorm


def _apply_causal_depthwise_conv1d_bcl(
    x_bcl: torch.Tensor,
    weight_c1w: torch.Tensor,
) -> torch.Tensor:
    """Apply causal depthwise Conv1d for token shifting (cannon layer). Stateless.
    x_bcl: [B, C, L], weight_c1w: [C, 1, W]. Output: [B, C, L].
    Use F.pad + F.conv1d only; causal_conv1d_fn requires conv_states (Mamba) and breaks torch.compile.
    """
    w = weight_c1w.shape[-1]
    x_pad = F.pad(x_bcl, (w - 1, 0))
    y = F.conv1d(
        x_pad,
        weight_c1w,
        bias=None,
        stride=1,
        padding=0,
        groups=weight_c1w.shape[0],
    )
    return y[..., : x_bcl.shape[-1]]


def show_tensor_info(tensor: torch.Tensor, name: str):
    print(
        f"[Tensor Info] {name}: shape={tensor.shape}, has_nan={torch.isnan(tensor).any().item()}, has_inf={torch.isinf(tensor).any().item()}, mean={tensor.mean().item()}, std={tensor.std().item()}"
    )


class Qwen3NextMLPWithTokenShift(nn.Module):
    """MLP with optional token shifting (cannon layer) at entry and before down_proj.
    Uses MergedColumnParallelLinear(gate_up_proj) so load_weights matches Qwen2MoeMLP.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        if config.hidden_act != "silu":
            raise ValueError(
                "Qwen3NextMLPWithTokenShift only supports hidden_act='silu' (same as Qwen2MoeMLP)."
            )
        kernel_size = getattr(config, "token_shift_conv_size", 4)
        self.ffn_token_shift = getattr(config, "ffn_token_shift", None)
        self.ffn_intermediate_token_shift = getattr(
            config, "ffn_intermediate_token_shift", None
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        if self.ffn_token_shift == "conv":
            self.token_shift_conv = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=kernel_size,
                padding=0,
                groups=hidden_size,
                bias=False,
            )
        else:
            self.token_shift_conv = None
        if self.ffn_intermediate_token_shift == "conv":
            self.intermediate_token_shift_conv = nn.Conv1d(
                intermediate_size,
                intermediate_size,
                kernel_size=kernel_size,
                padding=0,
                groups=intermediate_size,
                bias=False,
            )
        else:
            self.intermediate_token_shift_conv = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ffn_token_shift == "conv" and self.token_shift_conv is not None:
            x_bcl = hidden_states.t().unsqueeze(0)
            x_bcl = _apply_causal_depthwise_conv1d_bcl(
                x_bcl, self.token_shift_conv.weight
            )
            hidden_states = x_bcl.squeeze(0).t()
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate = self.act_fn(gate_up)
        if (
            self.ffn_intermediate_token_shift == "conv"
            and self.intermediate_token_shift_conv is not None
        ):
            inter_bcl = intermediate.t().unsqueeze(0)
            inter_bcl = _apply_causal_depthwise_conv1d_bcl(
                inter_bcl, self.intermediate_token_shift_conv.weight
            )
            intermediate = inter_bcl.squeeze(0).t()
        down, _ = self.down_proj(intermediate)
        return down


# ---------------------------------------------------------------------------
# @gyzp: training-side inference features (all config-gated; defaults no-op)
#   1) moe_router_sqrt_gate: softmax -> top-k -> L1 renorm -> sqrt(weight),
#      no renorm afterwards (Megatron router semantics).
#   2) activation_func_clamp_value: soft clamp c*tanh(x/c) on MLP gate/up
#      BEFORE silu(gate) * up; applies to dense MLPs, the shared expert and
#      routed experts (HF reference semantics).
#   3) final_logits_clamp_value: hard clamp on final logits.
# ---------------------------------------------------------------------------


def _get_activation_clamp_value(config) -> Optional[float]:
    value = getattr(config, "activation_func_clamp_value", None)
    if value is None:
        return None
    mode = getattr(config, "activation_func_clamp_mode", "soft")
    if mode != "soft":
        raise ValueError(
            f"activation_func_clamp_mode={mode!r} is not supported; "
            "only 'soft' (c * tanh(x / c)) is implemented."
        )
    return float(value)


_ACTIVATION_CLAMP_VALUE: Optional[float] = None
_ORIG_SILU_AND_MUL = None


class _SiluAndMulClampShim:
    """OpOverloadPacket-compatible wrapper for torch.ops._C.silu_and_mul.

    __call__ applies the optional soft clamp; every other attribute
    (.default, .cuda, ...) is delegated to the original OpOverloadPacket so
    consumers such as vllm.compilation pass infrastructure (which do
    `torch.ops._C.silu_and_mul.default` for pattern matching) keep working.
    With the clamp inactive, behavior is indistinguishable from the
    original packet.
    """

    def __init__(self, orig_packet) -> None:
        self._orig_packet = orig_packet

    def __call__(self, out: torch.Tensor, x: torch.Tensor):
        c = _ACTIVATION_CLAMP_VALUE
        if c is None:
            return self._orig_packet(out, x)
        gate, up = x.chunk(2, dim=-1)
        gate = c * torch.tanh(gate / c)
        up = c * torch.tanh(up / c)
        out.copy_(F.silu(gate) * up)

    def __getattr__(self, name: str):
        return getattr(self._orig_packet, name)


def _install_silu_clamp_shim() -> None:
    """Shadow torch.ops._C.silu_and_mul with a clamp-aware wrapper.

    The fused-MoE paths (Triton/modular) look the op up at call time, and
    SiluAndMul binds it at module construction (which happens after this
    module is imported), so a single shim covers dense MLPs, the shared
    expert and routed experts. It is a pure passthrough unless
    _ACTIVATION_CLAMP_VALUE is set, so other checkpoints are unaffected.
    Note: pure-torch native (CPU) fallbacks and the flashinfer-cutlass MoE
    path do not go through this op and are not covered.
    """
    global _ORIG_SILU_AND_MUL
    if _ORIG_SILU_AND_MUL is not None:
        return
    import vllm._custom_ops  # noqa: F401  (ensures torch.ops._C is loaded)

    _ORIG_SILU_AND_MUL = torch.ops._C.silu_and_mul
    torch.ops._C.silu_and_mul = _SiluAndMulClampShim(_ORIG_SILU_AND_MUL)


_install_silu_clamp_shim()


def _maybe_enable_activation_clamp(config) -> None:
    """Activate the silu_and_mul clamp shim for this process if configured."""
    global _ACTIVATION_CLAMP_VALUE
    value = _get_activation_clamp_value(config)
    if value is None:
        return
    if _ACTIVATION_CLAMP_VALUE is not None and _ACTIVATION_CLAMP_VALUE != value:
        raise ValueError(
            "Conflicting activation_func_clamp_value: "
            f"{_ACTIVATION_CLAMP_VALUE} vs {value}"
        )
    _ACTIVATION_CLAMP_VALUE = value


def _sqrt_gate_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax -> top-k -> L1 renorm -> sqrt of the selected weights.

    Matches Megatron moe_router_sqrt_gate: sqrt replaces the combine weight
    of selected experts only; selection is unchanged (sqrt is monotonic) and
    there is NO renormalization afterwards.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
    )
    return topk_weights.sqrt(), topk_ids


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.num_experts > 0 and self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            # @gyzp sqrt-gate: sqrt of selected (post-renorm) combine weights
            custom_routing_function=(
                _sqrt_gate_routing_function
                if getattr(config, "moe_router_sqrt_gate", False)
                else None
            ),
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        # Token shift at MoE entry (cannon layer, reuses ffn_token_shift) @gyzp
        self.ffn_token_shift = getattr(config, "ffn_token_shift", None)
        kernel_size = getattr(config, "token_shift_conv_size", 4)
        if self.ffn_token_shift == "conv":
            self.token_shift_conv = nn.Conv1d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=kernel_size,
                padding=0,
                groups=config.hidden_size,
                bias=False,
            )
        else:
            self.token_shift_conv = None

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=self.experts.must_reduce_shared_expert_outputs(),
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None
        if getattr(config, "use_shared_expert_gate", False):
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Token shift at MoE entry (cannon layer, conv mode) @gyzp
        if self.ffn_token_shift == "conv" and self.token_shift_conv is not None:
            x_bcl = hidden_states.t().unsqueeze(0)
            x_bcl = _apply_causal_depthwise_conv1d_bcl(
                x_bcl, self.token_shift_conv.weight
            )
            hidden_states = x_bcl.squeeze(0).t()

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
                )

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)


class Qwen3NextGatedDeltaNet(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str:
        return "linear_attention"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

        return GDNAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )
        # @gyzp optional RNN/GDN RoPE (rope/nope); getattr for backward compat, default nope
        self.rnn_position_embedding_type = getattr(
            config, "rnn_position_embedding_type", "nope"
        )
        self.rotary_emb = None
        if self.rnn_position_embedding_type == "rope":
            self.rotary_emb = get_rope(
                head_size=self.head_k_dim,
                rotary_dim=self.head_k_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                rope_scaling=getattr(config, "rope_scaling", None),
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 0.25),
            )

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.projection_size_qkvz,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )
        # ba_proj doesn't support blockwise fp8 quantization.
        self.in_proj_ba = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.projection_size_ba,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ],
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size, dtype=torch.float32),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        if "HACK_GDN_EPS" in os.environ:
            gdn_eps = float(os.getenv("HACK_GDN_EPS"))
            print(f"HACK_GDN_EPS 生效： {gdn_eps}")
        else:
            gdn_eps = self.layer_norm_epsilon
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=gdn_eps,
            group_size=None,
            norm_before_gate=True,
            device=current_platform.current_device(),
            dtype=config.torch_dtype,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz,
        mixed_ba,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: Optional[
            torch.Tensor
        ] = None,  # @gyzp for RNN RoPE when rnn_position_embedding_type == "rope"
    ):
        # @gyzp pass positions into context so _forward can apply RNN RoPE
        # ForwardContext is a dataclass (no .get/.setdefault); use dynamic attr
        if positions is not None:
            ctx = get_forward_context()
            gdn_positions = getattr(ctx, "gdn_positions", None)
            if gdn_positions is None:
                gdn_positions = {}
                setattr(ctx, "gdn_positions", gdn_positions)
            gdn_positions[self.prefix] = positions
        return torch.ops.vllm.gdn_attention(
            hidden_states,
            output,
            self.prefix,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_masks = attn_metadata.spec_token_masks
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        if spec_token_masks is not None:
            spec_token_masks = spec_token_masks[:num_actual_tokens]

        # 1. Set up dimensions for reshapes later
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states[:num_actual_tokens])
        projected_states_ba, _ = self.in_proj_ba(hidden_states[:num_actual_tokens])
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = map(
            lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv[spec_token_masks]
                mixed_qkv_non_spec = mixed_qkv[~spec_token_masks]
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 2.1: process the mutli-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 2.2: process the remaining part
        if attn_metadata.num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_decodes
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        # @gyzp apply RNN RoPE to q/k when rnn_position_embedding_type == "rope" (before recurrent)
        # ForwardContext is a dataclass; use getattr to avoid AttributeError under Dynamo
        gdn_positions = getattr(get_forward_context(), "gdn_positions", None)
        positions = gdn_positions.get(self.prefix) if isinstance(
            gdn_positions, dict) else None
        if self.rotary_emb is not None and positions is not None:
            if spec_token_masks is not None:
                positions_spec = positions[spec_token_masks]
                positions_non_spec = positions[~spec_token_masks]
            else:
                positions_spec = None
                positions_non_spec = positions
            if query_spec is not None and positions_spec is not None:
                query_spec, key_spec = self.rotary_emb(
                    positions_spec, query_spec, key_spec
                )
            if query_non_spec is not None and positions_non_spec is not None:
                query_non_spec, key_non_spec = self.rotary_emb(
                    positions_non_spec, query_non_spec, key_non_spec
                )

        beta = b.float().sigmoid()
        # g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        g = fused_gdn_gating(self.A_log, a, self.dt_bias)

        g, beta = map(lambda x: rearrange(x, "l d -> 1 l d"), (g, beta))

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g[:, spec_token_masks]
                beta_spec = beta[:, spec_token_masks]
                g_non_spec = g[:, ~spec_token_masks]
                beta_non_spec = beta[:, ~spec_token_masks]
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 3. Recurrent attention

        # 3.1: process the mutlti-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                g=g_spec,
                beta=beta_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 3.2: process the remaining part
        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            # Init cache
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            core_attn_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            core_attn_out[:, spec_token_masks] = core_attn_out_spec
            core_attn_out[:, ~spec_token_masks] = core_attn_out_non_spec
        elif spec_sequence_masks is not None:
            core_attn_out = core_attn_out_spec
        else:
            core_attn_out = core_attn_out_non_spec

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")

        output[:num_actual_tokens], _ = self.out_proj(core_attn_out)


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", False)
        # @gyzp optional attn RoPE (rope/nope) and logits scaling; getattr for backward compat
        self.attn_position_embedding_type = getattr(
            config, "attn_position_embedding_type", "rope"
        )
        self.attn_logits_scaling = getattr(config, "attn_logits_scaling", None)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", True),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # @gyzp attn q/k/v token shift (cannon layer, conv mode)
        kernel_size = getattr(config, "token_shift_conv_size", 4)
        self.attn_q_token_shift = getattr(config, "attn_q_token_shift", None)
        self.attn_k_token_shift = getattr(config, "attn_k_token_shift", None)
        self.attn_v_token_shift = getattr(config, "attn_v_token_shift", None)
        if self.attn_q_token_shift == "conv":
            self.attn_q_token_shift_conv = nn.Conv1d(
                self.q_size,
                self.q_size,
                kernel_size=kernel_size,
                padding=0,
                groups=self.q_size,
                bias=False,
            )
        else:
            self.attn_q_token_shift_conv = None
        if self.attn_k_token_shift == "conv":
            self.attn_k_token_shift_conv = nn.Conv1d(
                self.kv_size,
                self.kv_size,
                kernel_size=kernel_size,
                padding=0,
                groups=self.kv_size,
                bias=False,
            )
        else:
            self.attn_k_token_shift_conv = None
        if self.attn_v_token_shift == "conv":
            self.attn_v_token_shift_conv = nn.Conv1d(
                self.kv_size,
                self.kv_size,
                kernel_size=kernel_size,
                padding=0,
                groups=self.kv_size,
                bias=False,
            )
        else:
            self.attn_v_token_shift_conv = None

        # @gyzp only create RoPE when attn_position_embedding_type == "rope" (nope = no RoPE)
        self.rotary_emb = (
            get_rope(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                rope_scaling=config.rope_scaling,
                partial_rotary_factor=config.partial_rotary_factor,
                dual_chunk_attention_config=self.dual_chunk_attention_config,
            )
            if self.attn_position_embedding_type == "rope"
            else None
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        # @gyzp QK norm: default off, enable via config.attn_qk_norm
        self.attn_qk_norm = getattr(config, "attn_qk_norm", False)
        if self.attn_qk_norm:
            _nc = _get_qwen3_next_norm_cls(config)
            self.q_norm = _nc(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = _nc(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # @gyzp attn q/k/v token shift (cannon layer, conv mode)
        if self.attn_q_token_shift_conv is not None:
            q_bcl = q.t().unsqueeze(0)
            q_bcl = _apply_causal_depthwise_conv1d_bcl(
                q_bcl, self.attn_q_token_shift_conv.weight
            )
            q = q_bcl.squeeze(0).t()
        if self.attn_k_token_shift_conv is not None:
            k_bcl = k.t().unsqueeze(0)
            k_bcl = _apply_causal_depthwise_conv1d_bcl(
                k_bcl, self.attn_k_token_shift_conv.weight
            )
            k = k_bcl.squeeze(0).t()
        if self.attn_v_token_shift_conv is not None:
            v_bcl = v.t().unsqueeze(0)
            v_bcl = _apply_causal_depthwise_conv1d_bcl(
                v_bcl, self.attn_v_token_shift_conv.weight
            )
            v = v_bcl.squeeze(0).t()

        # @gyzp QK norm: apply when attn_qk_norm is True
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
                -1, self.num_heads * self.head_dim
            )
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
                -1, self.num_kv_heads * self.head_dim
            )
        else:
            q = q.view(-1, self.num_heads, self.head_dim).view(
                -1, self.num_heads * self.head_dim
            )
            k = k.view(-1, self.num_kv_heads, self.head_dim).view(
                -1, self.num_kv_heads * self.head_dim
            )

        # @gyzp apply RoPE only when attn_position_embedding_type == "rope"
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        # @gyzp optional attn logits scaling (float const or "log"/"log <a>") before attention
        if self.attn_logits_scaling is not None:
            if isinstance(self.attn_logits_scaling, (int, float)):
                q = q * self.attn_logits_scaling
            elif isinstance(self.attn_logits_scaling, str):
                parts = self.attn_logits_scaling.split()
                a = float(parts[1]) if len(parts) > 1 else 362.0
                pos_f = positions.to(device=q.device, dtype=q.dtype)
                scale = (torch.log(pos_f + a) / math.log(a)).unsqueeze(-1)
                q = q * scale

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # Attn token shift (cannon layer) before attention @gyzp
        self.attn_token_shift = getattr(config, "attn_token_shift", None)
        kernel_size = getattr(config, "token_shift_conv_size", 4)
        if self.attn_token_shift == "conv":
            self.attn_token_shift_conv = nn.Conv1d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=kernel_size,
                padding=0,
                groups=config.hidden_size,
                bias=False,
            )
        else:
            self.attn_token_shift_conv = None

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        use_mlp_token_shift = (
            getattr(config, "ffn_token_shift", None) is not None
            or getattr(config, "ffn_intermediate_token_shift", None) is not None
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif use_mlp_token_shift:
            self.mlp = Qwen3NextMLPWithTokenShift(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )

        norm_cls = _get_qwen3_next_norm_cls(config)
        self.input_layernorm = norm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = norm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.torch_dtype,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.torch_dtype,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Attn token shift (cannon layer, conv mode) before attention @gyzp
        if self.attn_token_shift == "conv" and self.attn_token_shift_conv is not None:
            # hidden_states: [num_tokens, hidden] -> [1, hidden, num_tokens]
            x_bcl = hidden_states.t().unsqueeze(0)
            x_bcl = _apply_causal_depthwise_conv1d_bcl(
                x_bcl, self.attn_token_shift_conv.weight
            )
            hidden_states = x_bcl.squeeze(0).t()

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            # @gyzp pass positions for GDN RNN RoPE when rnn_position_embedding_type == "rope"
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}"
                )  # noqa: E501
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        lora_config = vllm_config.lora_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            use_gated_final_norm = getattr(config, "final_gated_norm", False)
            if use_gated_final_norm:
                self.norm = _get_qwen3_next_norm_cls(config)(
                    config.hidden_size, eps=config.rms_norm_eps
                )
            else:
                self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # 如果没有专家，返回空列表
        if self.config.num_experts == 0:
            return []

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Resolve key: loader may pass "model.layers.*" but params_dict has "layers.*"
                    key = (
                        name
                        if name in params_dict
                        else (
                            name.removeprefix("model.")
                            if name.startswith("model.")
                            else name
                        )
                    )
                    if key not in params_dict:
                        continue
                    param = params_dict[key]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3NextForCausalLM(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, MixtureOfExperts, IsHybrid
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        GREEN = "\033[32m"
        RESET = "\033[0m"
        print(f"{GREEN}\n[vLLM Plugin] Qwen3NextForCausalLM init\n{RESET}")
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, (
            "Qwen3Next currently does not support prefix caching"
        )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        _maybe_enable_activation_clamp(config)  # @gyzp MLP gate/up soft clamp
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config
            else lora_config.lora_vocab_padding_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # Tie lm_head weight with embed_tokens weight if tie_word_embeddings is True
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers: list[FusedMoE] = []

        # 检查是否有 MoE 层, @gyzp 可禁用 MoE
        if config.num_experts == 0:
            # 当 num_experts 为 0 时，设置默认值
            self.num_moe_layers = 0
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
        else:
            # 原有的 example_layer 查找逻辑
            example_layer = None
            for layer in self.model.layers:
                if isinstance(layer, PPMissingLayer):
                    continue

                assert isinstance(layer, Qwen3NextDecoderLayer)
                if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                    example_layer = layer.mlp
                    self.moe_layers.append(layer.mlp.experts)

            if example_layer is None:
                raise RuntimeError("No Qwen3Next layer found in the model.layers.")

            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = example_layer.n_logical_experts
            self.num_physical_experts = example_layer.n_physical_experts
            self.num_local_physical_experts = example_layer.n_local_physical_experts
            self.num_routed_experts = example_layer.n_routed_experts
            self.num_redundant_experts = example_layer.n_redundant_experts

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        # 如果没有 MoE 层，直接返回
        if self.config.num_experts == 0:
            return

        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        # 如果没有 MoE 层，直接返回
        if self.config.num_experts == 0:
            return

        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        # @gyzp hard clamp on final logits (HF reference semantics)
        clamp = getattr(self.config, "final_logits_clamp_value", None)
        if clamp is not None:
            logits = logits.clamp(-clamp, clamp)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        loaded_params = loader.load_weights(weights)
        # If tie_word_embeddings is True, lm_head.weight is tied to embed_tokens.weight
        # So we need to mark lm_head.weight as loaded even if it's not in the checkpoint
        if getattr(self.config, "tie_word_embeddings", False):
            # Add lm_head.weight to loaded_params to avoid "not initialized" error
            # The weight is actually initialized via the tie with embed_tokens.weight
            loaded_params.add("lm_head.weight")
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def gdn_attention(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(hidden_states=hidden_states, output=output)


def gdn_attention_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="gdn_attention",
    op_func=gdn_attention,
    mutates_args=["output"],
    fake_impl=gdn_attention_fake,
)


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty_like(a, dtype=torch.float32)
    fused_gdn_gating_kernel[grid](
        g, A_log, a, dt_bias, seq_len, num_heads, beta, threshold, 8, num_warps=1
    )
    return g
