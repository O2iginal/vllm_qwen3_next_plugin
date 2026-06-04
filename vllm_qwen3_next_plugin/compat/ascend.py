from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import torch
import torch.nn.functional as F


PAD_SLOT_ID = -1


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _cdiv(left: int, right: int) -> int:
    return (left + right - 1) // right


def _missing_triton_runtime_symbol(*_args, **_kwargs):
    raise RuntimeError(
        "vLLM Triton placeholder math symbol was executed. "
        "Ascend runtime patches should replace the Triton FLA kernels before use."
    )


def _causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    _, width = weight.shape
    if initial_states is not None:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=weight.size(0))
    else:
        out = F.conv1d(
            x,
            weight.unsqueeze(1),
            bias,
            padding=width - 1,
            groups=weight.size(0),
        )
    out = out[..., :seqlen]
    if final_states_out is not None:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        final_states_out.copy_(final_states)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out.to(dtype=dtype_in)


def _torch_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    conv_states: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
):
    if query_start_loc is None or cache_indices is None or conv_states is None:
        raise NotImplementedError("Ascend torch causal conv fallback requires cache metadata")
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    outputs: list[torch.Tensor] = []
    seqlens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
    for i, x_s in enumerate(torch.split(x, seqlens, dim=-1)):
        cache_idx = int(cache_indices[i].item())
        if cache_idx == pad_slot_id:
            outputs.append(x_s.new_zeros(x_s.shape))
            continue
        use_initial = bool(has_initial_state[i].item()) if has_initial_state is not None else False
        outputs.append(
            _causal_conv1d_ref(
                x_s,
                weight,
                bias,
                initial_states=conv_states[cache_idx] if use_initial else None,
                final_states_out=conv_states[cache_idx].unsqueeze(0),
                activation=activation,
            )
        )
    return torch.cat(outputs, dim=-1) if outputs else x.new_empty((x.size(0), 0))


def _torch_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    intermediate_conv_window: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data: bool = False,
    **_kwargs,
) -> torch.Tensor:
    del cache_seqlens, num_accepted_tokens, intermediate_conv_window, metadata
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.size(1) == dim
        assert conv_state.size(2) >= width - 1

    outputs: list[torch.Tensor] = []
    for i in range(batch):
        cache_idx = int(conv_state_indices[i].item()) if conv_state_indices is not None else i
        if cache_idx == pad_slot_id:
            outputs.append(x.new_zeros((dim, seqlen)))
            continue
        y = _causal_conv1d_ref(
            x[i],
            weight,
            bias,
            initial_states=conv_state[cache_idx],
            final_states_out=conv_state[cache_idx].unsqueeze(0),
            activation=activation,
        )
        outputs.append(y)
    out = torch.stack(outputs, dim=0)
    return out.squeeze(-1) if unsqueeze else out


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    initial_dtype = query.dtype
    if key.shape[2] != value.shape[2]:
        if value.shape[2] % key.shape[2] != 0:
            raise RuntimeError(
                f"Cannot expand {key.shape[2]} key heads to {value.shape[2]} value heads"
            )
        repeat = value.shape[2] // key.shape[2]
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_tokens = sequence_length + pad_size
    query = query * (query.shape[-1] ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        last_recurrent_state = torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, device=value.device
        ).to(value)
    else:
        last_recurrent_state = initial_state.to(value)

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    for i in range(0, total_tokens // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
            mask, 0
        )
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (
                k_i
                * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0],
        core_attn_out.shape[1],
        -1,
        core_attn_out.shape[-1],
    )[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class _TorchLayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        z: torch.Tensor | None = None,
        eps: float = 1e-6,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        is_rms_norm: bool = False,
    ):
        del ctx
        x_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        z_2d = z.reshape(-1, z.shape[-1]) if z is not None else None
        hidden_size = x_2d.shape[-1]
        if group_size is None:
            group_size = hidden_size
        if hidden_size % group_size != 0:
            raise RuntimeError(
                f"hidden size {hidden_size} must be divisible by group size {group_size}"
            )

        x_norm = x_2d
        if z_2d is not None and not norm_before_gate:
            x_norm = x_norm * F.silu(z_2d)

        grouped = x_norm.reshape(x_norm.shape[0], -1, group_size).float()
        if is_rms_norm:
            variance = grouped.pow(2).mean(dim=-1, keepdim=True)
            normed = grouped * torch.rsqrt(variance + eps)
        else:
            mean = grouped.mean(dim=-1, keepdim=True)
            centered = grouped - mean
            variance = centered.pow(2).mean(dim=-1, keepdim=True)
            normed = centered * torch.rsqrt(variance + eps)
        y = normed.reshape_as(x_norm).to(dtype=x.dtype)
        y = y * weight
        if bias is not None:
            y = y + bias
        if z_2d is not None and norm_before_gate:
            y = y * F.silu(z_2d)
        return y.reshape(x_shape)


def _torch_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
):
    if initial_state is None:
        raise NotImplementedError("initial_state is required for Ascend recurrent fallback")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(g)
    if k.shape[2] != v.shape[2]:
        if v.shape[2] % k.shape[2] != 0:
            raise RuntimeError(
                f"Cannot expand {k.shape[2]} key heads to {v.shape[2]} value heads"
            )
        repeat = v.shape[2] // k.shape[2]
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

    final_state = initial_state if inplace_final_state else initial_state.clone()
    out = torch.empty_like(v)
    num_sequences = int(cu_seqlens.numel()) - 1 if cu_seqlens is not None else q.shape[0]
    for seq_idx in range(num_sequences):
        start = int(cu_seqlens[seq_idx].item()) if cu_seqlens is not None else 0
        end = int(cu_seqlens[seq_idx + 1].item()) if cu_seqlens is not None else q.shape[1]
        cache_idx = seq_idx
        if ssm_state_indices is not None and ssm_state_indices.ndim == 1:
            cache_idx = int(ssm_state_indices[seq_idx].item())
        state = final_state[cache_idx].float()
        for tok_idx in range(start, end):
            if ssm_state_indices is not None and ssm_state_indices.ndim == 2:
                token_offset = tok_idx - start
                accepted = (
                    int(num_accepted_tokens[seq_idx].item())
                    if num_accepted_tokens is not None
                    else ssm_state_indices.shape[1]
                )
                if token_offset >= accepted:
                    continue
                cache_idx = int(ssm_state_indices[seq_idx, token_offset].item())
                if cache_idx == PAD_SLOT_ID:
                    continue
                state = final_state[cache_idx].float()
            q_t = q[:, tok_idx, ...].float().squeeze(0)
            k_t = k[:, tok_idx, ...].float().squeeze(0)
            v_t = v[:, tok_idx, ...].float().squeeze(0)
            g_t = g[:, tok_idx, ...].float().squeeze(0)
            beta_t = beta[:, tok_idx, ...].float().squeeze(0)
            v_beta = v_t * beta_t.unsqueeze(-1)
            k_beta = k_t * beta_t.unsqueeze(-1)
            v_prime = torch.einsum("hkv,hk->hv", state, k_t)
            v_new = v_beta - v_prime
            out_t = (
                torch.einsum("hk,hkv->hv", q_t * scale, state)
                + (q_t * scale * k_t).sum(dim=-1, keepdim=True) * v_new
            )
            out[:, tok_idx, ...] = out_t.to(v.dtype).unsqueeze(0)
            state = state * g_t.exp().view(-1, 1, 1) + torch.einsum(
                "hk,hv->hkv", k_beta, v_new
            )
            if ssm_state_indices is not None and ssm_state_indices.ndim == 2:
                final_state[cache_idx] = state.to(final_state.dtype)
        if ssm_state_indices is None or ssm_state_indices.ndim == 1:
            final_state[cache_idx] = state.to(final_state.dtype)
    return out, final_state


def _wrap_ascend_chunk_gated_delta_rule(torch_chunk_gated_delta_rule):
    def _ascend_chunk_gated_delta_rule(
        q=None,
        k=None,
        v=None,
        query=None,
        key=None,
        value=None,
        g=None,
        beta=None,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unexpected Ascend FLA kwargs: {sorted(kwargs)}")
        if head_first:
            raise NotImplementedError("Ascend FLA wrapper expects head_first=False")

        query = query if query is not None else q
        key = key if key is not None else k
        value = value if value is not None else v
        if query is None or key is None or value is None or g is None or beta is None:
            raise TypeError("query/key/value/g/beta are required")

        chunk_gated_delta_rule = (
            _torch_chunk_gated_delta_rule
            if key.shape[2] != value.shape[2]
            else torch_chunk_gated_delta_rule
        )
        if cu_seqlens is None:
            return chunk_gated_delta_rule(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        batch_size = int(cu_seqlens.numel()) - 1
        core_attn_outs: list[torch.Tensor] = []
        last_states: list[torch.Tensor] = []
        for batch_idx in range(batch_size):
            start = int(cu_seqlens[batch_idx].item())
            end = int(cu_seqlens[batch_idx + 1].item())
            cur_initial_state = (
                None
                if initial_state is None
                else initial_state[batch_idx].unsqueeze(0)
            )
            cur_out, cur_state = chunk_gated_delta_rule(
                query=query[:, start:end, ...],
                key=key[:, start:end, ...],
                value=value[:, start:end, ...],
                g=g[:, start:end, ...],
                beta=beta[:, start:end, ...],
                initial_state=cur_initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            core_attn_outs.append(cur_out)
            if cur_state is not None:
                last_states.append(cur_state)

        if core_attn_outs:
            core_attn_out = torch.empty_like(value)
            for batch_idx, cur_out in enumerate(core_attn_outs):
                start = int(cu_seqlens[batch_idx].item())
                end = int(cu_seqlens[batch_idx + 1].item())
                core_attn_out[:, start:end, ...] = cur_out
        else:
            core_attn_out = query.new_empty(query.shape)

        last_state = torch.cat(last_states, dim=0) if last_states else None
        return core_attn_out, last_state

    _ascend_chunk_gated_delta_rule.__name__ = "ascend_chunk_gated_delta_rule"
    return _ascend_chunk_gated_delta_rule


def patch_triton_placeholder_math_symbols() -> None:
    """Make vLLM's Triton placeholder import-compatible with FLA modules."""
    try:
        from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
    except Exception:
        return

    if HAS_TRITON:
        return

    missing_symbols: dict[object, tuple[str, ...]] = {
        tl: ("exp", "log", "log2", "sqrt", "sigmoid"),
        tldevice: ("fast_expf", "fast_logf", "fast_log2f", "fast_dividef"),
    }
    for module, names in missing_symbols.items():
        for name in names:
            if not hasattr(module, name):
                setattr(module, name, _missing_triton_runtime_symbol)

    python_symbols = {
        "next_power_of_2": _next_power_of_2,
        "cdiv": _cdiv,
    }
    for name, value in python_symbols.items():
        if not hasattr(triton, name):
            setattr(triton, name, value)


def patch_gdn_aclgraph_support() -> None:
    """Expose Ascend ACL graph support metadata for vLLM's GDN builder."""
    try:
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
        from vllm.v1.attention.backends.utils import AttentionCGSupport
    except Exception:
        return

    if not hasattr(GDNAttentionMetadataBuilder, "aclgraph_support"):
        GDNAttentionMetadataBuilder.aclgraph_support = (
            AttentionCGSupport.UNIFORM_BATCH
        )


def patch_plugin_causal_conv1d_ops(qwen3_next_model_cls: object) -> bool:
    """Use Ascend-safe causal conv ops for the active plugin variant."""
    if find_spec("vllm_ascend") is None:
        return False

    try:
        from vllm_ascend.ops.casual_conv1d import (
            causal_conv1d_fn,
            causal_conv1d_update_npu,
        )
    except ImportError:
        causal_conv1d_fn = _torch_causal_conv1d_fn
        causal_conv1d_update_npu = _torch_causal_conv1d_update

    module_name = getattr(qwen3_next_model_cls, "__module__", "")
    if not module_name:
        return

    variant_module = import_module(module_name)
    variant_module.causal_conv1d_fn = causal_conv1d_fn
    variant_module.causal_conv1d_update = causal_conv1d_update_npu
    return True


def patch_plugin_fla_ops(qwen3_next_model_cls: object) -> None:
    """Use Ascend-safe FLA ops for the active plugin variant."""
    try:
        from vllm_ascend.ops.fla import torch_chunk_gated_delta_rule
    except ImportError:
        return
    try:
        from vllm.model_executor.layers.fla.ops import layernorm_guard

        layernorm_guard.LayerNormFn = _TorchLayerNormFn
    except Exception:
        pass

    module_name = getattr(qwen3_next_model_cls, "__module__", "")
    if not module_name:
        return

    variant_module = import_module(module_name)
    variant_module.chunk_gated_delta_rule = _wrap_ascend_chunk_gated_delta_rule(
        torch_chunk_gated_delta_rule
    )
    variant_module.fused_recurrent_gated_delta_rule = (
        _torch_fused_recurrent_gated_delta_rule
    )


def patch_ascend_qwen3_next_registration(
    qwen3_next_model_cls: object,
    qwen3_next_mtp_cls: object,
) -> None:
    """Keep plugin Qwen3Next registrations after vllm-ascend model hooks run."""
    try:
        import vllm_ascend
    except ImportError:
        return

    if getattr(vllm_ascend, "_qwen3_next_plugin_registration_patch", False):
        return

    original_register_model = getattr(vllm_ascend, "register_model", None)
    if original_register_model is None:
        return

    def patched_register_model(*args, **kwargs):
        result = original_register_model(*args, **kwargs)
        try:
            from vllm import ModelRegistry

            ModelRegistry.register_model("Qwen3NextForCausalLM", qwen3_next_model_cls)
            ModelRegistry.register_model("Qwen3NextMTP", qwen3_next_mtp_cls)
        except Exception:
            pass
        return result

    vllm_ascend.register_model = patched_register_model
    vllm_ascend._qwen3_next_plugin_registration_patch = True
