from __future__ import annotations

import os
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
            decay_t = g_t.exp()
            v_beta = v_t * beta_t.unsqueeze(-1)
            k_beta = k_t * beta_t.unsqueeze(-1)
            v_prime = torch.einsum(
                "hk,hkv->hv", k_beta * decay_t.unsqueeze(-1), state
            )
            v_new = v_beta - v_prime
            out_t = (
                torch.einsum("hk,hkv->hv", q_t * scale * decay_t.unsqueeze(-1), state)
                + (q_t * scale * k_t).sum(dim=-1, keepdim=True) * v_new
            )
            out[:, tok_idx, ...] = out_t.to(v.dtype).unsqueeze(0)
            state = state * decay_t.view(-1, 1, 1) + torch.einsum(
                "hk,hv->hkv", k_t, v_new
            )
            if ssm_state_indices is not None and ssm_state_indices.ndim == 2:
                final_state[cache_idx] = state.to(final_state.dtype)
        if ssm_state_indices is None or ssm_state_indices.ndim == 1:
            final_state[cache_idx] = state.to(final_state.dtype)
    return out, final_state


def _torch_recurrent_gated_delta_rule_decode(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None,
    state: torch.Tensor,
    scale: float,
    ssm_state_indices: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Run recurrent decode without scalar host synchronization.

    The vLLM decode metadata supplies one distinct cache slot for every active
    sequence.  Speculative multi-token state indices are two-dimensional and do
    not enter this fallback.  Keep cache selection as a tensor to avoid
    ``.item()`` device synchronization, but retain the original per-token math
    order: a fully batched einsum changes rounding enough to alter long greedy
    generations after the recurrent error accumulates through 49 GDN layers.
    """
    if beta is None:
        beta = torch.ones_like(g)
    target_heads = max(query.shape[1], key.shape[1], value.shape[1], beta.shape[1])
    if key.shape[1] != target_heads:
        if target_heads % key.shape[1] != 0:
            raise RuntimeError(
                f"Cannot expand {key.shape[1]} key heads to {target_heads} target heads"
            )
        repeat = target_heads // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
    if query.shape[1] != target_heads:
        if target_heads % query.shape[1] != 0:
            raise RuntimeError(
                f"Cannot expand {query.shape[1]} query heads to {target_heads} target heads"
            )
        query = query.repeat_interleave(target_heads // query.shape[1], dim=1)
    if value.shape[1] != target_heads:
        if target_heads % value.shape[1] != 0:
            raise RuntimeError(
                f"Cannot expand {value.shape[1]} value heads to {target_heads} target heads"
            )
        value = value.repeat_interleave(target_heads // value.shape[1], dim=1)
    if beta.shape[1] != target_heads:
        if target_heads % beta.shape[1] != 0:
            raise RuntimeError(
                f"Cannot expand {beta.shape[1]} beta heads to {target_heads} target heads"
            )
        beta = beta.repeat_interleave(target_heads // beta.shape[1], dim=1)
    output = torch.zeros_like(value)
    for token_idx in range(query.shape[0]):
        cache_idx = ssm_state_indices[token_idx : token_idx + 1]
        valid = cache_idx.ne(PAD_SLOT_ID)
        safe_cache_idx = torch.where(valid, cache_idx, torch.zeros_like(cache_idx))
        state_t = state.index_select(0, safe_cache_idx).squeeze(0).float()
        q_t = query[token_idx].float()
        k_t = key[token_idx].float()
        v_t = value[token_idx].float()
        g_t = g[token_idx].float()
        beta_t = beta[token_idx].float()
        decay_t = g_t.exp()
        v_beta = v_t * beta_t.unsqueeze(-1)
        k_beta = k_t * beta_t.unsqueeze(-1)
        v_prime = torch.einsum(
            "hk,hkv->hv", k_beta * decay_t.unsqueeze(-1), state_t
        )
        v_new = v_beta - v_prime
        out_t = (
            torch.einsum(
                "hk,hkv->hv", q_t * scale * decay_t.unsqueeze(-1), state_t
            )
            + (q_t * scale * k_t).sum(dim=-1, keepdim=True) * v_new
        )
        output[token_idx] = out_t.to(value.dtype) * valid.to(value.dtype)
        next_state = state_t * decay_t.view(-1, 1, 1) + torch.einsum(
            "hk,hv->hkv", k_t, v_new
        )
        next_state = torch.where(valid.view(1, 1, 1), next_state, state_t)
        state.index_copy_(0, safe_cache_idx, next_state.unsqueeze(0).to(state.dtype))
    return output


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


def patch_ascend_attention_direct_call_trace_flag() -> None:
    """Optionally bypass the nested Ascend attention custom-op dispatch."""
    if os.getenv("VLLM_QWEN3_NEXT_FORCE_DIRECT_ASCEND_ATTENTION") != "1":
        return
    if find_spec("vllm_ascend") is None:
        return

    try:
        from vllm.forward_context import get_forward_context
        from vllm.attention.layer import Attention
    except Exception:
        return
    if getattr(Attention.forward, "_qwen3_next_force_direct_attention", False):
        return

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self.calculate_kv_scales:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata.enable_kv_scales_calculation:
                self.calc_kv_scales(query, key, value)

        output_dtype = query.dtype
        if self.query_quant is not None:
            query, _ = self.query_quant(query, self._q_scale)

        if self.use_output:
            output_shape = output_shape if output_shape is not None else query.shape
            output = torch.empty(
                output_shape,
                dtype=output_dtype,
                device=query.device,
            )
            hidden_size = output_shape[-1]
            if not self.use_mla:
                query = query.view(-1, self.num_heads, self.head_size)
                output = output.view(-1, self.num_heads, self.head_size)
                if key is not None:
                    key = key.view(-1, self.num_kv_heads, self.head_size)
                if value is not None:
                    value = value.view(-1, self.num_kv_heads, self.head_size)
            if self.use_direct_call:
                forward_context = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                self.impl.forward(
                    self,
                    query,
                    key,
                    value,
                    self_kv_cache,
                    attn_metadata,
                    output=output,
                    trace_flag=False,
                )
            else:
                torch.ops.vllm.unified_attention_with_output(
                    query, key, value, output, self.layer_name
                )
            return output.view(-1, hidden_size)

        if self.use_direct_call:
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            return self.impl.forward(
                self,
                query,
                key,
                value,
                self_kv_cache,
                attn_metadata,
                trace_flag=False,
            )
        return torch.ops.vllm.unified_attention(query, key, value, self.layer_name)

    forward._qwen3_next_force_direct_attention = True
    Attention.forward = forward
    print("[vLLM Plugin] Forcing Ascend direct attention trace_flag=False")


def _torch_decode_attention(
    impl,
    query: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    key_cache = impl.key_cache
    value_cache = impl.value_cache
    debug = os.getenv("VLLM_QWEN3_NEXT_DEBUG_TORCH_DECODE_ATTENTION") == "1"
    if key_cache is None or value_cache is None:
        if debug:
            print(
                "Qwen3Next torch decode attention",
                {
                    "stage": "missing_cache",
                    "query_shape": list(query.shape),
                    "attn_state": str(getattr(attn_metadata, "attn_state", None)),
                },
            )
        return output.zero_()

    output.zero_()
    block_size = key_cache.shape[1]
    num_tokens = query.shape[0]
    seq_lens = attn_metadata.seq_lens.tolist()
    block_tables = attn_metadata.block_tables
    kv_repeat = impl.num_heads // impl.num_kv_heads
    gathered_key_abs_sum = 0.0
    gathered_value_abs_sum = 0.0
    gathered_token_count = 0

    for req_idx, seq_len in enumerate(seq_lens[:num_tokens]):
        seq_len = int(seq_len)
        if seq_len <= 0:
            continue
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        blocks_needed = _cdiv(seq_len, block_size)
        for block_pos in range(blocks_needed):
            block_id = int(block_tables[req_idx, block_pos].item())
            if block_id < 0:
                continue
            token_start = block_pos * block_size
            take_len = min(block_size, seq_len - token_start)
            keys.append(key_cache[block_id, :take_len])
            values.append(value_cache[block_id, :take_len])
        if not keys:
            continue
        key_seq = torch.cat(keys, dim=0).float()
        value_seq = torch.cat(values, dim=0).float()
        if debug:
            gathered_key_abs_sum += float(key_seq.abs().sum().cpu().item())
            gathered_value_abs_sum += float(value_seq.abs().sum().cpu().item())
            gathered_token_count += int(key_seq.shape[0])
        if kv_repeat != 1:
            key_seq = key_seq.repeat_interleave(kv_repeat, dim=1)
            value_seq = value_seq.repeat_interleave(kv_repeat, dim=1)
        q = query[req_idx].float()
        scores = torch.einsum("hd,thd->ht", q, key_seq) * impl.scale
        probs = torch.softmax(scores, dim=-1)
        output[req_idx] = torch.einsum("ht,thd->hd", probs, value_seq).to(output.dtype)
    if debug:
        output_float = output.detach().float()
        print(
            "Qwen3Next torch decode attention",
            {
                "stage": "computed",
                "query_shape": list(query.shape),
                "output_shape": list(output.shape),
                "output_abs_sum": float(output_float.abs().sum().cpu().item()),
                "output_max_abs": float(output_float.abs().max().cpu().item())
                if output_float.numel()
                else 0.0,
                "key_cache_shape": list(key_cache.shape),
                "value_cache_shape": list(value_cache.shape),
                "gathered_key_abs_sum": gathered_key_abs_sum,
                "gathered_value_abs_sum": gathered_value_abs_sum,
                "gathered_token_count": gathered_token_count,
                "block_tables_head": block_tables.detach().cpu().reshape(-1)[:8].tolist(),
                "slot_mapping_head": getattr(
                    attn_metadata, "slot_mapping", torch.tensor([], device=query.device)
                )
                .detach()
                .cpu()
                .reshape(-1)[:8]
                .tolist(),
                "seq_lens_head": [int(x) for x in seq_lens[: min(4, len(seq_lens))]],
                "attn_state": str(getattr(attn_metadata, "attn_state", None)),
            },
        )
    return output


def _derive_slot_mapping_from_block_tables(attn_metadata, block_size: int) -> torch.Tensor:
    seq_lens = attn_metadata.seq_lens
    query_lens = getattr(attn_metadata, "query_lens", None)
    block_tables = attn_metadata.block_tables
    if query_lens is None:
        query_lens = torch.ones_like(seq_lens)
    if not torch.is_tensor(query_lens):
        query_lens = torch.tensor(query_lens, device=seq_lens.device)
    seq_lens = seq_lens.to(device=block_tables.device, dtype=torch.long)
    query_lens = query_lens.to(device=block_tables.device, dtype=torch.long)

    slots: list[torch.Tensor] = []
    for req_idx in range(int(query_lens.numel())):
        query_len = int(query_lens[req_idx].item())
        seq_len = int(seq_lens[req_idx].item())
        start_pos = seq_len - query_len
        for position in range(start_pos, seq_len):
            block_pos = position // block_size
            block_offset = position % block_size
            block_id = block_tables[req_idx, block_pos].to(torch.long)
            slots.append(block_id * block_size + block_offset)
    if not slots:
        return torch.empty(0, dtype=torch.long, device=block_tables.device)
    return torch.stack(slots).to(torch.long)


def _write_kv_cache_by_slot_mapping(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    block_size = key_cache.shape[1]
    slots = slot_mapping.to(device=key.device, dtype=torch.long)
    valid = slots >= 0
    if not bool(valid.any().item()):
        return
    slots = slots[valid]
    block_indices = slots // block_size
    block_offsets = slots % block_size
    key_cache[block_indices, block_offsets] = key[: slots.numel()].to(key_cache.dtype)
    value_cache[block_indices, block_offsets] = value[: slots.numel()].to(
        value_cache.dtype
    )


def patch_ascend_torch_decode_attention() -> None:
    """Optionally replace Ascend standard decode attention with a torch fallback."""
    if os.getenv("VLLM_QWEN3_NEXT_FORCE_TORCH_DECODE_ATTENTION") != "1":
        return
    if find_spec("vllm_ascend") is None:
        return
    try:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    except Exception:
        return
    if getattr(
        AscendAttentionBackendImpl.forward,
        "_qwen3_next_torch_decode_attention_forward",
        False,
    ):
        return

    original_forward = AscendAttentionBackendImpl.forward

    def _should_use_torch_decode(query: torch.Tensor, attn_metadata) -> bool:
        if attn_metadata is None:
            if os.getenv("VLLM_QWEN3_NEXT_DEBUG_TORCH_DECODE_ATTENTION") == "1":
                print(
                    "Qwen3Next torch decode attention",
                    {"stage": "predicate", "reason": "missing_metadata"},
                )
            return False
        if getattr(attn_metadata, "num_actual_tokens", query.shape[0]) != query.shape[0]:
            if os.getenv("VLLM_QWEN3_NEXT_DEBUG_TORCH_DECODE_ATTENTION") == "1":
                print(
                    "Qwen3Next torch decode attention",
                    {
                        "stage": "predicate",
                        "reason": "num_actual_tokens_mismatch",
                        "query_shape": list(query.shape),
                        "num_actual_tokens": getattr(attn_metadata, "num_actual_tokens", None),
                        "attn_state": str(getattr(attn_metadata, "attn_state", None)),
                    },
                )
            return False
        query_lens = getattr(attn_metadata, "query_lens", None)
        if query_lens is None:
            allowed = query.shape[0] == 1
            if os.getenv("VLLM_QWEN3_NEXT_DEBUG_TORCH_DECODE_ATTENTION") == "1":
                print(
                    "Qwen3Next torch decode attention",
                    {
                        "stage": "predicate",
                        "allowed": allowed,
                        "query_shape": list(query.shape),
                        "query_lens": None,
                        "attn_state": str(getattr(attn_metadata, "attn_state", None)),
                    },
                )
            return allowed
        if not torch.is_tensor(query_lens):
            query_lens = torch.tensor(query_lens, device=query.device)
        allowed = bool(query_lens.numel() > 0 and int(query_lens.max().item()) <= 1)
        if os.getenv("VLLM_QWEN3_NEXT_DEBUG_TORCH_DECODE_ATTENTION") == "1":
            print(
                "Qwen3Next torch decode attention",
                {
                    "stage": "predicate",
                    "allowed": allowed,
                    "query_shape": list(query.shape),
                    "query_lens_head": query_lens.detach().cpu().reshape(-1)[:4].tolist(),
                    "num_actual_tokens": getattr(attn_metadata, "num_actual_tokens", None),
                    "attn_state": str(getattr(attn_metadata, "attn_state", None)),
                },
            )
        return allowed

    def _forward_decode_only(self, query, attn_metadata, output=None):
        if output is None:
            output = torch.empty(
                query.shape[0],
                self.num_heads,
                self.head_size,
                dtype=query.dtype,
                device=query.device,
            )
        return _torch_decode_attention(self, query, attn_metadata, output)

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        trace_flag=True,
    ):
        use_torch_decode = not trace_flag and _should_use_torch_decode(
            query, attn_metadata
        )
        if os.getenv("VLLM_QWEN3_NEXT_DEBUG_TORCH_DECODE_ATTENTION") == "1":
            num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", 0) or 0
            key_float = key[:num_actual_tokens].detach().float() if key is not None else None
            value_float = (
                value[:num_actual_tokens].detach().float() if value is not None else None
            )
            print(
                "Qwen3Next torch decode attention",
                {
                    "stage": "forward",
                    "trace_flag": trace_flag,
                    "use_torch_decode": use_torch_decode,
                    "query_shape": list(query.shape),
                    "key_abs_sum": float(key_float.abs().sum().cpu().item())
                    if key_float is not None and key_float.numel()
                    else 0.0,
                    "value_abs_sum": float(value_float.abs().sum().cpu().item())
                    if value_float is not None and value_float.numel()
                    else 0.0,
                    "attn_state": str(getattr(attn_metadata, "attn_state", None)),
                    "num_actual_tokens": getattr(attn_metadata, "num_actual_tokens", None),
                },
            )
        if not use_torch_decode:
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                trace_flag=trace_flag,
            )

        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(
                num_tokens,
                self.num_heads,
                self.head_size,
                dtype=query.dtype,
                device=query.device,
            )
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size).contiguous()
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            corrected_slot_mapping = _derive_slot_mapping_from_block_tables(
                attn_metadata, self.key_cache.shape[1]
            ).to(device=attn_metadata.slot_mapping.device)
            attn_metadata.slot_mapping = corrected_slot_mapping
            _write_kv_cache_by_slot_mapping(
                key[: attn_metadata.num_actual_tokens],
                value[: attn_metadata.num_actual_tokens],
                self.key_cache,
                self.value_cache,
                corrected_slot_mapping,
            )
        return _torch_decode_attention(self, query, attn_metadata, output).view(
            num_tokens, self.hidden_size
        )

    _forward_decode_only._qwen3_next_torch_decode_attention = True
    forward._qwen3_next_torch_decode_attention_forward = True
    AscendAttentionBackendImpl._forward_decode_only = _forward_decode_only
    AscendAttentionBackendImpl.forward = forward
    print("[vLLM Plugin] Forcing Ascend standard decode attention torch fallback")


def patch_ascend_gdn_hd64_runtime_workarounds() -> bool:
    """Install CANN9/vLLM-Ascend GDN workarounds for head_dim < 128.

    The trusted hybrid-GDN checkpoint uses 64-dim linear-attention heads. On the
    CANN9/vLLM-Ascend 0.20.x stack, two fast paths are numerically wrong for
    that geometry:
    - prefill chunk state carry in the AscendC/triton FLA path
    - packed recurrent decode fast path

    Keep the upstream 128-dim path unchanged, and make this patch opt-out for
    bug report minimization.
    """
    if os.getenv("VLLM_QWEN3_NEXT_DISABLE_GDN_HD64_FIX") == "1":
        return False
    if find_spec("vllm_ascend") is None:
        return False

    patched = False
    try:
        chunk_module = import_module("vllm_ascend.ops.triton.fla.chunk")
        if not getattr(chunk_module, "_qwen3_next_hd64_padded_chunk_fwd", False):
            original_chunk_fwd = chunk_module.chunk_gated_delta_rule_fwd

            def chunk_gated_delta_rule_fwd(
                q,
                k,
                v,
                g,
                beta,
                scale,
                initial_state,
                output_final_state,
                cu_seqlens=None,
                prebuilt_meta=None,
            ):
                if k.shape[-1] >= 128 or chunk_module.get_pcp_group().world_size != 1:
                    return original_chunk_fwd(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        scale,
                        initial_state,
                        output_final_state,
                        cu_seqlens=cu_seqlens,
                        prebuilt_meta=prebuilt_meta,
                    )

                chunk_size = 64
                block_indices_cumsum = (
                    None if prebuilt_meta is None else prebuilt_meta.block_indices_cumsum
                )
                chunk_indices_chunk64 = (
                    None if prebuilt_meta is None else prebuilt_meta.chunk_indices_chunk64
                )
                chunk_offsets_chunk64 = (
                    None if prebuilt_meta is None else prebuilt_meta.chunk_offsets_chunk64
                )
                chunk_indices_large_block = (
                    None if prebuilt_meta is None else prebuilt_meta.chunk_indices_large_block
                )

                g = chunk_module.chunk_local_cumsum(
                    g,
                    chunk_size=chunk_size,
                    cu_seqlens=cu_seqlens,
                    block_indices=block_indices_cumsum,
                )
                A = chunk_module.chunk_scaled_dot_kkt_fwd(
                    k=k,
                    beta=beta,
                    g_cumsum=g,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices_chunk64,
                    output_dtype=torch.float32,
                )
                A = chunk_module.solve_tril(
                    A=A,
                    cu_seqlens=cu_seqlens,
                    chunk_indices_large_block=chunk_indices_large_block,
                    chunk_indices_bt=chunk_indices_chunk64,
                    output_dtype=k.dtype,
                )
                w, u = chunk_module.recompute_w_u_fwd(
                    k=k,
                    v=v,
                    beta=beta,
                    A=A,
                    g_cumsum=g,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices_chunk64,
                )

                real_k_dim = k.shape[-1]
                real_v_dim = u.shape[-1]
                target_dim = 128

                def pad_last(t: torch.Tensor | None, target: int):
                    if t is None or t.shape[-1] >= target:
                        return t
                    return F.pad(t, (0, target - t.shape[-1]))

                k_padded = pad_last(k, target_dim)
                q_padded = pad_last(q, target_dim)
                w_padded = pad_last(w, target_dim)
                u_padded = pad_last(u, target_dim)
                if initial_state is None:
                    initial_state_padded = None
                else:
                    initial_state_padded = initial_state
                    if initial_state_padded.shape[-1] < target_dim:
                        initial_state_padded = F.pad(
                            initial_state_padded,
                            (0, target_dim - initial_state_padded.shape[-1]),
                        )
                    if initial_state_padded.shape[-2] < target_dim:
                        initial_state_padded = F.pad(
                            initial_state_padded,
                            (0, 0, 0, target_dim - initial_state_padded.shape[-2]),
                        )

                h, v_new, final_state = chunk_module.chunk_gated_delta_rule_fwd_h(
                    k=k_padded,
                    w=w_padded,
                    u=u_padded,
                    g=g,
                    initial_state=initial_state_padded,
                    output_final_state=True,
                    chunk_size=chunk_size,
                    save_new_value=True,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices_chunk64,
                    chunk_offsets=chunk_offsets_chunk64,
                )
                o = chunk_module.chunk_fwd_o(
                    q=q_padded,
                    k=k_padded,
                    v=v_new,
                    h=h,
                    g=g,
                    scale=scale,
                    cu_seqlens=cu_seqlens,
                    chunk_size=chunk_size,
                    chunk_offsets=chunk_offsets_chunk64,
                )
                return (
                    g,
                    o[..., :real_v_dim].contiguous(),
                    A,
                    final_state[..., :real_k_dim, :real_v_dim].contiguous(),
                    None,
                    None,
                    None,
                )

            chunk_gated_delta_rule_fwd._qwen3_next_hd64_padded_chunk_fwd = True
            chunk_gated_delta_rule_fwd._qwen3_next_original = original_chunk_fwd
            chunk_module.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
            patched = True
    except Exception:
        pass

    try:
        gdn_module = import_module("vllm.model_executor.layers.mamba.gdn_linear_attn")
        GatedDeltaNetAttention = gdn_module.GatedDeltaNetAttention
        original_forward_core = GatedDeltaNetAttention._forward_core
        if not getattr(original_forward_core, "_qwen3_next_hd64_packed_decode_guard", False):

            def _forward_core(
                self,
                mixed_qkv,
                b,
                a,
                core_attn_out,
            ):
                original_enabled = self.enable_packed_recurrent_decode
                use_packed_recurrent_decode = (
                    self.enable_packed_recurrent_decode
                    and self.head_k_dim >= 128
                    and self.head_v_dim >= 128
                )
                if not use_packed_recurrent_decode:
                    self.enable_packed_recurrent_decode = False
                try:
                    return original_forward_core(self, mixed_qkv, b, a, core_attn_out)
                finally:
                    self.enable_packed_recurrent_decode = original_enabled

            _forward_core._qwen3_next_hd64_packed_decode_guard = True
            _forward_core._qwen3_next_original = original_forward_core
            GatedDeltaNetAttention._forward_core = _forward_core
            patched = True
    except Exception:
        pass

    try:
        ascend_gdn_module = import_module("vllm_ascend.ops.gdn")
        AscendGatedDeltaNetAttention = ascend_gdn_module.AscendGatedDeltaNetAttention
        original_forward_core = AscendGatedDeltaNetAttention._forward_core
        if not getattr(original_forward_core, "_qwen3_next_hd64_decode_guard", False):

            def _forward_core(
                self,
                mixed_qkv,
                b,
                a,
                core_attn_out,
            ):
                if self.head_k_dim >= 128 and self.head_v_dim >= 128:
                    return original_forward_core(self, mixed_qkv, b, a, core_attn_out)

                original_recurrent = torch.ops._C_ascend.npu_recurrent_gated_delta_rule

                def npu_recurrent_gated_delta_rule(*args, **kwargs):
                    if kwargs and "query" in kwargs:
                        query = kwargs["query"]
                        ssm_state_indices = kwargs.get("ssm_state_indices")
                    else:
                        query = args[0] if args else None
                        ssm_state_indices = (
                            kwargs.get("ssm_state_indices")
                            if kwargs
                            else args[7] if len(args) > 7 else None
                        )
                    if (
                        query is not None
                        and ssm_state_indices is not None
                        and query.ndim == 3
                        and ssm_state_indices.ndim == 1
                        and query.shape[-1] < 128
                    ):
                        if args:
                            raise TypeError(
                                "Positional npu_recurrent_gated_delta_rule fallback is not supported"
                            )
                        return _torch_recurrent_gated_delta_rule_decode(**kwargs)
                    return original_recurrent(*args, **kwargs)

                torch.ops._C_ascend.npu_recurrent_gated_delta_rule = (
                    npu_recurrent_gated_delta_rule
                )
                try:
                    return original_forward_core(self, mixed_qkv, b, a, core_attn_out)
                finally:
                    torch.ops._C_ascend.npu_recurrent_gated_delta_rule = original_recurrent

            _forward_core._qwen3_next_hd64_decode_guard = True
            _forward_core._qwen3_next_original = original_forward_core
            AscendGatedDeltaNetAttention._forward_core = _forward_core
            patched = True
    except Exception:
        pass

    if patched:
        print("[vLLM Plugin] Installed Ascend GDN head_dim<128 runtime workarounds")
    return patched


def ensure_ascend_custom_ops_registered() -> bool:
    """Register vLLM-Ascend OOT ops before plugin model construction.

    vLLM-Ascend normally registers OOT CustomOp/PluggableLayer replacements
    during worker setup. The plugin model can be imported earlier than that, so
    make the dependency explicit here. This is required for GDN layers to be
    instantiated as AscendGatedDeltaNetAttention instead of the upstream class.
    """
    if find_spec("vllm_ascend") is None:
        return False
    try:
        from vllm_ascend.utils import register_ascend_customop

        register_ascend_customop()
    except Exception:
        return False
    return True


def patch_plugin_causal_conv1d_ops(qwen3_next_model_cls: object) -> bool:
    """Use Ascend-safe causal conv ops for the active plugin variant."""
    if find_spec("vllm_ascend") is None:
        return False

    if os.getenv("VLLM_QWEN3_NEXT_FORCE_TORCH_CAUSAL_CONV") == "1":
        causal_conv1d_fn = _torch_causal_conv1d_fn
        causal_conv1d_update_npu = _torch_causal_conv1d_update
    else:
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


def patch_plugin_qwen3_next_attention(qwen3_next_model_cls: object) -> bool:
    """Apply vLLM-Ascend's Qwen3NextAttention forward patch to plugin variants."""
    if find_spec("vllm_ascend") is None:
        return False
    module_name = getattr(qwen3_next_model_cls, "__module__", "")
    if not module_name:
        return False
    variant_module = import_module(module_name)
    attention_cls = getattr(variant_module, "Qwen3NextAttention", None)
    if attention_cls is None:
        return False
    if getattr(attention_cls.forward, "_qwen3_next_plugin_ascend_attention", False):
        return True

    def forward(self, positions: torch.Tensor, output: torch.Tensor, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        if "qwen3_5" in self.config.model_type:
            cos_sin = self.rotary_emb.cos_sin_cache[positions]
            if cos_sin.device != qkv.device:
                cos_sin = cos_sin.to(qkv.device)
            if cos_sin.dtype != qkv.dtype:
                cos_sin = cos_sin.to(qkv.dtype)

            q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + self.q_norm.weight,
                k_weight=1.0 + self.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=self.config.rms_norm_eps,
                mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved,
                rope_dim=self.rotary_emb.rotary_dim,
                has_gate=self.attn_output_gate,
            )
        else:
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

            if getattr(self, "enable_qk_norm", True):
                q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
                    -1, self.num_heads * self.head_dim
                )
                k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
                    -1, self.num_kv_heads * self.head_dim
                )

            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)

    forward._qwen3_next_plugin_ascend_attention = True
    attention_cls.forward = forward
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
