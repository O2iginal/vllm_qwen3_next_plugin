from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


_LAYER_NAME_RE = re.compile(r"(?:^|[.])layers\.(\d+)\.(linear_attn|self_attn)$")


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    match = _LAYER_NAME_RE.search(layer_name)
    if match is None:
        return (1 << 30, layer_name)
    return (int(match.group(1)), layer_name)


def find_qwen3_next_mtp_draft_layer_names(
    layer_names: Iterable[str], *, num_hidden_layers: int
) -> list[str]:
    draft_layer_names: list[str] = []
    for layer_name in layer_names:
        match = _LAYER_NAME_RE.search(layer_name)
        if match is None:
            continue
        if int(match.group(1)) < num_hidden_layers:
            continue
        draft_layer_names.append(layer_name)
    draft_layer_names.sort(key=_layer_sort_key)
    return draft_layer_names


def get_metadata_builder_for_layer(
    runner: Any, layer_name: str, *, ubatch_id: int
) -> Any:
    for kv_cache_group in runner.attn_groups:
        for attn_group in kv_cache_group:
            if layer_name in attn_group.layer_names:
                return attn_group.metadata_builders[ubatch_id]
    raise AssertionError(
        f"Failed to find attention metadata builder for draft layer {layer_name}"
    )


def _get_attn_group_position_for_layer(
    runner: Any, layer_name: str
) -> tuple[int, int] | None:
    for kv_group_idx, kv_cache_group in enumerate(runner.attn_groups):
        for attn_group_idx, attn_group in enumerate(kv_cache_group):
            if layer_name in attn_group.layer_names:
                return kv_group_idx, attn_group_idx
    return None


def _is_qwen3_next_linear_attention_mtp(proposer: Any) -> bool:
    draft_model_config = getattr(proposer, "draft_model_config", None)
    if draft_model_config is None:
        return False
    hf_config = getattr(draft_model_config, "hf_config", None)
    if hf_config is None:
        return False
    if getattr(hf_config, "model_type", None) != "qwen3_next_mtp":
        return False
    mtp_layer_types = getattr(hf_config, "mtp_layer_types", None) or []
    return any(layer_type == "linear_attention" for layer_type in mtp_layer_types)


def _get_draft_layer_names(proposer: Any) -> list[str]:
    if hasattr(proposer, "_draft_attn_layer_names"):
        return sorted(getattr(proposer, "_draft_attn_layer_names"), key=_layer_sort_key)
    return list(getattr(proposer, "attn_layer_names", []) or [])


def _set_draft_layer_names(proposer: Any, layer_names: list[str]) -> None:
    # vLLM 0.18 renamed the public-ish list used by 0.11 to a private set and
    # initializes draft_attn_groups from it later. Keep both fields populated so
    # the same plugin wheel remains compatible with both supported branches.
    proposer.attn_layer_names = layer_names
    proposer._draft_attn_layer_names = set(layer_names)


def apply_runtime_patches() -> None:
    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.model_executor.layers.mamba.abstract import MambaBase
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
        from vllm.v1.spec_decode.eagle import EagleProposer
        from vllm.v1.worker.utils import AttentionGroup
    except ImportError:
        return

    if getattr(EagleProposer, "_qwen3_next_plugin_patch_applied", False):
        return

    original_load_model = EagleProposer.load_model
    original_propose = EagleProposer.propose
    original_initialize_attn_backend = getattr(
        EagleProposer, "initialize_attn_backend", None
    )

    def patched_load_model(self: Any, target_model: Any) -> None:
        target_mamba_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, MambaBase).keys()
        )
        original_load_model(self, target_model)
        if not _is_qwen3_next_linear_attention_mtp(self):
            return
        if _get_draft_layer_names(self):
            return

        draft_mamba_layer_names = sorted(
            set(get_layers_from_vllm_config(self.vllm_config, MambaBase).keys())
            - target_mamba_layer_names,
            key=_layer_sort_key,
        )
        if not draft_mamba_layer_names:
            draft_mamba_layer_names = find_qwen3_next_mtp_draft_layer_names(
                self.vllm_config.compilation_config.static_forward_context.keys(),
                num_hidden_layers=self.draft_model_config.hf_config.num_hidden_layers,
            )

        _set_draft_layer_names(self, draft_mamba_layer_names)

    def patched_propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_qwen3_next_linear_attention_mtp(self):
            return original_propose(self, *args, **kwargs)
        draft_layer_names = _get_draft_layer_names(self)
        if not draft_layer_names:
            return original_propose(self, *args, **kwargs)

        if hasattr(self, "draft_attn_groups"):
            # vLLM 0.18 builds dedicated draft attention groups from
            # _draft_attn_layer_names. The 0.11 runner.attn_groups swap below is
            # only needed for the older EagleProposer implementation.
            if hasattr(self, "model"):
                setattr(self.model, "_qwen3_next_mtp_spec_step_idx", 0)
            try:
                return original_propose(self, *args, **kwargs)
            finally:
                if hasattr(self, "model") and hasattr(
                    self.model, "_qwen3_next_mtp_spec_step_idx"
                ):
                    delattr(self.model, "_qwen3_next_mtp_spec_step_idx")

        chosen_layer = draft_layer_names[0]
        location = _get_attn_group_position_for_layer(self.runner, chosen_layer)
        if location is None or location == (0, 0):
            return original_propose(self, *args, **kwargs)

        kv_group_idx, attn_group_idx = location
        attn_groups = self.runner.attn_groups
        attn_groups[0][0], attn_groups[kv_group_idx][attn_group_idx] = (
            attn_groups[kv_group_idx][attn_group_idx],
            attn_groups[0][0],
        )
        if hasattr(self, "model"):
            setattr(self.model, "_qwen3_next_mtp_spec_step_idx", 0)
        try:
            return original_propose(self, *args, **kwargs)
        finally:
            if hasattr(self, "model") and hasattr(
                self.model, "_qwen3_next_mtp_spec_step_idx"
            ):
                delattr(self.model, "_qwen3_next_mtp_spec_step_idx")
            attn_groups[0][0], attn_groups[kv_group_idx][attn_group_idx] = (
                attn_groups[kv_group_idx][attn_group_idx],
                attn_groups[0][0],
            )

    if original_initialize_attn_backend is not None:

        def patched_initialize_attn_backend(
            self: Any,
            kv_cache_config: Any,
            kernel_block_sizes: list[int] | None = None,
        ) -> None:
            if not _is_qwen3_next_linear_attention_mtp(self):
                return original_initialize_attn_backend(
                    self, kv_cache_config, kernel_block_sizes
                )

            all_attn_layers = get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
            )
            layer_to_group: dict[str, tuple[int, Any]] = {}
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                for layer_name in group.layer_names:
                    layer_to_group[layer_name] = (gid, group.kv_cache_spec)

            attention_groups: dict[tuple[int, str], Any] = {}
            for layer_name in sorted(self._draft_attn_layer_names, key=_layer_sort_key):
                if layer_name not in layer_to_group:
                    continue
                gid, kv_cache_spec = layer_to_group[layer_name]
                layer_kv_cache_spec = kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]

                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                backend_key = (gid, attn_backend.full_cls_name())
                if backend_key not in attention_groups:
                    kernel_block_size = (
                        kernel_block_sizes[gid]
                        if kernel_block_sizes is not None
                        and gid < len(kernel_block_sizes)
                        else None
                    )
                    attn_group = AttentionGroup(
                        backend=attn_backend,
                        layer_names=[layer_name],
                        kv_cache_spec=layer_kv_cache_spec,
                        kv_cache_group_id=gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    attention_groups[backend_key] = attn_group
                else:
                    attention_groups[backend_key].layer_names.append(layer_name)

            if not attention_groups:
                return original_initialize_attn_backend(
                    self, kv_cache_config, kernel_block_sizes
                )

            self.draft_attn_groups = list(attention_groups.values())
            self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
            self.block_size = (
                self.draft_attn_groups[0]
                .get_metadata_builder()
                .kv_cache_spec.block_size
            )

    EagleProposer.load_model = patched_load_model
    EagleProposer.propose = patched_propose
    if original_initialize_attn_backend is not None:
        EagleProposer.initialize_attn_backend = patched_initialize_attn_backend
    EagleProposer._qwen3_next_plugin_patch_applied = True
