from types import SimpleNamespace

from vllm_qwen3_next_plugin.compat.speculative import (
    find_qwen3_next_mtp_draft_layer_names,
    get_metadata_builder_for_layer,
)


def test_find_qwen3_next_mtp_draft_layer_names_uses_global_layer_indices() -> None:
    layer_names = [
        "model.layers.0.linear_attn",
        "model.layers.55.linear_attn",
        "mtp.layers.56.linear_attn",
        "mtp.layers.57.linear_attn",
    ]

    draft_layer_names = find_qwen3_next_mtp_draft_layer_names(
        layer_names, num_hidden_layers=56
    )

    assert draft_layer_names == [
        "mtp.layers.56.linear_attn",
        "mtp.layers.57.linear_attn",
    ]


def test_get_metadata_builder_for_layer_uses_matching_group_and_ubatch() -> None:
    runner = SimpleNamespace(
        attn_groups=[
            [
                SimpleNamespace(
                    layer_names=["model.layers.0.linear_attn"],
                    metadata_builders=["wrong-0", "wrong-1"],
                )
            ],
            [
                SimpleNamespace(
                    layer_names=["mtp.layers.56.linear_attn"],
                    metadata_builders=["right-0", "right-1"],
                )
            ],
        ]
    )

    builder = get_metadata_builder_for_layer(
        runner, "mtp.layers.56.linear_attn", ubatch_id=1
    )

    assert builder == "right-1"
