"""On-disk checkpoint format.

The contract is that what this package writes is what
`DFlashQwen3ForCausalLM.load_weights` reads. What can fail silently -- which tensors
are omitted, and whether the fused/split conversion preserves values rather than
merely names -- is checkable without an engine.
"""

import json

import pytest
import torch

from tests.conftest import (
    checkpoint_writer,
    draft_config_builder,
    head_factory,
    head_weights,
    make_arch,
    target_layer_planner,
    weight_name_rewriter,
)


def test_default_layer_ids_reproduce_the_reference_checkpoint():
    """`z-lab/Qwen3-4B-DFlash-b16` is 36 target layers, 5 features, `[1,9,17,25,33]`.

    These ids decide which target activations the head trains on, so drifting them
    trains a head that cannot be compared against any published number.
    """
    assert target_layer_planner.plan(36, 5) == [1, 9, 17, 25, 33]
    assert target_layer_planner.plan(36, 1) == [1]
    assert len(target_layer_planner.plan(36, 3)) == 3


def test_layer_ids_stay_inside_the_target():
    """Every id must leave room for vLLM's `+1` into aux numbering."""
    for num_target_layers in (8, 12, 28, 36, 64):
        for num_features in (1, 2, 3, 5, 8):
            if num_features > num_target_layers - 1:
                continue
            layer_ids = target_layer_planner.plan(num_target_layers, num_features)
            assert len(layer_ids) == num_features
            assert len(set(layer_ids)) == num_features, (
                f"duplicate ids {layer_ids}"
            )
            assert min(layer_ids) >= 1
            assert max(layer_ids) + 1 <= num_target_layers, (
                f"{layer_ids} escapes {num_target_layers} layers"
            )


def test_too_many_features_for_a_shallow_target_is_refused():
    """Distinct ids simply do not exist, so failing beats emitting duplicates that
    would make `fc` read the same layer twice."""
    with pytest.raises(ValueError, match="cannot place 8 distinct feature layers"):
        target_layer_planner.plan(8, 8)


@pytest.mark.parametrize("num_features", [0, -1])
def test_a_non_positive_feature_count_is_refused(num_features):
    with pytest.raises(ValueError, match="num_features must be >= 1"):
        target_layer_planner.plan(36, num_features)


def test_a_target_too_shallow_for_any_feature_is_refused():
    with pytest.raises(ValueError, match="cannot supply feature layers"):
        target_layer_planner.plan(1, 1)


def test_shared_tensors_are_not_written(head_arch):
    """`embed_tokens` and `lm_head` are bound from the target at serve time, so
    shipping them gives the draft a second copy that can only drift."""
    draft_head = head_factory.create(head_arch)
    exported_names = {
        name
        for name, _ in weight_name_rewriter.rewrite(
            head_weights.export(draft_head), head_arch
        )
    }
    assert exported_names, "conversion produced nothing"
    for shared_name in weight_name_rewriter.SHARED_WITH_TARGET:
        assert not any(shared_name in name for name in exported_names), (
            f"{shared_name} leaked into export"
        )


def test_fused_projections_split_by_value(head_arch):
    """Splitting must preserve values, not just produce plausible names.

    A transposed or mis-ordered split still yields tensors of the right shape and a
    head that trains -- against a projection the serving path will never apply.
    """
    draft_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(0)
    )
    exported = dict(
        weight_name_rewriter.rewrite(
            head_weights.export(draft_head), head_arch
        )
    )

    for layer_index in range(head_arch.num_layers):
        layer = draft_head.model.layers[layer_index]
        fused_qkv = layer.self_attn.qkv_proj.weight
        query = exported[f"layers.{layer_index}.self_attn.q_proj.weight"]
        key = exported[f"layers.{layer_index}.self_attn.k_proj.weight"]
        value = exported[f"layers.{layer_index}.self_attn.v_proj.weight"]
        assert query.shape == (head_arch.q_size, head_arch.hidden_size)
        assert key.shape == (head_arch.kv_size, head_arch.hidden_size)
        assert value.shape == (head_arch.kv_size, head_arch.hidden_size)
        # Re-fusing is what `hf_to_vllm_mapper`'s stacked rules do on load.
        torch.testing.assert_close(
            torch.cat([query, key, value], dim=0), fused_qkv
        )

        fused_gate_up = layer.mlp.gate_up_proj.weight
        gate = exported[f"layers.{layer_index}.mlp.gate_proj.weight"]
        up = exported[f"layers.{layer_index}.mlp.up_proj.weight"]
        assert gate.shape == (
            head_arch.intermediate_size,
            head_arch.hidden_size,
        )
        torch.testing.assert_close(
            torch.cat([gate, up], dim=0), fused_gate_up
        )


def test_attention_bias_is_split_too():
    """A biased attention head must split `qkv_proj.bias` the same way as the
    weight, or the loaded biases land on the wrong projections.
    """
    head_arch = make_arch(attention_bias=True)
    draft_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(0)
    )
    exported = dict(
        weight_name_rewriter.rewrite(
            head_weights.export(draft_head), head_arch
        )
    )
    fused_bias = draft_head.model.layers[0].self_attn.qkv_proj.bias
    split_biases = [
        exported[f"layers.0.self_attn.{name}_proj.bias"] for name in ("q", "k", "v")
    ]
    torch.testing.assert_close(torch.cat(split_biases, dim=0), fused_bias)


def test_unfused_tensors_pass_through_untouched(head_arch):
    draft_head = head_factory.create(head_arch)
    exported = dict(
        weight_name_rewriter.rewrite(
            head_weights.export(draft_head), head_arch
        )
    )
    for name in ("fc.weight", "hidden_norm.weight", "norm.weight"):
        assert name in exported
    assert exported["fc.weight"].shape == (
        head_arch.hidden_size,
        head_arch.feature_width,
    )


def test_config_matches_the_head_it_ships_with(head_arch):
    draft_config = draft_config_builder.build(
        head_arch, target_layer_ids=[1, 2, 3], num_target_layers=8, dtype="bfloat16"
    )
    assert draft_config["architectures"] == ["DFlashDraftModel"]
    assert draft_config["model_type"] == "qwen3"
    assert draft_config["num_hidden_layers"] == head_arch.num_layers
    assert len(draft_config["layer_types"]) == head_arch.num_layers
    assert draft_config["dflash_config"]["mask_token_id"] == head_arch.mask_token_id
    assert draft_config["dflash_config"]["target_layer_ids"] == [1, 2, 3]
    assert draft_config["block_size"] == head_arch.block_size


def test_config_omits_token_ids_that_were_not_given(head_arch):
    draft_config = draft_config_builder.build(
        head_arch, target_layer_ids=[1, 2, 3], num_target_layers=8
    )
    assert "bos_token_id" not in draft_config
    assert "eos_token_id" not in draft_config


def test_config_rejects_a_feature_count_mismatch(head_arch):
    """The fc width is `num_features * hidden`, so a wrong id count builds a head
    whose projection cannot consume the features being captured."""
    with pytest.raises(ValueError, match="fc would be built at the wrong width"):
        draft_config_builder.build(
            head_arch,
            target_layer_ids=list(range(head_arch.num_features + 1)),
            num_target_layers=8,
        )


def test_write_checkpoint_round_trips(tmp_path, head_arch):
    draft_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(1)
    )
    draft_config = draft_config_builder.build(
        head_arch,
        target_layer_ids=list(range(1, head_arch.num_features + 1)),
        num_target_layers=8,
    )
    written = checkpoint_writer.write(
        tmp_path,
        head_weights.export(draft_head),
        head_arch,
        config=draft_config,
        dtype=torch.float32,
    )

    assert written.is_file()
    written_config = json.loads(
        (tmp_path / checkpoint_writer.CONFIG_FILENAME).read_text()
    )
    assert written_config["model_type"] == "qwen3"

    from safetensors.torch import load_file

    loaded = load_file(str(written))
    expected = dict(
        weight_name_rewriter.rewrite(
            head_weights.export(draft_head), head_arch
        )
    )
    assert set(loaded) == set(expected)
    for name, tensor in expected.items():
        torch.testing.assert_close(loaded[name], tensor.detach())


def test_write_checkpoint_casts_before_writing(tmp_path, head_arch):
    """The trainable head is fp32; a checkpoint written at that precision is twice
    the size it needs to be."""
    from safetensors.torch import load_file

    draft_head = head_factory.create(head_arch)
    written = checkpoint_writer.write(
        tmp_path,
        head_weights.export(draft_head),
        head_arch,
        dtype=torch.bfloat16,
    )
    loaded = load_file(str(written))
    assert all(tensor.dtype == torch.bfloat16 for tensor in loaded.values())


def test_write_checkpoint_can_refresh_weights_only(tmp_path, head_arch):
    """Omitting the config refreshes a directory that already has one."""
    draft_head = head_factory.create(head_arch)
    checkpoint_writer.write(tmp_path, head_weights.export(draft_head), head_arch)
    assert not (tmp_path / checkpoint_writer.CONFIG_FILENAME).exists()
    assert (tmp_path / checkpoint_writer.WEIGHTS_FILENAME).is_file()
