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
    layer_planner,
    make_arch,
    weight_naming,
)


def test_default_layer_ids_reproduce_the_reference_checkpoint():
    """`z-lab/Qwen3-4B-DFlash-b16` is 36 target layers, 5 features, `[1,9,17,25,33]`.

    These ids decide which target activations the head trains on, so drifting them
    trains a head that cannot be compared against any published number.
    """
    assert layer_planner.plan(36, 5) == [1, 9, 17, 25, 33]
    assert layer_planner.plan(36, 1) == [1]
    assert len(layer_planner.plan(36, 3)) == 3


def test_layer_ids_stay_inside_the_target():
    """Every id must leave room for vLLM's `+1` into aux numbering."""
    for num_layers in (8, 12, 28, 36, 64):
        for features in (1, 2, 3, 5, 8):
            if features > num_layers - 1:
                continue
            ids = layer_planner.plan(num_layers, features)
            assert len(ids) == features
            assert len(set(ids)) == features, f"duplicate ids {ids}"
            assert min(ids) >= 1
            assert max(ids) + 1 <= num_layers, f"{ids} escapes {num_layers} layers"


def test_too_many_features_for_a_shallow_target_is_refused():
    """Distinct ids simply do not exist, so failing beats emitting duplicates that
    would make `fc` read the same layer twice."""
    with pytest.raises(ValueError, match="cannot place 8 distinct feature layers"):
        layer_planner.plan(8, 8)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_feature_count_is_refused(bad):
    with pytest.raises(ValueError, match="num_features must be >= 1"):
        layer_planner.plan(36, bad)


def test_a_target_too_shallow_for_any_feature_is_refused():
    with pytest.raises(ValueError, match="cannot supply feature layers"):
        layer_planner.plan(1, 1)


def test_shared_tensors_are_not_written(arch):
    """`embed_tokens` and `lm_head` are bound from the target at serve time, so
    shipping them gives the draft a second copy that can only drift."""
    head = head_factory.create(arch)
    names = {name for name, _ in weight_naming.rewrite(head_weights.export(head), arch)}
    assert names, "conversion produced nothing"
    for shared in weight_naming.SHARED_WITH_TARGET:
        assert not any(shared in name for name in names), f"{shared} leaked into export"


def test_fused_projections_split_by_value(arch):
    """Splitting must preserve values, not just produce plausible names.

    A transposed or mis-ordered split still yields tensors of the right shape and a
    head that trains -- against a projection the serving path will never apply.
    """
    head = head_factory.create(arch, generator=torch.Generator().manual_seed(0))
    exported = dict(weight_naming.rewrite(head_weights.export(head), arch))

    for layer in range(arch.num_layers):
        fused = head.model.layers[layer].self_attn.qkv_proj.weight
        q = exported[f"layers.{layer}.self_attn.q_proj.weight"]
        k = exported[f"layers.{layer}.self_attn.k_proj.weight"]
        v = exported[f"layers.{layer}.self_attn.v_proj.weight"]
        assert q.shape == (arch.q_size, arch.hidden_size)
        assert k.shape == (arch.kv_size, arch.hidden_size)
        assert v.shape == (arch.kv_size, arch.hidden_size)
        # Re-fusing is what `hf_to_vllm_mapper`'s stacked rules do on load.
        torch.testing.assert_close(torch.cat([q, k, v], dim=0), fused)

        fused_mlp = head.model.layers[layer].mlp.gate_up_proj.weight
        gate = exported[f"layers.{layer}.mlp.gate_proj.weight"]
        up = exported[f"layers.{layer}.mlp.up_proj.weight"]
        assert gate.shape == (arch.intermediate_size, arch.hidden_size)
        torch.testing.assert_close(torch.cat([gate, up], dim=0), fused_mlp)


def test_attention_bias_is_split_too():
    """A biased attention head must split `qkv_proj.bias` the same way as the
    weight, or the loaded biases land on the wrong projections.
    """
    arch = make_arch(attention_bias=True)
    head = head_factory.create(arch, generator=torch.Generator().manual_seed(0))
    exported = dict(weight_naming.rewrite(head_weights.export(head), arch))
    fused = head.model.layers[0].self_attn.qkv_proj.bias
    parts = [
        exported[f"layers.0.self_attn.{name}_proj.bias"] for name in ("q", "k", "v")
    ]
    torch.testing.assert_close(torch.cat(parts, dim=0), fused)


def test_unfused_tensors_pass_through_untouched(arch):
    head = head_factory.create(arch)
    exported = dict(weight_naming.rewrite(head_weights.export(head), arch))
    for name in ("fc.weight", "hidden_norm.weight", "norm.weight"):
        assert name in exported
    assert exported["fc.weight"].shape == (arch.hidden_size, arch.feature_width)


def test_config_matches_the_head_it_ships_with(arch):
    config = draft_config_builder.build(
        arch, target_layer_ids=[1, 2, 3], num_target_layers=8, dtype="bfloat16"
    )
    assert config["architectures"] == ["DFlashDraftModel"]
    assert config["model_type"] == "qwen3"
    assert config["num_hidden_layers"] == arch.num_layers
    assert len(config["layer_types"]) == arch.num_layers
    assert config["dflash_config"]["mask_token_id"] == arch.mask_token_id
    assert config["dflash_config"]["target_layer_ids"] == [1, 2, 3]
    assert config["block_size"] == arch.block_size


def test_config_omits_token_ids_that_were_not_given(arch):
    config = draft_config_builder.build(
        arch, target_layer_ids=[1, 2, 3], num_target_layers=8
    )
    assert "bos_token_id" not in config
    assert "eos_token_id" not in config


def test_config_rejects_a_feature_count_mismatch(arch):
    """The fc width is `num_features * hidden`, so a wrong id count builds a head
    whose projection cannot consume the features being captured."""
    with pytest.raises(ValueError, match="fc would be built at the wrong width"):
        draft_config_builder.build(
            arch,
            target_layer_ids=list(range(arch.num_features + 1)),
            num_target_layers=8,
        )


def test_write_checkpoint_round_trips(tmp_path, arch):
    head = head_factory.create(arch, generator=torch.Generator().manual_seed(1))
    config = draft_config_builder.build(
        arch,
        target_layer_ids=list(range(1, arch.num_features + 1)),
        num_target_layers=8,
    )
    written = checkpoint_writer.write(
        tmp_path, head_weights.export(head), arch, config=config, dtype=torch.float32
    )

    assert written.is_file()
    written_config = json.loads(
        (tmp_path / checkpoint_writer.CONFIG_FILENAME).read_text()
    )
    assert written_config["model_type"] == "qwen3"

    from safetensors.torch import load_file

    loaded = load_file(str(written))
    expected = dict(weight_naming.rewrite(head_weights.export(head), arch))
    assert set(loaded) == set(expected)
    for name, tensor in expected.items():
        torch.testing.assert_close(loaded[name], tensor.detach())


def test_write_checkpoint_casts_before_writing(tmp_path, arch):
    """The trainable head is fp32; a checkpoint written at that precision is twice
    the size it needs to be."""
    from safetensors.torch import load_file

    head = head_factory.create(arch)
    written = checkpoint_writer.write(
        tmp_path, head_weights.export(head), arch, dtype=torch.bfloat16
    )
    loaded = load_file(str(written))
    assert all(tensor.dtype == torch.bfloat16 for tensor in loaded.values())


def test_write_checkpoint_can_refresh_weights_only(tmp_path, arch):
    """Omitting the config refreshes a directory that already has one."""
    head = head_factory.create(arch)
    checkpoint_writer.write(tmp_path, head_weights.export(head), arch)
    assert not (tmp_path / checkpoint_writer.CONFIG_FILENAME).exists()
    assert (tmp_path / checkpoint_writer.WEIGHTS_FILENAME).is_file()
