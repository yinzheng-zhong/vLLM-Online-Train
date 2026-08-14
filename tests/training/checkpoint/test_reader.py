"""Adopting a checkpoint back into a trainable head.

The write path is covered by `test_checkpoint.py`; what matters here is that the read
path is its exact inverse. A load that silently covers only part of the head, or that
fuses the projections in the wrong order, produces a head that trains happily against
weights the serving path will never apply.
"""

import pytest
import torch

from tests.conftest import (
    checkpoint_reader,
    checkpoint_writer,
    head_factory,
    head_weights,
    make_arch,
    weight_name_rewriter,
)


def export_on_disk(draft_head, head_arch):
    """The on-disk form of a head.

    Args:
        draft_head: The head to export.
        head_arch: Supplies the split points.

    Returns:
        A name-to-tensor mapping in split naming.
    """
    return dict(
        weight_name_rewriter.rewrite(head_weights.export(draft_head), head_arch)
    )


@pytest.mark.parametrize("attention_bias", [False, True])
def test_fuse_inverts_rewrite(attention_bias):
    """Round-tripping must return the head's own fused tensors by value, not merely
    names of the right shape. A biased head splits `qkv_proj.bias` as well as its
    weight, and leaves `o_proj.bias` alone."""
    head_arch = make_arch(attention_bias=attention_bias)
    draft_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(3)
    )
    fused = dict(weight_name_rewriter.fuse(export_on_disk(draft_head, head_arch)))
    exported = dict(head_weights.export(draft_head))

    trained_names = {
        name
        for name in exported
        if not any(
            shared in name for shared in weight_name_rewriter.SHARED_WITH_TARGET
        )
    }
    assert set(fused) == trained_names
    for name in trained_names:
        torch.testing.assert_close(fused[name], exported[name])


def test_fuse_concatenates_the_bias_in_query_key_value_order():
    """A bias fused in the wrong order still loads and still trains, against an
    attention the serving path will never apply."""
    head_arch = make_arch(attention_bias=True)
    draft_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(4)
    )
    on_disk = export_on_disk(draft_head, head_arch)
    fused = dict(weight_name_rewriter.fuse(on_disk))

    for layer_index in range(head_arch.num_layers):
        prefix = f"layers.{layer_index}.self_attn"
        torch.testing.assert_close(
            fused[f"{prefix}.qkv_proj.bias"],
            torch.cat(
                [
                    on_disk[f"{prefix}.q_proj.bias"],
                    on_disk[f"{prefix}.k_proj.bias"],
                    on_disk[f"{prefix}.v_proj.bias"],
                ]
            ),
        )
        torch.testing.assert_close(
            fused[f"{prefix}.o_proj.bias"], on_disk[f"{prefix}.o_proj.bias"]
        )


def test_fuse_refuses_a_half_written_projection(head_arch):
    """Fusing `q_proj` and `v_proj` alone would yield a narrower `qkv_proj` that still
    loads, so the missing part has to be named."""
    on_disk = export_on_disk(head_factory.create(head_arch), head_arch)
    del on_disk["layers.0.self_attn.k_proj.weight"]

    with pytest.raises(ValueError, match="k_proj"):
        dict(weight_name_rewriter.fuse(on_disk))


def test_load_trained_restores_every_trained_tensor(head_arch):
    """A head loaded from another head's checkpoint must equal it everywhere it
    trains."""
    trained_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(5)
    )
    fresh_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(6)
    )
    num_loaded = head_weights.load_trained(
        fresh_head,
        weight_name_rewriter.fuse(export_on_disk(trained_head, head_arch)),
    )

    assert num_loaded == len(
        [
            name
            for name, parameter in fresh_head.named_parameters()
            if parameter.requires_grad
        ]
    )
    for name, parameter in fresh_head.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter, dict(trained_head.named_parameters())[name]
            )


def test_load_trained_keeps_the_heads_own_dtype_and_borrowed_tensors(head_arch):
    """The checkpoint is held at the serving dtype while the head trains at fp32, and
    the borrowed tensors come from the target rather than from the checkpoint."""
    trained_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(7)
    )
    fresh_head = head_factory.create(head_arch)
    head_weights.cast_shared(fresh_head, torch.bfloat16)
    borrowed_before = fresh_head.lm_head.weight.clone()

    on_disk = {
        name: tensor.to(torch.bfloat16)
        for name, tensor in export_on_disk(trained_head, head_arch).items()
    }
    head_weights.load_trained(fresh_head, weight_name_rewriter.fuse(on_disk))

    assert fresh_head.model.fc.weight.dtype == torch.float32
    torch.testing.assert_close(
        fresh_head.model.fc.weight,
        trained_head.model.fc.weight,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(fresh_head.lm_head.weight, borrowed_before)
    assert not fresh_head.lm_head.weight.requires_grad
    assert not fresh_head.model.embed_tokens.weight.requires_grad


def test_load_trained_refuses_a_shape_mismatch(head_arch):
    """A checkpoint trained for another width must fail at startup rather than after a
    night of training."""
    wide_head = head_factory.create(make_arch(hidden_size=64))
    on_disk = export_on_disk(wide_head, make_arch(hidden_size=64))

    with pytest.raises(ValueError, match="trained for a different shape"):
        head_weights.load_trained(
            head_factory.create(head_arch), weight_name_rewriter.fuse(on_disk)
        )


def test_load_trained_refuses_a_partial_checkpoint(head_arch):
    """Silently leaving `norm` small-initialised is the failure this whole path
    exists to prevent."""
    on_disk = export_on_disk(head_factory.create(head_arch), head_arch)
    del on_disk["norm.weight"]

    with pytest.raises(ValueError, match=r"model\.norm\.weight"):
        head_weights.load_trained(
            head_factory.create(head_arch), weight_name_rewriter.fuse(on_disk)
        )


def test_load_trained_refuses_a_name_this_head_has_no_tensor_for(head_arch):
    on_disk = export_on_disk(head_factory.create(head_arch), head_arch)
    on_disk["layers.0.self_attn.rotary_emb.inv_freq"] = torch.zeros(4)

    with pytest.raises(ValueError, match="not a tensor of this head"):
        head_weights.load_trained(
            head_factory.create(head_arch), weight_name_rewriter.fuse(on_disk)
        )


def test_a_written_checkpoint_loads_back_into_a_head(tmp_path, head_arch):
    """The end-to-end path a restart takes: export a head, read the directory back,
    and land on the same weights."""
    trained_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(8)
    )
    checkpoint_writer.write(
        tmp_path,
        head_weights.export(trained_head),
        head_arch,
        dtype=torch.float32,
    )

    fresh_head = head_factory.create(head_arch)
    head_weights.load_trained(
        fresh_head, weight_name_rewriter.fuse(checkpoint_reader.read(tmp_path))
    )

    torch.testing.assert_close(
        fresh_head.model.layers[0].self_attn.qkv_proj.weight,
        trained_head.model.layers[0].self_attn.qkv_proj.weight,
    )
    torch.testing.assert_close(
        fresh_head.model.layers[0].mlp.gate_up_proj.weight,
        trained_head.model.layers[0].mlp.gate_up_proj.weight,
    )


def test_reading_a_directory_with_no_weights_is_empty(tmp_path):
    """An empty read is what lets a served draft in another format keep serving rather
    than take the engine down."""
    assert checkpoint_reader.read(tmp_path) == []
    assert checkpoint_reader.read(tmp_path / "absent") == []
