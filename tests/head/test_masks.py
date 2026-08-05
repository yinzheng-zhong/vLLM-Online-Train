"""The anchored block mask: this head's attention geometry.

An anchor whose context includes its own token still trains, having been shown the
answer. A block that can see another block still trains, having been shown a
neighbour's proposal.
"""

import pytest
import torch

from tests.conftest import mask_builder


def allow_for(batch, config, *, causal: bool = False) -> torch.Tensor:
    return mask_builder.anchored_block_mask(
        anchor_positions=batch.anchor_positions,
        block_keep_mask=batch.block_keep_mask,
        context_len=batch.input_ids.shape[1],
        block_size=config.shapes.block_size,
        intra_block_causal=causal,
        context_valid_mask=batch.attention_mask,
    )


def test_anchor_is_excluded_from_its_own_context(config, batch):
    """The rule is `kv < anchor`, not `<=`: the anchor is re-fed as block slot 0, so
    leaving it in the context KV would present it twice."""
    allow = allow_for(batch, config)
    anchor = int(batch.anchor_positions[0, 0])
    assert not bool(allow[0, 0, 0, anchor])
    assert bool(allow[0, 0, 0, anchor - 1])


def test_block_attention_is_bidirectional(config, batch):
    """The defining DFlash property, and why one pass produces the whole block."""
    allow = allow_for(batch, config)
    context_len = batch.input_ids.shape[1]
    k = config.shapes.block_size
    assert bool(allow[0, 0, 0, context_len + k - 1])
    assert bool(allow[0, 0, k - 1, context_len])


def test_causal_variant_closes_the_upper_triangle(config, batch):
    allow = allow_for(batch, config, causal=True)
    context_len = batch.input_ids.shape[1]
    k = config.shapes.block_size
    assert not bool(allow[0, 0, 0, context_len + k - 1])
    assert bool(allow[0, 0, k - 1, context_len])


def test_blocks_cannot_see_each_other(config, batch):
    """Leakage would let one anchor read another's proposed tokens."""
    if batch.num_anchors < 2:
        pytest.skip("needs at least two anchors")
    allow = allow_for(batch, config)
    context_len = batch.input_ids.shape[1]
    k = config.shapes.block_size
    other = allow[0, 0, :k, context_len + k : context_len + 2 * k]
    assert not bool(other.any())


def test_no_row_is_fully_masked(config, batch):
    """A softmax over an all -inf row yields NaN, which would poison the batch."""
    allow = allow_for(batch, config)
    assert bool(allow.any(-1).all())
    bias = mask_builder.to_additive_mask(allow, torch.float32)
    assert torch.isfinite(bias.max(dim=-1).values).all()


def test_additive_mask_is_zero_where_allowed():
    allow = torch.tensor([[True, False]])
    bias = mask_builder.to_additive_mask(allow, torch.float32)
    assert bias[0, 0] == 0.0
    assert bias[0, 1] == torch.finfo(torch.float32).min


def test_context_padding_is_closed():
    """A row padded up to the batch's context length must not be attended over."""
    anchors = torch.tensor([[6]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    context_valid = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
    allow = mask_builder.anchored_block_mask(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_len=8,
        block_size=4,
        intra_block_causal=False,
        context_valid_mask=context_valid,
    )[0, 0]
    assert allow[0, :3].all()
    assert not allow[0, 3:8].any()


def test_padding_blocks_keep_only_their_own_diagonal():
    """Closing a padding block entirely would leave an all -inf row."""
    anchors = torch.tensor([[5, 0]])
    keep = torch.tensor([[True, False]])
    allow = mask_builder.anchored_block_mask(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_len=8,
        block_size=4,
        intra_block_causal=False,
    )[0, 0]
    padded = allow[4:8]
    assert padded.any(-1).all()
    assert int(padded.sum()) == 4


def test_position_ids_are_reanchored(config, batch):
    """Block slots sit at absolute positions `a .. a+K-1` while packed adjacently,
    so position cannot be inferred from index."""
    k = config.shapes.block_size
    positions = mask_builder.block_position_ids(batch.anchor_positions, k)
    anchor = int(batch.anchor_positions[0, 0])
    assert positions[0, :k].tolist() == list(range(anchor, anchor + k))


def test_mask_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        mask_builder.anchored_block_mask(
            anchor_positions=torch.zeros(2, 3, dtype=torch.long),
            block_keep_mask=torch.ones(2, 4, dtype=torch.bool),
            context_len=8,
            block_size=4,
            intra_block_causal=False,
        )
    with pytest.raises(ValueError, match=r"\[B, A\]"):
        mask_builder.anchored_block_mask(
            anchor_positions=torch.zeros(3, dtype=torch.long),
            block_keep_mask=torch.ones(3, dtype=torch.bool),
            context_len=8,
            block_size=4,
            intra_block_causal=False,
        )
