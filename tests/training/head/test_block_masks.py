"""The anchored block mask: this head's attention geometry.

An anchor whose context includes its own token still trains, having been shown the
answer. A block that can see another block still trains, having been shown a
neighbour's proposal.
"""

import pytest
import torch

from tests.conftest import block_mask_builder


def allow_for(
    replay_batch, resolved_config, *, causal: bool = False
) -> torch.Tensor:
    return block_mask_builder.anchored_block_mask(
        anchor_positions=replay_batch.anchor_positions,
        block_keep_mask=replay_batch.block_keep_mask,
        context_len=replay_batch.input_ids.shape[1],
        block_size=resolved_config.engine_shapes.block_size,
        intra_block_causal=causal,
        context_valid_mask=replay_batch.attention_mask,
    )


def test_anchor_is_excluded_from_its_own_context(resolved_config, replay_batch):
    """The rule is `kv < anchor`, not `<=`: the anchor is re-fed as block slot 0, so
    leaving it in the context KV would present it twice."""
    allow_mask = allow_for(replay_batch, resolved_config)
    anchor = int(replay_batch.anchor_positions[0, 0])
    assert not bool(allow_mask[0, 0, 0, anchor])
    assert bool(allow_mask[0, 0, 0, anchor - 1])


def test_block_attention_is_bidirectional(resolved_config, replay_batch):
    """The defining DFlash property, and why one pass produces the whole block."""
    allow_mask = allow_for(replay_batch, resolved_config)
    context_len = replay_batch.input_ids.shape[1]
    block_size = resolved_config.engine_shapes.block_size
    assert bool(allow_mask[0, 0, 0, context_len + block_size - 1])
    assert bool(allow_mask[0, 0, block_size - 1, context_len])


def test_causal_variant_closes_the_upper_triangle(resolved_config, replay_batch):
    allow_mask = allow_for(replay_batch, resolved_config, causal=True)
    context_len = replay_batch.input_ids.shape[1]
    block_size = resolved_config.engine_shapes.block_size
    assert not bool(allow_mask[0, 0, 0, context_len + block_size - 1])
    assert bool(allow_mask[0, 0, block_size - 1, context_len])


def test_blocks_cannot_see_each_other(resolved_config, replay_batch):
    """Leakage would let one anchor read another's proposed tokens."""
    if replay_batch.num_anchors < 2:
        pytest.skip("needs at least two anchors")
    allow_mask = allow_for(replay_batch, resolved_config)
    context_len = replay_batch.input_ids.shape[1]
    block_size = resolved_config.engine_shapes.block_size
    other_block = allow_mask[
        0, 0, :block_size, context_len + block_size : context_len + 2 * block_size
    ]
    assert not bool(other_block.any())


def test_no_row_is_fully_masked(resolved_config, replay_batch):
    """A softmax over an all -inf row yields NaN, which would poison the batch."""
    allow_mask = allow_for(replay_batch, resolved_config)
    assert bool(allow_mask.any(-1).all())
    attn_bias = block_mask_builder.to_additive_mask(allow_mask, torch.float32)
    assert torch.isfinite(attn_bias.max(dim=-1).values).all()


def test_additive_mask_is_zero_where_allowed():
    allow_mask = torch.tensor([[True, False]])
    attn_bias = block_mask_builder.to_additive_mask(allow_mask, torch.float32)
    assert attn_bias[0, 0] == 0.0
    assert attn_bias[0, 1] == torch.finfo(torch.float32).min


def test_context_padding_is_closed():
    """A row padded up to the batch's context length must not be attended over."""
    anchor_positions = torch.tensor([[6]])
    block_keep_mask = torch.ones_like(anchor_positions, dtype=torch.bool)
    context_valid_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
    allow_mask = block_mask_builder.anchored_block_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        context_len=8,
        block_size=4,
        intra_block_causal=False,
        context_valid_mask=context_valid_mask,
    )[0, 0]
    assert allow_mask[0, :3].all()
    assert not allow_mask[0, 3:8].any()


def test_padding_blocks_keep_only_their_own_diagonal():
    """Closing a padding block entirely would leave an all -inf row."""
    anchor_positions = torch.tensor([[5, 0]])
    block_keep_mask = torch.tensor([[True, False]])
    allow_mask = block_mask_builder.anchored_block_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        context_len=8,
        block_size=4,
        intra_block_causal=False,
    )[0, 0]
    padding_rows = allow_mask[4:8]
    assert padding_rows.any(-1).all()
    assert int(padding_rows.sum()) == 4


def test_position_ids_are_reanchored(resolved_config, replay_batch):
    """Block slots sit at absolute positions `a .. a+K-1` while packed adjacently,
    so position cannot be inferred from index."""
    block_size = resolved_config.engine_shapes.block_size
    slot_position_ids = block_mask_builder.block_position_ids(
        replay_batch.anchor_positions, block_size
    )
    anchor = int(replay_batch.anchor_positions[0, 0])
    assert slot_position_ids[0, :block_size].tolist() == list(
        range(anchor, anchor + block_size)
    )


def test_mask_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        block_mask_builder.anchored_block_mask(
            anchor_positions=torch.zeros(2, 3, dtype=torch.long),
            block_keep_mask=torch.ones(2, 4, dtype=torch.bool),
            context_len=8,
            block_size=4,
            intra_block_causal=False,
        )
    with pytest.raises(ValueError, match=r"\[B, A\]"):
        block_mask_builder.anchored_block_mask(
            anchor_positions=torch.zeros(3, dtype=torch.long),
            block_keep_mask=torch.ones(3, dtype=torch.bool),
            context_len=8,
            block_size=4,
            intra_block_causal=False,
        )
