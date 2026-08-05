"""Anchor placement and block layout.

These are the conventions that fail silently. A block scored on `K` positions instead
of `K - 1` still trains -- the head just becomes a parallel next-token predictor that
appears to draft one extra token.
"""

import random

from tests.conftest import (
    CPU,
    block_builder,
    make_collator,
    make_config,
    make_record,
    make_settings,
)
from vllm_online_train.collate.anchors import AnchorSampler
from vllm_online_train.collate.transfer import DeviceTransfer


def test_slot_zero_is_not_a_scored_position(config, batch):
    """A `K`-slot block yields `K - 1` labels; slot 0 re-feeds the anchor."""
    assert batch.num_draft_tokens == config.shapes.block_size - 1
    assert batch.block_token_ids.shape[-1] == config.shapes.block_size - 1


def test_first_anchor_predicts_the_first_response_token(config):
    record = make_record(num_tokens=40, prompt_len=10)
    draft = config.shapes.num_draft_tokens
    blocks = block_builder.build(record, num_draft_tokens=draft, stride=1)
    # Anchor at prompt_len - 1 means the block's labels start at prompt_len.
    assert blocks[0].anchor == 9
    expected = record.token_ids[10 : 10 + draft]
    assert blocks[0].draft_token_ids == expected.tolist()


def test_last_anchor_leaves_a_full_block_of_labels(config):
    record = make_record(num_tokens=40, prompt_len=10)
    draft = config.shapes.num_draft_tokens
    blocks = block_builder.build(record, num_draft_tokens=draft, stride=1)
    assert blocks[-1].anchor + draft == record.num_tokens - 1
    assert all(len(block.draft_token_ids) == draft for block in blocks)


def test_sft_blocks_are_fully_accepted(batch):
    """The labels are the target's own continuation, so nothing is rejected."""
    accepted, rejected = batch.acceptance_index_masks()
    assert bool(accepted[batch.block_keep_mask].all())
    assert not bool(rejected.any())


def test_stride_controls_anchor_supply(config):
    record = make_record(num_tokens=60, prompt_len=5)
    draft = config.shapes.num_draft_tokens
    dense = block_builder.build(record, num_draft_tokens=draft, stride=1)
    sparse = block_builder.build(record, num_draft_tokens=draft, stride=4)
    assert len(sparse) < len(dense)
    assert [b.anchor for b in sparse] == [b.anchor for b in dense][::4]


def test_a_rollout_shorter_than_a_block_yields_nothing(config):
    record = make_record(num_tokens=config.shapes.block_size, prompt_len=2)
    assert (
        block_builder.build(
            record, num_draft_tokens=config.shapes.num_draft_tokens, stride=1
        )
        == []
    )


def test_anchor_cap_is_subsampled_and_kept_sorted():
    config = make_config(make_settings(anchors_per_sequence=3))
    batch = make_collator(config).collate([make_record(60, 5, seed=3)])
    assert batch.num_anchors == 3
    anchors = batch.anchor_positions[0].tolist()
    assert anchors == sorted(anchors)


def test_anchor_sampler_returns_the_input_below_the_cap():
    """Below the cap there is nothing to choose, so the list must pass through
    untouched rather than be re-sorted into a different order."""
    sampler = AnchorSampler(
        make_settings(anchors_per_sequence=8).anchors, random.Random(0)
    )
    record = make_record(40, 10)
    blocks = block_builder.build(record, num_draft_tokens=3, stride=8)
    assert sampler.select(blocks) is blocks


def test_deterministic_subsample_takes_the_first_n():
    sampler = AnchorSampler(
        make_settings(
            anchors_per_sequence=2, random_anchor_subsample=False
        ).anchors,
        random.Random(0),
    )
    record = make_record(60, 5)
    blocks = block_builder.build(record, num_draft_tokens=3, stride=1)
    assert [b.anchor for b in sampler.select(blocks)] == [
        b.anchor for b in blocks[:2]
    ]


def test_context_reaches_far_enough_for_teacher_positions(config, records):
    """`final_hidden` is read as far as `max_anchor + D - 1`.

    Trimming to the attention reach (`max_anchor`) would score the last anchor of
    every batch against zero-padding.
    """
    batch = make_collator(config).collate(records)
    max_anchor = int(batch.anchor_positions[batch.block_keep_mask].max())
    context_len = batch.input_ids.shape[1]
    assert context_len >= max_anchor + batch.num_draft_tokens


def test_padding_blocks_are_masked_out_of_the_loss():
    """Rows are padded to the widest sample's anchor count."""
    config = make_config(make_settings(anchors_per_sequence=32))
    batch = make_collator(config).collate(
        [make_record(60, 5, seed=4), make_record(20, 2, seed=5)]
    )
    assert not bool(batch.block_keep_mask.all())
    accepted, rejected = batch.acceptance_index_masks()
    padded = ~batch.block_keep_mask
    assert not bool(accepted[padded].any())
    assert not bool(rejected[padded].any())


def test_returns_none_when_no_rollout_yields_an_anchor(config, collator):
    tiny = make_record(num_tokens=config.shapes.block_size, prompt_len=2)
    assert collator.collate([tiny]) is None
    assert collator.collate([]) is None


def test_block_offsets_are_one_based(batch):
    """The decay is `exp(-(k-1)/gamma)`, so `k` starts at 1 and the first position
    is unweighted."""
    offsets = batch.block_offsets()
    assert offsets[0, 0].tolist() == list(range(1, batch.num_draft_tokens + 1))


def test_meta_records_the_batch_shape(batch, records):
    assert batch.meta["num_rollouts"] == len(records)
    assert batch.meta["context_len"] == batch.input_ids.shape[1]


def test_attention_mask_marks_only_real_context(config, collator):
    """A short rollout padded up to the batch's context length must not be scored
    against its own padding."""
    batch = collator.collate([make_record(60, 5, seed=6), make_record(24, 2, seed=7)])
    lengths = batch.attention_mask.sum(dim=1).tolist()
    assert lengths[0] != lengths[1]
    assert all(length <= batch.input_ids.shape[1] for length in lengths)


def test_cpu_transfer_is_a_plain_move():
    import torch

    transfer = DeviceTransfer(CPU)
    tensor = torch.ones(3, 4)
    moved = transfer.send(tensor)
    assert moved.device.type == "cpu"
    assert torch.equal(moved, tensor)
