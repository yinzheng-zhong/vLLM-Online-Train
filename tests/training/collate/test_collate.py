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
from vllm_online_train.training.collate.anchor_sampler import AnchorSampler
from vllm_online_train.training.collate.device_transfer import DeviceTransfer


def test_slot_zero_is_not_a_scored_position(resolved_config, replay_batch):
    """A `K`-slot block yields `K - 1` labels; slot 0 re-feeds the anchor."""
    block_size = resolved_config.engine_shapes.block_size
    assert replay_batch.num_draft_tokens == block_size - 1
    assert replay_batch.block_token_ids.shape[-1] == block_size - 1


def test_first_anchor_predicts_the_first_response_token(resolved_config):
    rollout_record = make_record(num_tokens=40, prompt_len=10)
    num_draft_tokens = resolved_config.engine_shapes.num_draft_tokens
    block_records = block_builder.build(
        rollout_record, num_draft_tokens=num_draft_tokens, stride=1
    )
    # Anchor at prompt_len - 1 means the block's labels start at prompt_len.
    assert block_records[0].anchor == 9
    expected_labels = rollout_record.token_ids[10 : 10 + num_draft_tokens]
    assert block_records[0].draft_token_ids == expected_labels.tolist()


def test_last_anchor_leaves_a_full_block_of_labels(resolved_config):
    rollout_record = make_record(num_tokens=40, prompt_len=10)
    num_draft_tokens = resolved_config.engine_shapes.num_draft_tokens
    block_records = block_builder.build(
        rollout_record, num_draft_tokens=num_draft_tokens, stride=1
    )
    assert (
        block_records[-1].anchor + num_draft_tokens
        == rollout_record.num_tokens - 1
    )
    assert all(
        len(block_record.draft_token_ids) == num_draft_tokens
        for block_record in block_records
    )


def test_sft_blocks_are_fully_accepted(replay_batch):
    """The labels are the target's own continuation, so nothing is rejected."""
    accepted_mask, rejected_mask = replay_batch.acceptance_index_masks()
    assert bool(accepted_mask[replay_batch.block_keep_mask].all())
    assert not bool(rejected_mask.any())


def test_stride_controls_anchor_supply(resolved_config):
    rollout_record = make_record(num_tokens=60, prompt_len=5)
    num_draft_tokens = resolved_config.engine_shapes.num_draft_tokens
    dense_records = block_builder.build(
        rollout_record, num_draft_tokens=num_draft_tokens, stride=1
    )
    sparse_records = block_builder.build(
        rollout_record, num_draft_tokens=num_draft_tokens, stride=4
    )
    assert len(sparse_records) < len(dense_records)
    assert [record.anchor for record in sparse_records] == [
        record.anchor for record in dense_records
    ][::4]


def test_a_rollout_shorter_than_a_block_yields_nothing(resolved_config):
    engine_shapes = resolved_config.engine_shapes
    rollout_record = make_record(
        num_tokens=engine_shapes.block_size, prompt_len=2
    )
    assert (
        block_builder.build(
            rollout_record,
            num_draft_tokens=engine_shapes.num_draft_tokens,
            stride=1,
        )
        == []
    )


def test_anchor_cap_is_subsampled_and_kept_sorted():
    resolved_config = make_config(make_settings(anchors_per_sequence=3))
    replay_batch = make_collator(resolved_config).collate(
        [make_record(60, 5, seed=3)]
    )
    assert replay_batch.num_anchors == 3
    anchor_positions = replay_batch.anchor_positions[0].tolist()
    assert anchor_positions == sorted(anchor_positions)


def test_anchor_sampler_returns_the_input_below_the_cap():
    """Below the cap there is nothing to choose, so the list must pass through
    untouched rather than be re-sorted into a different order."""
    anchor_sampler = AnchorSampler(
        make_settings(anchors_per_sequence=8).anchors, random.Random(0)
    )
    rollout_record = make_record(40, 10)
    block_records = block_builder.build(
        rollout_record, num_draft_tokens=3, stride=8
    )
    assert anchor_sampler.select(block_records) is block_records


def test_deterministic_subsample_takes_the_first_n():
    anchor_sampler = AnchorSampler(
        make_settings(
            anchors_per_sequence=2, random_anchor_subsample=False
        ).anchors,
        random.Random(0),
    )
    rollout_record = make_record(60, 5)
    block_records = block_builder.build(
        rollout_record, num_draft_tokens=3, stride=1
    )
    assert [
        record.anchor for record in anchor_sampler.select(block_records)
    ] == [record.anchor for record in block_records[:2]]


def test_context_reaches_far_enough_for_teacher_positions(
    resolved_config, rollout_records
):
    """`final_hidden` is read as far as `max_anchor + D - 1`.

    Trimming to the attention reach (`max_anchor`) would score the last anchor of
    every batch against zero-padding.
    """
    replay_batch = make_collator(resolved_config).collate(rollout_records)
    max_anchor = int(
        replay_batch.anchor_positions[replay_batch.block_keep_mask].max()
    )
    context_len = replay_batch.input_ids.shape[1]
    assert context_len >= max_anchor + replay_batch.num_draft_tokens


def test_padding_blocks_are_masked_out_of_the_loss():
    """Rows are padded to the widest sample's anchor count."""
    resolved_config = make_config(make_settings(anchors_per_sequence=32))
    replay_batch = make_collator(resolved_config).collate(
        [make_record(60, 5, seed=4), make_record(20, 2, seed=5)]
    )
    assert not bool(replay_batch.block_keep_mask.all())
    accepted_mask, rejected_mask = replay_batch.acceptance_index_masks()
    padding_mask = ~replay_batch.block_keep_mask
    assert not bool(accepted_mask[padding_mask].any())
    assert not bool(rejected_mask[padding_mask].any())


def test_returns_none_when_no_rollout_yields_an_anchor(
    resolved_config, rollout_collator
):
    tiny_record = make_record(
        num_tokens=resolved_config.engine_shapes.block_size, prompt_len=2
    )
    assert rollout_collator.collate([tiny_record]) is None
    assert rollout_collator.collate([]) is None


def test_block_offsets_are_one_based(replay_batch):
    """The decay is `exp(-(k-1)/gamma)`, so `k` starts at 1 and the first position
    is unweighted."""
    block_offsets = replay_batch.block_offsets()
    assert block_offsets[0, 0].tolist() == list(
        range(1, replay_batch.num_draft_tokens + 1)
    )


def test_meta_records_the_batch_shape(replay_batch, rollout_records):
    assert replay_batch.meta["num_rollouts"] == len(rollout_records)
    assert replay_batch.meta["context_len"] == replay_batch.input_ids.shape[1]


def test_attention_mask_marks_only_real_context(rollout_collator):
    """A short rollout padded up to the batch's context length must not be scored
    against its own padding."""
    replay_batch = rollout_collator.collate(
        [make_record(60, 5, seed=6), make_record(24, 2, seed=7)]
    )
    context_lengths = replay_batch.attention_mask.sum(dim=1).tolist()
    assert context_lengths[0] != context_lengths[1]
    assert all(
        length <= replay_batch.input_ids.shape[1] for length in context_lengths
    )


def test_cpu_transfer_is_a_plain_move():
    import torch

    device_transfer = DeviceTransfer(CPU)
    tensor = torch.ones(3, 4)
    moved = device_transfer.send(tensor)
    assert moved.device.type == "cpu"
    assert torch.equal(moved, tensor)
