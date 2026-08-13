"""Rollout accumulation and pooling.

The buffer's job is to only ever admit rollouts that are contiguous from position 0
and long enough to yield an anchor. Admitting a rollout with a hole in it is the
failure that matters: nothing downstream can detect it.
"""

import torch

from tests.conftest import (
    HIDDEN,
    NUM_FEATURES,
    block_builder,
    make_buffer,
    make_config,
    make_settings,
)
from vllm_online_train.engine.capture.rollouts.rollout_buffer import RolloutBuffer
from vllm_online_train.engine.capture.rollouts.rollout_records import BufferStats

FEATURE_WIDTH = NUM_FEATURES * HIDDEN


def make_chunk(num_tokens: int, start: int = 0) -> dict:
    return dict(
        token_ids=torch.arange(start, start + num_tokens, dtype=torch.int32),
        features=torch.ones(num_tokens, FEATURE_WIDTH),
        final_hidden=torch.ones(num_tokens, HIDDEN),
    )


def test_accumulates_chunks_across_steps():
    """A prompt may be chunked and a response arrives a round at a time."""
    rollout_buffer = make_buffer(make_config())
    rollout_buffer.begin("a", prompt_len=10)
    rollout_buffer.add_chunk("a", **make_chunk(20, start=0))
    rollout_buffer.add_chunk("a", **make_chunk(20, start=20))
    rollout_record = rollout_buffer.seal("a")

    assert rollout_record is not None
    assert rollout_record.num_tokens == 40
    assert rollout_record.features.shape == (40, FEATURE_WIDTH)
    # Chunks must land in arrival order, not be interleaved or reordered.
    assert torch.equal(rollout_record.token_ids, torch.arange(40, dtype=torch.int32))
    assert rollout_buffer.num_rollouts == 1
    assert rollout_buffer.num_tokens == 40


def test_add_chunk_rejects_misaligned_tensors():
    rollout_buffer = make_buffer(make_config())
    rollout_buffer.begin("a", prompt_len=2)
    try:
        rollout_buffer.add_chunk(
            "a",
            token_ids=torch.zeros(4, dtype=torch.int32),
            features=torch.zeros(3, FEATURE_WIDTH),
            final_hidden=torch.zeros(4, HIDDEN),
        )
    except ValueError as exc:
        assert "disagrees on length" in str(exc)
    else:
        raise AssertionError("a misaligned chunk was accepted")


def test_drops_rollout_that_would_exceed_max_length():
    """Dropped, not truncated: cutting from the left shifts the context
    distribution, and cutting from the right discards the freshest positions."""
    rollout_buffer = make_buffer(make_config(make_settings(max_rollout_tokens=32)))
    rollout_buffer.begin("a", prompt_len=4)
    rollout_buffer.add_chunk("a", **make_chunk(30))
    rollout_buffer.add_chunk("a", **make_chunk(10))
    assert rollout_buffer.seal("a") is None
    assert rollout_buffer.buffer_stats.dropped_too_long == 1


def test_ignores_chunks_after_a_drop():
    """A dropped request must not restart mid-sequence, which would produce a record
    whose position 0 is not the start of the sequence."""
    rollout_buffer = make_buffer(make_config())
    rollout_buffer.begin("a", prompt_len=4)
    rollout_buffer.add_chunk("a", **make_chunk(10))
    rollout_buffer.drop("a", reason="aborted")
    rollout_buffer.add_chunk("a", **make_chunk(30))
    assert rollout_buffer.seal("a") is None
    assert rollout_buffer.buffer_stats.dropped_aborted == 1


def test_dropping_an_unknown_request_still_blocks_later_chunks():
    """Capture may drop a request before the buffer has seen it begin."""
    rollout_buffer = make_buffer(make_config())
    rollout_buffer.drop("a", reason="prefix_cache_hit")
    rollout_buffer.begin("a", prompt_len=4)
    rollout_buffer.add_chunk("a", **make_chunk(30))
    assert rollout_buffer.seal("a") is None


def test_drops_rollouts_too_short_to_yield_an_anchor():
    """The first anchor sits at `prompt_len - 1` and needs `num_draft_tokens`
    labelled positions after it.

    Pinned at the exact boundary in both directions, because the admission guard has
    to agree with `BlockBuilder.build`: one token stricter and it discards rollouts
    that do yield anchors, one looser and it admits rollouts that collate to nothing.
    """
    resolved_config = make_config(make_settings(min_rollout_tokens=4))
    num_draft_tokens = resolved_config.engine_shapes.num_draft_tokens
    prompt_len = 20
    first_anchor = prompt_len - 1

    rollout_buffer = make_buffer(resolved_config)
    rollout_buffer.begin("a", prompt_len=prompt_len)
    rollout_buffer.add_chunk("a", **make_chunk(first_anchor + num_draft_tokens))
    assert rollout_buffer.seal("a") is None
    assert rollout_buffer.buffer_stats.dropped_too_short == 1

    # One more token and the anchor at `prompt_len - 1` has a full block of labels.
    rollout_buffer.begin("b", prompt_len=prompt_len)
    rollout_buffer.add_chunk("b", **make_chunk(first_anchor + num_draft_tokens + 1))
    rollout_record = rollout_buffer.seal("b")
    assert rollout_record is not None
    assert block_builder.build(
        rollout_record, num_draft_tokens=num_draft_tokens, stride=1
    )


def test_evicts_oldest_once_capacity_is_exceeded():
    """Capacity is counted in tokens: a pool sized in rollouts would vary by orders
    of magnitude in bytes."""
    resolved_config = make_config(
        make_settings(buffer_capacity_tokens=100, max_rollout_tokens=50)
    )
    rollout_buffer = make_buffer(resolved_config)
    for index in range(4):
        req_id = f"r{index}"
        rollout_buffer.begin(req_id, prompt_len=2)
        rollout_buffer.add_chunk(req_id, **make_chunk(40))
        assert rollout_buffer.seal(req_id) is not None

    assert rollout_buffer.num_tokens <= 100
    assert rollout_buffer.buffer_stats.admitted == 4
    assert rollout_buffer.buffer_stats.evicted == 2
    assert rollout_buffer.num_rollouts == 2


def test_sampling_does_not_consume():
    """At low load, consuming rollouts would starve training exactly when it has the
    most idle capacity to spend."""
    rollout_buffer = make_buffer(make_config())
    for index in range(4):
        req_id = f"r{index}"
        rollout_buffer.begin(req_id, prompt_len=2)
        rollout_buffer.add_chunk(req_id, **make_chunk(40))
        rollout_buffer.seal(req_id)

    assert rollout_buffer.can_sample(2)
    assert len(rollout_buffer.sample(2)) == 2
    assert rollout_buffer.num_rollouts == 4
    assert len(rollout_buffer.sample(2)) == 2


def test_sampling_draws_from_a_snapshot_of_the_pool():
    """The trainer thread draws while the engine thread admits and evicts. Handing the
    live list out lets an eviction land between the sampler's length read and its
    indexing, which raises where it should have drawn."""
    online_train_settings = make_settings()
    resolved_config = make_config(online_train_settings)

    class Evicting:
        """Stands in for an eviction landing inside the draw."""

        def draw(self, pooled_rollouts, count):
            pooled_rollouts.clear()
            return []

    rollout_buffer = RolloutBuffer(
        online_train_settings.buffer,
        resolved_config.engine_shapes,
        Evicting(),
        BufferStats(),
    )
    for index in range(3):
        req_id = f"r{index}"
        rollout_buffer.begin(req_id, prompt_len=2)
        rollout_buffer.add_chunk(req_id, **make_chunk(40))
        rollout_buffer.seal(req_id)

    rollout_buffer.sample(2)
    assert rollout_buffer.num_rollouts == 3
    assert rollout_buffer.num_tokens == 120


def test_cannot_sample_below_batch_size():
    rollout_buffer = make_buffer(make_config())
    rollout_buffer.begin("a", prompt_len=2)
    rollout_buffer.add_chunk("a", **make_chunk(40))
    rollout_buffer.seal("a")
    assert not rollout_buffer.can_sample(3)
    # `sample` still degrades gracefully rather than raising.
    assert len(rollout_buffer.sample(3)) == 1


def test_begin_is_idempotent():
    """Chunked prefill calls it every step without tracking whether it is first."""
    rollout_buffer = make_buffer(make_config())
    rollout_buffer.begin("a", prompt_len=10)
    rollout_buffer.add_chunk("a", **make_chunk(20))
    rollout_buffer.begin("a", prompt_len=10)
    rollout_buffer.add_chunk("a", **make_chunk(20, start=20))
    rollout_record = rollout_buffer.seal("a")
    assert rollout_record is not None and rollout_record.num_tokens == 40


def test_clear_empties_both_stages():
    rollout_buffer = make_buffer(make_config())
    rollout_buffer.begin("a", prompt_len=2)
    rollout_buffer.add_chunk("a", **make_chunk(40))
    rollout_buffer.seal("a")
    rollout_buffer.begin("b", prompt_len=2)
    rollout_buffer.clear()
    assert rollout_buffer.num_rollouts == 0
    assert rollout_buffer.num_tokens == 0
    assert rollout_buffer.num_pending == 0


def test_metrics_report_fill_and_drop_reasons():
    rollout_buffer = make_buffer(make_config())
    rollout_buffer.begin("a", prompt_len=2)
    rollout_buffer.add_chunk("a", **make_chunk(40))
    rollout_buffer.seal("a")
    rollout_buffer.drop("b", reason="prefix_cache_hit")

    metrics = rollout_buffer.as_metrics()
    assert metrics["buffer/rollouts"] == 1.0
    assert metrics["buffer/tokens"] == 40.0
    assert metrics["buffer/dropped_prefix_cache_hit"] == 1.0
    assert 0.0 < metrics["buffer/fill"] < 1.0
