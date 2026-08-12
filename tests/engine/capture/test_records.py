"""The data containers, and the invariants they carry themselves."""

import pytest
import torch

from tests.conftest import HIDDEN, NUM_FEATURES, make_arch
from vllm_online_train.engine.capture.records import (
    BufferStats,
    PendingRollout,
    RolloutRecord,
)

FEATURE_WIDTH = NUM_FEATURES * HIDDEN


def test_record_rejects_misaligned_fields():
    """Every field indexes the same position, so a length disagreement is a bug."""
    with pytest.raises(ValueError, match="disagree on length"):
        RolloutRecord(
            token_ids=torch.zeros(4, dtype=torch.int32),
            features=torch.zeros(3, FEATURE_WIDTH),
            final_hidden=torch.zeros(4, HIDDEN),
            prompt_len=1,
        )


def test_pending_counts_tokens_as_chunks_arrive():
    pending_rollout = PendingRollout(prompt_len=4)
    for _ in range(3):
        pending_rollout.append(
            torch.zeros(5, dtype=torch.int32),
            torch.zeros(5, FEATURE_WIDTH),
            torch.zeros(5, HIDDEN),
        )
    assert pending_rollout.num_tokens == 15
    assert len(pending_rollout.token_ids) == 3


def test_discard_releases_the_chunks_it_accumulated():
    """A dropped rollout holds host memory until the request retires, so the chunks
    go immediately rather than at `finish`."""
    pending_rollout = PendingRollout(prompt_len=4)
    pending_rollout.append(
        torch.zeros(5, dtype=torch.int32),
        torch.zeros(5, FEATURE_WIDTH),
        torch.zeros(5, HIDDEN),
    )
    pending_rollout.discard()
    assert pending_rollout.dropped
    assert pending_rollout.token_ids == []
    assert pending_rollout.features == []
    assert pending_rollout.final_hidden == []


def test_stats_count_known_drop_reasons():
    buffer_stats = BufferStats()
    buffer_stats.count_drop("too_long")
    buffer_stats.count_drop("too_long")
    buffer_stats.count_drop("aborted")
    assert buffer_stats.dropped_too_long == 2
    assert buffer_stats.dropped_aborted == 1


def test_stats_ignore_an_unknown_reason():
    """An unrecognised reason must not create an attribute that never reaches the
    metrics schema."""
    buffer_stats = BufferStats()
    buffer_stats.count_drop("no_such_reason")
    assert "buffer/dropped_no_such_reason" not in buffer_stats.as_metrics()


def test_stats_export_under_a_buffer_prefix():
    metrics = BufferStats(admitted=3).as_metrics()
    assert metrics["buffer/admitted"] == 3.0
    assert all(key.startswith("buffer/") for key in metrics)


def test_arch_derives_the_fused_widths():
    head_arch = make_arch()
    assert head_arch.q_size == head_arch.num_heads * head_arch.head_dim
    assert head_arch.kv_size == head_arch.num_kv_heads * head_arch.head_dim
    assert head_arch.feature_width == (
        head_arch.num_features * head_arch.hidden_size
    )
    assert head_arch.n_rep == head_arch.num_heads // head_arch.num_kv_heads
