"""The data containers, and the invariants they carry themselves."""

import pytest
import torch

from tests.conftest import HIDDEN, NUM_FEATURES, make_arch
from vllm_online_train.capture.records import BufferStats, PendingRollout, RolloutRecord

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
    pending = PendingRollout(prompt_len=4)
    for _ in range(3):
        pending.append(
            torch.zeros(5, dtype=torch.int32),
            torch.zeros(5, FEATURE_WIDTH),
            torch.zeros(5, HIDDEN),
        )
    assert pending.num_tokens == 15
    assert len(pending.token_ids) == 3


def test_discard_releases_the_chunks_it_accumulated():
    """A dropped rollout holds host memory until the request retires, so the chunks
    go immediately rather than at `finish`."""
    pending = PendingRollout(prompt_len=4)
    pending.append(
        torch.zeros(5, dtype=torch.int32),
        torch.zeros(5, FEATURE_WIDTH),
        torch.zeros(5, HIDDEN),
    )
    pending.discard()
    assert pending.dropped
    assert pending.token_ids == []
    assert pending.features == []
    assert pending.final_hidden == []


def test_stats_count_known_drop_reasons():
    stats = BufferStats()
    stats.count_drop("too_long")
    stats.count_drop("too_long")
    stats.count_drop("aborted")
    assert stats.dropped_too_long == 2
    assert stats.dropped_aborted == 1


def test_stats_ignore_an_unknown_reason():
    """An unrecognised reason must not create an attribute that never reaches the
    metrics schema."""
    stats = BufferStats()
    stats.count_drop("no_such_reason")
    assert "buffer/dropped_no_such_reason" not in stats.as_metrics()


def test_stats_export_under_a_buffer_prefix():
    metrics = BufferStats(admitted=3).as_metrics()
    assert metrics["buffer/admitted"] == 3.0
    assert all(key.startswith("buffer/") for key in metrics)


def test_arch_derives_the_fused_widths():
    arch = make_arch()
    assert arch.q_size == arch.num_heads * arch.head_dim
    assert arch.kv_size == arch.num_kv_heads * arch.head_dim
    assert arch.feature_width == arch.num_features * arch.hidden_size
    assert arch.n_rep == arch.num_heads // arch.num_kv_heads
