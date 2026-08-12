"""The SFT objective: full-vocabulary KL with in-block position decay.

The load-bearing property is the label alignment. An off-by-one in the teacher gather
still converges -- to a head one position out of phase, which drafts confidently and
is rejected constantly.
"""

import math

import pytest
import torch

from tests.conftest import (
    HIDDEN,
    kl_divergence,
    make_collator,
    make_config,
    make_objective,
    make_settings,
    teacher_scorer,
)
from vllm_online_train.training.loss.position_decay import PositionDecay


def test_teacher_hidden_reads_the_position_before_each_label(
    replay_batch, resolved_config
):
    """The distribution over the label at `a + j` is the target's output at
    `a + j - 1`, so block `m` reads positions `a_m .. a_m + D - 1`."""
    num_draft_tokens = resolved_config.engine_shapes.num_draft_tokens
    gathered = teacher_scorer.hidden_for_labels(replay_batch, num_draft_tokens)
    batch_size, num_anchors = replay_batch.anchor_positions.shape
    assert gathered.shape == (batch_size, num_anchors, num_draft_tokens, HIDDEN)

    context_len = replay_batch.final_hidden.shape[1]
    for row in range(batch_size):
        for block in range(num_anchors):
            anchor = int(replay_batch.anchor_positions[row, block])
            for offset in range(num_draft_tokens):
                position = min(anchor + offset, context_len - 1)
                torch.testing.assert_close(
                    gathered[row, block, offset],
                    replay_batch.final_hidden[row, position],
                )


def test_forward_kl_is_zero_when_the_student_matches_the_teacher():
    logits = torch.randn(6, 12)
    kl = kl_divergence(logits, logits.clone(), "forward")
    torch.testing.assert_close(kl, torch.zeros(6), atol=1e-6, rtol=0)


def test_kl_directions_differ_and_are_both_non_negative():
    """Forward is KL(teacher || student): weighted by the target's own mass, so it
    supervises where the target actually puts probability."""
    student_logits = torch.randn(8, 32)
    teacher_logits = torch.randn(8, 32) * 2.0
    forward_kl = kl_divergence(student_logits, teacher_logits, "forward")
    reverse_kl = kl_divergence(student_logits, teacher_logits, "reverse")
    assert (forward_kl >= -1e-6).all() and (reverse_kl >= -1e-6).all()
    assert not torch.allclose(forward_kl, reverse_kl)


def test_kl_is_invariant_to_a_logit_shift():
    """Softmax is shift-invariant, so an added constant must not move the loss."""
    student_logits = torch.randn(4, 16)
    teacher_logits = torch.randn(4, 16)
    base_kl = kl_divergence(student_logits, teacher_logits, "forward")
    shifted_kl = kl_divergence(
        student_logits + 3.0, teacher_logits - 2.0, "forward"
    )
    torch.testing.assert_close(base_kl, shifted_kl, atol=1e-5, rtol=1e-5)


def test_position_decay_follows_dflash_equation_4(replay_batch):
    """`w_k = exp(-(k-1)/gamma)`: an early wrong token invalidates everything after
    it in the block, so early positions must weigh more than a uniform mean."""
    gamma = 4.0
    weights = PositionDecay(gamma).weights(replay_batch)
    assert weights is not None
    for offset in range(replay_batch.num_draft_tokens):
        expected_weight = math.exp(-offset / gamma)
        assert weights[0, 0, offset].item() == pytest.approx(
            expected_weight, rel=1e-5
        )
    assert weights[0, 0, 0] > weights[0, 0, -1]


def test_zero_gamma_selects_a_uniform_mean(replay_batch):
    assert PositionDecay(0.0).weights(replay_batch) is None


def test_loss_is_independent_of_the_chunk_size(
    draft_head, teacher_projection, rollout_records
):
    """Chunking is a memory strategy, not a change to the objective.

    Normalising by the per-chunk weight sum instead of the global one would make the
    loss depend on how the batch happened to be split.
    """
    losses = []
    for chunk_size in (1, 4, 512):
        resolved_config = make_config(make_settings(kl_chunk_size=chunk_size))
        replay_batch = make_collator(resolved_config).collate(rollout_records)
        sft_objective = make_objective(
            resolved_config, draft_head, teacher_projection
        )
        loss, _ = sft_objective(
            student_hidden=draft_head.replay(replay_batch),
            replay_batch=replay_batch,
        )
        losses.append(float(loss.detach()))
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)
    assert losses[0] == pytest.approx(losses[2], rel=1e-5)


def test_padding_blocks_do_not_contribute(
    draft_head, teacher_projection, resolved_config, rollout_records
):
    """Only positions inside kept blocks are scored."""
    replay_batch = make_collator(resolved_config).collate(rollout_records)
    accepted_mask, rejected_mask = replay_batch.acceptance_index_masks()
    scored_mask = accepted_mask | rejected_mask
    assert scored_mask.shape == replay_batch.block_token_ids.shape
    assert bool(
        (scored_mask & ~replay_batch.block_keep_mask.unsqueeze(-1)).sum() == 0
    )

    sft_objective = make_objective(resolved_config, draft_head, teacher_projection)
    _, metrics = sft_objective(
        student_hidden=draft_head.replay(replay_batch), replay_batch=replay_batch
    )
    assert metrics["count/tokens"] == float(scored_mask.sum())


def test_mismatched_vocabularies_are_rejected(
    draft_head, replay_batch, resolved_config
):
    """A reduced draft vocabulary needs the teacher restricted to the same support
    first, or forward KL diverges on tokens the student cannot represent."""
    vocab_size = resolved_config.engine_shapes.vocab_size
    sft_objective = make_objective(
        resolved_config,
        draft_head,
        lambda hidden_states: torch.zeros(hidden_states.shape[0], vocab_size + 5),
    )
    with pytest.raises(ValueError, match="student vocabulary is"):
        sft_objective(
            student_hidden=draft_head.replay(replay_batch),
            replay_batch=replay_batch,
        )


def test_top1_agreement_is_reported(
    draft_head, teacher_projection, resolved_config, replay_batch
):
    """The cheapest proxy for acceptance: a drafted token is accepted only if it
    matches, so top-1 agreement bounds what tau can reach."""
    sft_objective = make_objective(resolved_config, draft_head, teacher_projection)
    _, metrics = sft_objective(
        student_hidden=draft_head.replay(replay_batch), replay_batch=replay_batch
    )
    assert 0.0 <= metrics["train/top1_agree"] <= 1.0
    assert metrics["count/tokens"] > 0


def test_an_auxiliary_ce_term_is_added_when_weighted(
    draft_head, teacher_projection, rollout_records
):
    resolved_config = make_config(make_settings(label_smoothing_ce_weight=0.5))
    replay_batch = make_collator(resolved_config).collate(rollout_records)
    sft_objective = make_objective(resolved_config, draft_head, teacher_projection)
    _, metrics = sft_objective(
        student_hidden=draft_head.replay(replay_batch), replay_batch=replay_batch
    )
    assert "loss/ce" in metrics
    assert metrics["loss/total"] != metrics["loss/kl"]


def test_no_ce_term_at_zero_weight(
    draft_head, teacher_projection, resolved_config, replay_batch
):
    sft_objective = make_objective(resolved_config, draft_head, teacher_projection)
    _, metrics = sft_objective(
        student_hidden=draft_head.replay(replay_batch), replay_batch=replay_batch
    )
    assert "loss/ce" not in metrics
    assert metrics["loss/total"] == pytest.approx(metrics["loss/kl"])


def test_a_batch_with_no_scored_position_returns_a_zero_loss(
    draft_head, replay_batch, resolved_config
):
    """A degenerate batch must still return a scalar carrying the graph, so the
    backward pass does not raise."""
    replay_batch.block_keep_mask = torch.zeros_like(replay_batch.block_keep_mask)
    sft_objective = make_objective(
        resolved_config, draft_head, lambda hidden_states: hidden_states
    )
    loss, metrics = sft_objective(
        student_hidden=draft_head.replay(replay_batch), replay_batch=replay_batch
    )
    assert metrics["loss/total"] == 0.0
    loss.backward()
