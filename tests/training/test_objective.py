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


def test_teacher_hidden_reads_the_position_before_each_label(batch, config):
    """The distribution over the label at `a + j` is the target's output at
    `a + j - 1`, so block `m` reads positions `a_m .. a_m + D - 1`."""
    draft = config.shapes.num_draft_tokens
    gathered = teacher_scorer.hidden_for_labels(batch, draft)
    batch_size, num_anchors = batch.anchor_positions.shape
    assert gathered.shape == (batch_size, num_anchors, draft, HIDDEN)

    context_len = batch.final_hidden.shape[1]
    for row in range(batch_size):
        for block in range(num_anchors):
            anchor = int(batch.anchor_positions[row, block])
            for offset in range(draft):
                position = min(anchor + offset, context_len - 1)
                torch.testing.assert_close(
                    gathered[row, block, offset], batch.final_hidden[row, position]
                )


def test_forward_kl_is_zero_when_the_student_matches_the_teacher():
    logits = torch.randn(6, 12)
    kl = kl_divergence(logits, logits.clone(), "forward")
    torch.testing.assert_close(kl, torch.zeros(6), atol=1e-6, rtol=0)


def test_kl_directions_differ_and_are_both_non_negative():
    """Forward is KL(teacher || student): weighted by the target's own mass, so it
    supervises where the target actually puts probability."""
    student = torch.randn(8, 32)
    teacher = torch.randn(8, 32) * 2.0
    forward = kl_divergence(student, teacher, "forward")
    reverse = kl_divergence(student, teacher, "reverse")
    assert (forward >= -1e-6).all() and (reverse >= -1e-6).all()
    assert not torch.allclose(forward, reverse)


def test_kl_is_invariant_to_a_logit_shift():
    """Softmax is shift-invariant, so an added constant must not move the loss."""
    student = torch.randn(4, 16)
    teacher = torch.randn(4, 16)
    base = kl_divergence(student, teacher, "forward")
    shifted = kl_divergence(student + 3.0, teacher - 2.0, "forward")
    torch.testing.assert_close(base, shifted, atol=1e-5, rtol=1e-5)


def test_position_decay_follows_dflash_equation_4(batch):
    """`w_k = exp(-(k-1)/gamma)`: an early wrong token invalidates everything after
    it in the block, so early positions must weigh more than a uniform mean."""
    gamma = 4.0
    weights = PositionDecay(gamma).weights(batch)
    assert weights is not None
    for offset in range(batch.num_draft_tokens):
        expected = math.exp(-offset / gamma)
        assert weights[0, 0, offset].item() == pytest.approx(expected, rel=1e-5)
    assert weights[0, 0, 0] > weights[0, 0, -1]


def test_zero_gamma_selects_a_uniform_mean(batch):
    assert PositionDecay(0.0).weights(batch) is None


def test_loss_is_independent_of_the_chunk_size(head, teacher, records):
    """Chunking is a memory strategy, not a change to the objective.

    Normalising by the per-chunk weight sum instead of the global one would make the
    loss depend on how the batch happened to be split.
    """
    losses = []
    for chunk_size in (1, 4, 512):
        config = make_config(make_settings(kl_chunk_size=chunk_size))
        batch = make_collator(config).collate(records)
        objective = make_objective(config, head, teacher)
        loss, _ = objective(student_hidden=head.replay(batch), batch=batch)
        losses.append(float(loss.detach()))
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)
    assert losses[0] == pytest.approx(losses[2], rel=1e-5)


def test_padding_blocks_do_not_contribute(head, teacher, config, records):
    """Only positions inside kept blocks are scored."""
    batch = make_collator(config).collate(records)
    accepted, rejected = batch.acceptance_index_masks()
    scored = accepted | rejected
    assert scored.shape == batch.block_token_ids.shape
    assert bool((scored & ~batch.block_keep_mask.unsqueeze(-1)).sum() == 0)

    objective = make_objective(config, head, teacher)
    _, metrics = objective(student_hidden=head.replay(batch), batch=batch)
    assert metrics["count/tokens"] == float(scored.sum())


def test_mismatched_vocabularies_are_rejected(head, batch, config):
    """A reduced draft vocabulary needs the teacher restricted to the same support
    first, or forward KL diverges on tokens the student cannot represent."""
    objective = make_objective(
        config,
        head,
        lambda hidden: torch.zeros(hidden.shape[0], config.shapes.vocab_size + 5),
    )
    with pytest.raises(ValueError, match="student vocabulary is"):
        objective(student_hidden=head.replay(batch), batch=batch)


def test_top1_agreement_is_reported(head, teacher, config, batch):
    """The cheapest proxy for acceptance: a drafted token is accepted only if it
    matches, so top-1 agreement bounds what tau can reach."""
    objective = make_objective(config, head, teacher)
    _, metrics = objective(student_hidden=head.replay(batch), batch=batch)
    assert 0.0 <= metrics["train/top1_agree"] <= 1.0
    assert metrics["count/tokens"] > 0


def test_an_auxiliary_ce_term_is_added_when_weighted(head, teacher, records):
    config = make_config(make_settings(label_smoothing_ce_weight=0.5))
    batch = make_collator(config).collate(records)
    objective = make_objective(config, head, teacher)
    _, metrics = objective(student_hidden=head.replay(batch), batch=batch)
    assert "loss/ce" in metrics
    assert metrics["loss/total"] != metrics["loss/kl"]


def test_no_ce_term_at_zero_weight(head, teacher, config, batch):
    objective = make_objective(config, head, teacher)
    _, metrics = objective(student_hidden=head.replay(batch), batch=batch)
    assert "loss/ce" not in metrics
    assert metrics["loss/total"] == pytest.approx(metrics["loss/kl"])


def test_a_batch_with_no_scored_position_returns_a_zero_loss(head, batch, config):
    """A degenerate batch must still return a scalar carrying the graph, so the
    backward pass does not raise."""
    batch.block_keep_mask = torch.zeros_like(batch.block_keep_mask)
    objective = make_objective(config, head, lambda hidden: hidden)
    loss, metrics = objective(student_hidden=head.replay(batch), batch=batch)
    assert metrics["loss/total"] == 0.0
    loss.backward()
