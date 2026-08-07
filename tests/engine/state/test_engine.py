"""The provider that owns the borrowed target state.

Nothing else holds the target handle, so the head can be tested against a stub and
the objective can be scored on CPU without an engine.
"""

import pytest
import torch
from torch import nn

from tests.conftest import (
    CPU,
    HIDDEN,
    REMOTE,
    VOCAB,
    FakeMultiModalTarget,
    FakeTarget,
    placed,
)
from vllm_online_train.contracts.provider import StateProvider
from vllm_online_train.engine.state.engine import EngineStateProvider
from vllm_online_train.engine.state.target_locator import TargetLocator


def provider(target: nn.Module | None = None) -> EngineStateProvider:
    return EngineStateProvider(
        TargetLocator(), target or FakeTarget(), CPU, torch.bfloat16
    )


def test_it_satisfies_the_contract():
    assert isinstance(provider(), StateProvider)


def test_it_reports_both_dtypes():
    target = FakeTarget()
    subject = provider(target)
    assert subject.serving_dtype == torch.bfloat16
    assert subject.target_dtype == target.embed.weight.dtype
    assert subject.device == CPU


def test_borrowed_weights_come_back_detached_in_fp32():
    """The head is trained at fp32 and copies these in; a live view would let a
    training step write into the engine's own weights."""
    target = FakeTarget()
    subject = provider(target)

    embed = subject.embedding_weight()
    assert embed.dtype == torch.float32
    assert not embed.requires_grad
    assert embed.shape == (VOCAB, HIDDEN)
    assert embed.data_ptr() != target.embed.weight.data_ptr()

    lm_head = subject.lm_head_weight()
    assert lm_head.dtype == torch.float32
    assert not lm_head.requires_grad
    assert lm_head.shape == (VOCAB, HIDDEN)
    assert lm_head.data_ptr() != target.lm_head.weight.data_ptr()


def test_borrowed_weights_can_be_asked_for_on_another_device():
    """What lets the head train off the serving GPU: the fp32 copy is allocated where
    the head is, not where the engine is."""
    subject = provider(FakeTarget().to(torch.bfloat16))

    embed = subject.embedding_weight(REMOTE)
    assert embed.device == placed(REMOTE)
    assert embed.dtype == torch.float32
    assert subject.lm_head_weight(REMOTE).device == placed(REMOTE)


def test_teacher_logits_go_through_the_targets_own_projection():
    """This is what makes an exact full-vocabulary teacher affordable: the buffer
    holds `[T, hidden]` and the `[T, vocab]` logits are regenerated here."""
    target = FakeTarget()
    subject = provider(target)
    hidden = torch.randn(6, HIDDEN)
    logits = subject.teacher_logits(hidden)

    assert logits.shape == (6, VOCAB)
    assert logits.dtype == torch.float32
    assert target.compute_logits_calls == 1
    torch.testing.assert_close(logits, target.lm_head(hidden), atol=1e-5, rtol=1e-5)


def test_teacher_logits_cast_to_the_target_dtype():
    subject = provider(FakeTarget().to(torch.bfloat16))
    logits = subject.teacher_logits(torch.randn(4, HIDDEN, dtype=torch.float32))
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_a_multimodal_target_is_borrowed_from_through_its_language_model():
    """A multimodal wrapper carries neither the embedding table nor the LM head: both
    sit one level down, in the text stack it delegates generation to."""
    target = FakeMultiModalTarget()
    subject = provider(target)
    hidden = torch.randn(3, HIDDEN)

    assert subject.embedding_weight().shape == (VOCAB, HIDDEN)
    assert subject.lm_head_weight().shape == (VOCAB, HIDDEN)
    torch.testing.assert_close(
        subject.teacher_logits(hidden),
        target.language_model.lm_head(hidden),
        atol=1e-5,
        rtol=1e-5,
    )


def test_a_target_without_an_lm_head_is_refused():
    class NoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)

    with pytest.raises(ValueError, match="LM head"):
        provider(NoHead())


def test_a_target_without_embeddings_is_refused():
    class NoEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    with pytest.raises(ValueError, match="embedding table"):
        provider(NoEmbed())


def test_the_objective_can_score_through_a_stub(head, batch, config):
    """The whole point of the contract: no engine, no GPU, real objective."""
    from tests.conftest import make_objective

    subject = provider()
    objective = make_objective(config, head, subject.teacher_logits)
    loss, metrics = objective(student_hidden=head.replay(batch), batch=batch)
    assert torch.isfinite(loss)
    assert metrics["count/tokens"] > 0
