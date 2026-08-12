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


def make_provider(target_model: nn.Module | None = None) -> EngineStateProvider:
    return EngineStateProvider(
        TargetLocator(), target_model or FakeTarget(), CPU, torch.bfloat16
    )


def test_it_satisfies_the_contract():
    assert isinstance(make_provider(), StateProvider)


def test_it_reports_both_dtypes():
    target_model = FakeTarget()
    state_provider = make_provider(target_model)
    assert state_provider.serving_dtype == torch.bfloat16
    assert state_provider.target_dtype == target_model.embed.weight.dtype
    assert state_provider.device == CPU


def test_borrowed_weights_come_back_detached_in_fp32():
    """The head is trained at fp32 and copies these in; a live view would let a
    training step write into the engine's own weights."""
    target_model = FakeTarget()
    state_provider = make_provider(target_model)

    embedding_weight = state_provider.embedding_weight()
    assert embedding_weight.dtype == torch.float32
    assert not embedding_weight.requires_grad
    assert embedding_weight.shape == (VOCAB, HIDDEN)
    assert embedding_weight.data_ptr() != target_model.embed.weight.data_ptr()

    lm_head_weight = state_provider.lm_head_weight()
    assert lm_head_weight.dtype == torch.float32
    assert not lm_head_weight.requires_grad
    assert lm_head_weight.shape == (VOCAB, HIDDEN)
    assert lm_head_weight.data_ptr() != target_model.lm_head.weight.data_ptr()


def test_borrowed_weights_can_be_asked_for_on_another_device():
    """What lets the head train off the serving GPU: the fp32 copy is allocated where
    the head is, not where the engine is."""
    state_provider = make_provider(FakeTarget().to(torch.bfloat16))

    embedding_weight = state_provider.embedding_weight(REMOTE)
    assert embedding_weight.device == placed(REMOTE)
    assert embedding_weight.dtype == torch.float32
    assert state_provider.lm_head_weight(REMOTE).device == placed(REMOTE)


def test_a_borrow_comes_back_at_the_requested_dtype():
    """The move and the cast are one step, so the copy is never materialised at fp32
    on the way to a narrower dtype."""
    target_model = FakeTarget().to(torch.bfloat16)
    state_provider = make_provider(target_model)

    embedding_weight = state_provider.embedding_weight(dtype=torch.bfloat16)
    assert embedding_weight.dtype == torch.bfloat16
    assert embedding_weight.shape == (VOCAB, HIDDEN)
    assert embedding_weight.data_ptr() != target_model.embed.weight.data_ptr()

    lm_head_weight = state_provider.lm_head_weight(REMOTE, torch.bfloat16)
    assert lm_head_weight.dtype == torch.bfloat16
    assert lm_head_weight.device == placed(REMOTE)
    assert lm_head_weight.data_ptr() != target_model.lm_head.weight.data_ptr()


def test_teacher_logits_go_through_the_targets_own_projection():
    """This is what makes an exact full-vocabulary teacher affordable: the buffer
    holds `[T, hidden]` and the `[T, vocab]` logits are regenerated here."""
    target_model = FakeTarget()
    state_provider = make_provider(target_model)
    hidden_states = torch.randn(6, HIDDEN)
    logits = state_provider.teacher_logits(hidden_states)

    assert logits.shape == (6, VOCAB)
    assert logits.dtype == torch.float32
    assert target_model.compute_logits_calls == 1
    torch.testing.assert_close(
        logits, target_model.lm_head(hidden_states), atol=1e-5, rtol=1e-5
    )


def test_teacher_logits_cast_to_the_target_dtype():
    state_provider = make_provider(FakeTarget().to(torch.bfloat16))
    logits = state_provider.teacher_logits(
        torch.randn(4, HIDDEN, dtype=torch.float32)
    )
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_a_multimodal_target_is_borrowed_from_through_its_language_model():
    """A multimodal wrapper carries neither the embedding table nor the LM head: both
    sit one level down, in the text stack it delegates generation to."""
    target_model = FakeMultiModalTarget()
    state_provider = make_provider(target_model)
    hidden_states = torch.randn(3, HIDDEN)

    assert state_provider.embedding_weight().shape == (VOCAB, HIDDEN)
    assert state_provider.lm_head_weight().shape == (VOCAB, HIDDEN)
    torch.testing.assert_close(
        state_provider.teacher_logits(hidden_states),
        target_model.language_model.lm_head(hidden_states),
        atol=1e-5,
        rtol=1e-5,
    )


def test_a_target_without_an_lm_head_is_refused():
    class NoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)

    with pytest.raises(ValueError, match="LM head"):
        make_provider(NoHead())


def test_a_target_without_embeddings_is_refused():
    class NoEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    with pytest.raises(ValueError, match="embedding table"):
        make_provider(NoEmbed())


def test_the_objective_can_score_through_a_stub(
    draft_head, replay_batch, resolved_config
):
    """The whole point of the contract: no engine, no GPU, real objective."""
    from tests.conftest import make_objective

    state_provider = make_provider()
    sft_objective = make_objective(
        resolved_config, draft_head, state_provider.teacher_logits
    )
    loss, metrics = sft_objective(
        student_hidden=draft_head.replay(replay_batch), replay_batch=replay_batch
    )
    assert torch.isfinite(loss)
    assert metrics["count/tokens"] > 0
