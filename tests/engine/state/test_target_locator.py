"""Finding the borrowed modules across the layouts vLLM serves models in."""

import re

import pytest
import torch
from torch import nn

from tests.conftest import HIDDEN, VOCAB, FakeMultiModalTarget, FakeTarget
from vllm_online_train.engine.state.target_locator import TargetLocator


def test_a_causal_lm_is_read_at_its_own_paths():
    target = FakeTarget()
    embed, lm_head = TargetLocator().find(target)
    assert embed is target.model.embed_tokens
    assert lm_head is target.lm_head


def test_a_multimodal_wrapper_is_read_through_its_language_model():
    target = FakeMultiModalTarget()
    embed, lm_head = TargetLocator().find(target)
    assert embed is target.language_model.model.embed_tokens
    assert lm_head is target.language_model.lm_head


def test_an_embedding_table_at_the_top_level_is_accepted():
    class Flat(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    target = Flat()
    embed, lm_head = TargetLocator().find(target)
    assert embed is target.embed_tokens
    assert lm_head is target.lm_head


def test_the_outer_model_wins_over_the_language_model():
    class Both(FakeMultiModalTarget):
        def __init__(self):
            super().__init__()
            self.model = FakeTarget().model
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    target = Both()
    embed, lm_head = TargetLocator().find(target)
    assert embed is target.model.embed_tokens
    assert lm_head is target.lm_head


def test_a_language_model_that_cannot_be_reached_is_not_fatal():
    """vLLM raises `NotImplementedError` from `get_language_model()` when the model
    marked none, which leaves the outer model as the only place to look."""

    class Unmarked(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeTarget().model
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

        def get_language_model(self):
            raise NotImplementedError("No language model found in Unmarked!")

    target = Unmarked()
    embed, lm_head = TargetLocator().find(target)
    assert embed is target.model.embed_tokens
    assert lm_head is target.lm_head


def test_a_weightless_module_at_a_known_path_is_refused():
    """The path exists on a rank that holds no LM head, so the module being there is
    not enough; it has to carry a weight."""

    class Missing(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeTarget().model
            self.lm_head = nn.Identity()

    with pytest.raises(ValueError, match="LM head"):
        TargetLocator().find(Missing())


def test_the_refusal_names_what_it_searched():
    class Empty(nn.Module):
        pass

    with pytest.raises(ValueError, match=re.escape("model.embed_tokens, embed_tokens")):
        TargetLocator().find(Empty())


def test_a_borrowed_weight_survives_the_walk():
    """The located modules are the live ones, so a weight written on the target shows
    through them."""
    target = FakeMultiModalTarget()
    embed, _ = TargetLocator().find(target)
    with torch.no_grad():
        target.language_model.model.embed_tokens.weight[0].fill_(3.0)
    assert torch.equal(embed.weight[0], torch.full((HIDDEN,), 3.0))
