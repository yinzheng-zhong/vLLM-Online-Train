"""The head's weight lifecycle, which is a different job from its forward."""

import torch

from tests.conftest import (
    HIDDEN,
    VOCAB,
    block_mask_builder,
    head_factory,
    head_weights,
    make_arch,
    weight_name_rewriter,
)
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead


def test_the_factory_freezes_what_it_builds():
    """A head straight from the factory must already have the borrowed tensors
    frozen, or the first optimizer step trains an 800 MB embedding table."""
    draft_head = head_factory.create(make_arch())
    assert not draft_head.model.embed_tokens.weight.requires_grad
    assert not draft_head.lm_head.weight.requires_grad


def test_load_shared_converts_dtype():
    """`copy_` converts, so this works whichever precision the target holds."""
    draft_head = head_factory.create(make_arch())
    head_weights.cast_shared(draft_head, torch.bfloat16)
    head_weights.load_shared(
        draft_head, torch.randn(VOCAB, HIDDEN), torch.randn(VOCAB, HIDDEN)
    )
    assert draft_head.model.embed_tokens.weight.dtype == torch.bfloat16
    assert draft_head.lm_head.weight.dtype == torch.bfloat16


def test_cast_shared_does_not_thaw():
    draft_head = head_factory.create(make_arch())
    head_weights.cast_shared(draft_head, torch.bfloat16)
    assert not draft_head.lm_head.weight.requires_grad
    assert not draft_head.model.embed_tokens.weight.requires_grad


def test_shared_tensors_can_be_held_at_a_lower_precision(replay_batch):
    """The borrowed tensors dwarf everything being trained, and training must still
    run end to end with them cast down."""
    draft_head = head_factory.create(
        make_arch(), generator=torch.Generator().manual_seed(0)
    )
    head_weights.cast_shared(draft_head, torch.bfloat16)

    assert draft_head.compute_dtype == torch.float32
    scored_hidden = draft_head.replay(replay_batch)
    assert scored_hidden.dtype == torch.float32
    assert torch.isfinite(scored_hidden).all()

    logits = draft_head.project_to_vocab(scored_hidden)
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    assert draft_head.model.fc.weight.grad is not None
    assert draft_head.lm_head.weight.grad is None, (
        "gradient leaked into a frozen tensor"
    )


def test_init_is_deterministic_under_a_seed():
    """A checkpoint you cannot regenerate is a checkpoint you cannot bisect against.

    Scoped to the tensors that are actually written: `embed_tokens` keeps its
    unseeded constructor values because it is never exported.
    """
    head_arch = make_arch()

    def checkpoint_of(seed: int) -> dict[str, torch.Tensor]:
        draft_head = head_factory.create(
            head_arch, generator=torch.Generator().manual_seed(seed)
        )
        return dict(
            weight_name_rewriter.rewrite(
                head_weights.export(draft_head), head_arch
            )
        )

    first, second = checkpoint_of(7), checkpoint_of(7)
    assert set(first) == set(second)
    for name, tensor in first.items():
        torch.testing.assert_close(
            tensor, second[name], msg=f"{name} differs across identical seeds"
        )

    assert not torch.allclose(first["fc.weight"], checkpoint_of(8)["fc.weight"])


def test_init_sets_norm_weights_to_one():
    """Norms start at all-ones in both this head and vLLM's `RMSNorm`, which is what
    makes the two comparable at step zero."""
    draft_head = head_factory.create(make_arch())
    torch.testing.assert_close(
        draft_head.model.norm.weight,
        torch.ones_like(draft_head.model.norm.weight),
    )
    torch.testing.assert_close(
        draft_head.model.hidden_norm.weight,
        torch.ones_like(draft_head.model.hidden_norm.weight),
    )


def test_export_strips_the_state_dict_model_prefix():
    """`load_weights` re-adds `model.`, so exporting `state_dict()` unchanged would
    produce `model.model.*` and load nothing."""
    draft_head = head_factory.create(make_arch())
    exported_names = {name for name, _ in head_weights.export(draft_head)}
    assert "fc.weight" in exported_names
    assert "lm_head.weight" in exported_names
    assert not any(name.startswith("model.") for name in exported_names)


def test_export_covers_every_parameter():
    head_arch = make_arch()
    draft_head = head_factory.create(head_arch)
    assert len(dict(head_weights.export(draft_head))) == len(
        draft_head.state_dict()
    )


def test_freeze_survives_a_manual_thaw():
    draft_head = TrainableDFlashHead(make_arch(), block_mask_builder)
    draft_head.lm_head.weight.requires_grad_(True)
    head_weights.freeze_shared(draft_head)
    assert not draft_head.lm_head.weight.requires_grad
