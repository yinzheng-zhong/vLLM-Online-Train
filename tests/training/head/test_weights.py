"""The head's weight lifecycle, which is a different job from its forward."""

import torch

from tests.conftest import (
    HIDDEN,
    VOCAB,
    head_factory,
    head_weights,
    make_arch,
    mask_builder,
    weight_naming,
)
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead


def test_the_factory_freezes_what_it_builds():
    """A head straight from the factory must already have the borrowed tensors
    frozen, or the first optimizer step trains an 800 MB embedding table."""
    head = head_factory.create(make_arch())
    assert not head.model.embed_tokens.weight.requires_grad
    assert not head.lm_head.weight.requires_grad


def test_load_shared_converts_dtype():
    """`copy_` converts, so this works whichever precision the target holds."""
    head = head_factory.create(make_arch())
    head_weights.cast_shared(head, torch.bfloat16)
    head_weights.load_shared(
        head, torch.randn(VOCAB, HIDDEN), torch.randn(VOCAB, HIDDEN)
    )
    assert head.model.embed_tokens.weight.dtype == torch.bfloat16
    assert head.lm_head.weight.dtype == torch.bfloat16


def test_cast_shared_does_not_thaw():
    head = head_factory.create(make_arch())
    head_weights.cast_shared(head, torch.bfloat16)
    assert not head.lm_head.weight.requires_grad
    assert not head.model.embed_tokens.weight.requires_grad


def test_shared_tensors_can_be_held_at_a_lower_precision(batch):
    """The borrowed tensors dwarf everything being trained, and training must still
    run end to end with them cast down."""
    head = head_factory.create(
        make_arch(), generator=torch.Generator().manual_seed(0)
    )
    head_weights.cast_shared(head, torch.bfloat16)

    assert head.compute_dtype == torch.float32
    hidden = head.replay(batch)
    assert hidden.dtype == torch.float32
    assert torch.isfinite(hidden).all()

    logits = head.project_to_vocab(hidden)
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    assert head.model.fc.weight.grad is not None
    assert head.lm_head.weight.grad is None, "gradient leaked into a frozen tensor"


def test_init_is_deterministic_under_a_seed():
    """A checkpoint you cannot regenerate is a checkpoint you cannot bisect against.

    Scoped to the tensors that are actually written: `embed_tokens` keeps its
    unseeded constructor values because it is never exported.
    """
    arch = make_arch()

    def checkpoint_of(seed: int) -> dict[str, torch.Tensor]:
        head = head_factory.create(
            arch, generator=torch.Generator().manual_seed(seed)
        )
        return dict(weight_naming.rewrite(head_weights.export(head), arch))

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
    head = head_factory.create(make_arch())
    torch.testing.assert_close(
        head.model.norm.weight, torch.ones_like(head.model.norm.weight)
    )
    torch.testing.assert_close(
        head.model.hidden_norm.weight, torch.ones_like(head.model.hidden_norm.weight)
    )


def test_export_strips_the_state_dict_model_prefix():
    """`load_weights` re-adds `model.`, so exporting `state_dict()` unchanged would
    produce `model.model.*` and load nothing."""
    head = head_factory.create(make_arch())
    names = {name for name, _ in head_weights.export(head)}
    assert "fc.weight" in names
    assert "lm_head.weight" in names
    assert not any(name.startswith("model.") for name in names)


def test_export_covers_every_parameter():
    arch = make_arch()
    head = head_factory.create(arch)
    assert len(dict(head_weights.export(head))) == len(head.state_dict())


def test_freeze_survives_a_manual_thaw():
    head = TrainableDFlashHead(make_arch(), mask_builder)
    head.lm_head.weight.requires_grad_(True)
    head_weights.freeze_shared(head)
    assert not head.lm_head.weight.requires_grad
