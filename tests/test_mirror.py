"""Serving the borrowed target state from a second device.

`train_device` puts the head, its gradients, its optimizer state and its batches on a
device the engine is not serving on, which means the teacher can no longer be scored
through the engine's own module. These pin the substitute: its own copy of the
projection, on its own device, checked against the engine's before any step runs.

`conftest.REMOTE` stands in for the second GPU throughout. `.to()` copies onto it and it
compares unequal to a plain `"cpu"`, so the two-device path is exercised on a machine
with one CUDA device or none.
"""

import pytest
import torch

from tests.conftest import (
    CPU,
    HIDDEN,
    REMOTE,
    VOCAB,
    FakeTarget,
    fake_vllm_config,
    head_factory,
    head_weights,
    make_arch,
    make_collator,
    make_config,
    make_objective,
    make_settings,
    placed,
)
from vllm_online_train.assembler import SessionAssembler
from vllm_online_train.contracts.provider import StateProvider
from vllm_online_train.mirror import MirroredStateProvider
from vllm_online_train.provider import EngineStateProvider


def engine_provider(
    target: FakeTarget | None = None, serving_dtype: torch.dtype = torch.float32
) -> EngineStateProvider:
    return EngineStateProvider(target or FakeTarget(), CPU, serving_dtype)


def mirror(
    source: EngineStateProvider | None = None, dtype: torch.dtype = torch.float32
) -> MirroredStateProvider:
    return MirroredStateProvider(source or engine_provider(), REMOTE, dtype)


def test_it_satisfies_the_contract():
    assert isinstance(mirror(), StateProvider)


def test_it_reports_the_training_device_and_forwards_both_dtypes():
    """The head, the collator and the optimizer all follow `provider.device`, so the
    mirror reporting the engine's device would leave them on the serving GPU."""
    source = engine_provider(serving_dtype=torch.bfloat16)
    subject = mirror(source)
    assert subject.device == REMOTE
    assert subject.serving_dtype == torch.bfloat16
    assert subject.target_dtype == source.target_dtype


def test_borrowed_weights_land_on_the_training_device():
    subject = mirror()
    assert subject.embedding_weight().device == placed(REMOTE)
    assert subject.lm_head_weight().device == placed(REMOTE)
    assert subject.embedding_weight().dtype == torch.float32
    assert subject.lm_head_weight().dtype == torch.float32


def test_an_explicit_device_still_wins():
    """`build_head` takes the default, but the argument is what keeps the fp32 upcast
    off the engine's device."""
    assert mirror().lm_head_weight(CPU).device == CPU


def test_the_projection_is_held_at_the_requested_dtype():
    subject = mirror(dtype=torch.bfloat16)
    assert subject.projection_dtype == torch.bfloat16
    logits = subject.teacher_logits(torch.randn(4, HIDDEN, device=REMOTE))
    assert logits.dtype == torch.float32


def test_the_teacher_never_touches_the_engines_module():
    """The whole point: no matmul on the serving GPU and no `[N, vocab]` logits crossing
    the bus once training is under way."""
    target = FakeTarget()
    subject = mirror(engine_provider(target))
    hidden = torch.randn(6, HIDDEN)

    logits = subject.teacher_logits(hidden.to(REMOTE))

    assert logits.shape == (6, VOCAB)
    assert target.compute_logits_calls == 0
    torch.testing.assert_close(
        logits.to(CPU), target.lm_head(hidden), atol=1e-5, rtol=1e-5
    )


def test_verify_parity_accepts_a_plain_projection():
    target = FakeTarget()
    subject = mirror(engine_provider(target))
    assert subject.verify_parity() < subject.PARITY_TOLERANCE
    assert target.compute_logits_calls == 1


def test_verify_parity_rejects_a_scaled_projection():
    """`LogitsProcessor` carries a scale and a soft cap; either makes a bare matmul the
    wrong teacher, and a wrong teacher trains silently."""

    class ScaledTarget(FakeTarget):
        def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
            return super().compute_logits(hidden) * 10.0

    with pytest.raises(ValueError, match="deviates from the engine"):
        mirror(engine_provider(ScaledTarget())).verify_parity()


def test_verify_parity_rejects_a_different_vocabulary():
    """A padded or tensor-parallel-sharded head projects to a width the mirrored copy
    does not."""

    class ShardedTarget(FakeTarget):
        def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
            return super().compute_logits(hidden)[:, : VOCAB // 2]

    with pytest.raises(ValueError, match="yields"):
        mirror(engine_provider(ShardedTarget())).verify_parity()


def test_a_batch_on_the_training_device_scores_through_the_mirror(config, records):
    """The end of the path: the head, the batch and the teacher all on the training
    device, with the engine's module untouched."""
    target = FakeTarget()
    subject = mirror(engine_provider(target))
    head = head_factory.create(make_arch()).to(device=REMOTE, dtype=torch.float32)
    head_weights.load_shared(
        head, subject.embedding_weight(), subject.lm_head_weight()
    )
    batch = make_collator(config, device=REMOTE).collate(records)

    objective = make_objective(config, head, subject.teacher_logits)
    loss, metrics = objective(student_hidden=head.replay(batch), batch=batch)

    assert torch.isfinite(loss)
    assert metrics["count/tokens"] > 0
    assert target.compute_logits_calls == 0


@pytest.fixture(scope="module")
def assembler() -> SessionAssembler:
    return SessionAssembler()


@pytest.mark.parametrize("requested", [None, "cpu"])
def test_the_assembler_keeps_the_engine_provider_on_one_device(assembler, requested):
    """Training on the serving device must stay exactly what it was: the engine's own
    projection, no second copy of the LM head."""
    config = make_config(make_settings(train_device=requested))
    engine = engine_provider()
    assert assembler.build_provider(config, engine) is engine


def test_the_assembler_mirrors_onto_a_second_device(assembler):
    config = make_config(make_settings(train_device="cpu:0"))
    provider = assembler.build_provider(config, engine_provider())
    assert isinstance(provider, MirroredStateProvider)
    assert provider.device == REMOTE


def test_the_head_is_built_on_the_training_device(assembler):
    """The head is what drags the gradients and the AdamW moments along with it, so it
    following the provider is the whole of what `train_device` buys."""
    config = make_config(make_settings(train_device="cpu:0"))
    provider = assembler.build_provider(
        config, engine_provider(serving_dtype=torch.bfloat16)
    )

    head = assembler.build_head(fake_vllm_config(), config, provider)

    assert head.model.fc.weight.device == placed(REMOTE)
    assert head.model.fc.weight.dtype == torch.float32
    assert head.lm_head.weight.device == placed(REMOTE)
    assert head.lm_head.weight.dtype == torch.bfloat16
    assert not head.lm_head.weight.requires_grad
    assert not head.model.embed_tokens.weight.requires_grad


def test_the_mirrored_projection_matches_the_heads_shared_precision(assembler):
    """The head's borrowed LM head and the teacher's copy of it are the same matrix; a
    KL between two precisions of one projection measures the cast, not the head."""
    engine = engine_provider(serving_dtype=torch.bfloat16)

    config = make_config(make_settings(train_device="cpu:0"))
    provider = assembler.build_provider(config, engine)
    assert provider.projection_dtype == torch.bfloat16

    config = make_config(make_settings(train_device="cpu:0", shared_dtype="float32"))
    provider = assembler.build_provider(config, engine)
    assert provider.projection_dtype == torch.float32
