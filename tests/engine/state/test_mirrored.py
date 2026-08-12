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
from vllm_online_train.engine.state.engine import EngineStateProvider
from vllm_online_train.engine.state.mirrored import MirroredStateProvider
from vllm_online_train.engine.state.target_locator import TargetLocator


def make_engine_provider(
    target_model: FakeTarget | None = None,
    serving_dtype: torch.dtype = torch.float32,
) -> EngineStateProvider:
    return EngineStateProvider(
        TargetLocator(), target_model or FakeTarget(), CPU, serving_dtype
    )


def make_mirror(
    source_provider: EngineStateProvider | None = None,
    dtype: torch.dtype = torch.float32,
) -> MirroredStateProvider:
    return MirroredStateProvider(
        source_provider or make_engine_provider(), REMOTE, dtype
    )


def test_it_satisfies_the_contract():
    assert isinstance(make_mirror(), StateProvider)


def test_it_reports_the_training_device_and_forwards_both_dtypes():
    """The head, the collator and the optimizer all follow `provider.device`, so the
    mirror reporting the engine's device would leave them on the serving GPU."""
    source_provider = make_engine_provider(serving_dtype=torch.bfloat16)
    mirrored_provider = make_mirror(source_provider)
    assert mirrored_provider.device == REMOTE
    assert mirrored_provider.serving_dtype == torch.bfloat16
    assert mirrored_provider.target_dtype == source_provider.target_dtype


def test_borrowed_weights_land_on_the_training_device():
    mirrored_provider = make_mirror()
    assert mirrored_provider.embedding_weight().device == placed(REMOTE)
    assert mirrored_provider.lm_head_weight().device == placed(REMOTE)
    assert mirrored_provider.embedding_weight().dtype == torch.float32
    assert mirrored_provider.lm_head_weight().dtype == torch.float32


def test_an_explicit_device_still_wins():
    """`build_head` takes the default, but the argument is what keeps the fp32 upcast
    off the engine's device."""
    assert make_mirror().lm_head_weight(CPU).device == CPU


def test_the_projection_is_held_at_the_requested_dtype():
    mirrored_provider = make_mirror(dtype=torch.bfloat16)
    assert mirrored_provider.projection_dtype == torch.bfloat16
    logits = mirrored_provider.teacher_logits(
        torch.randn(4, HIDDEN, device=REMOTE)
    )
    assert logits.dtype == torch.float32


def test_the_projection_is_borrowed_at_the_dtype_it_is_held_at(monkeypatch):
    """Borrowing at fp32 and casting afterwards would put a second vocabulary-sized
    copy on the training device."""
    source_provider = make_engine_provider()
    asked_dtypes: list[torch.dtype | None] = []
    borrow = source_provider.lm_head_weight

    def recording(
        device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        asked_dtypes.append(dtype)
        return borrow(device, dtype)

    monkeypatch.setattr(source_provider, "lm_head_weight", recording)
    assert MirroredStateProvider(
        source_provider, REMOTE, torch.bfloat16
    ).projection_dtype == torch.bfloat16
    assert asked_dtypes == [torch.bfloat16]


def test_a_borrow_through_the_mirror_carries_the_dtype():
    mirrored_provider = make_mirror()
    assert (
        mirrored_provider.embedding_weight(dtype=torch.bfloat16).dtype
        == torch.bfloat16
    )
    assert (
        mirrored_provider.lm_head_weight(dtype=torch.bfloat16).dtype
        == torch.bfloat16
    )


def test_the_teacher_never_touches_the_engines_module():
    """The whole point: no matmul on the serving GPU and no `[N, vocab]` logits crossing
    the bus once training is under way."""
    target_model = FakeTarget()
    mirrored_provider = make_mirror(make_engine_provider(target_model))
    hidden_states = torch.randn(6, HIDDEN)

    logits = mirrored_provider.teacher_logits(hidden_states.to(REMOTE))

    assert logits.shape == (6, VOCAB)
    assert target_model.compute_logits_calls == 0
    torch.testing.assert_close(
        logits.to(CPU), target_model.lm_head(hidden_states), atol=1e-5, rtol=1e-5
    )


def test_verify_parity_accepts_a_plain_projection():
    target_model = FakeTarget()
    mirrored_provider = make_mirror(make_engine_provider(target_model))
    assert mirrored_provider.verify_parity() < mirrored_provider.PARITY_TOLERANCE
    assert target_model.compute_logits_calls == 1


def test_verify_parity_rejects_a_scaled_projection():
    """`LogitsProcessor` carries a scale and a soft cap; either makes a bare matmul the
    wrong teacher, and a wrong teacher trains silently."""

    class ScaledTarget(FakeTarget):
        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return super().compute_logits(hidden_states) * 10.0

    with pytest.raises(ValueError, match="deviates from the engine"):
        make_mirror(make_engine_provider(ScaledTarget())).verify_parity()


def test_verify_parity_rejects_a_different_vocabulary():
    """A padded or tensor-parallel-sharded head projects to a width the mirrored copy
    does not."""

    class ShardedTarget(FakeTarget):
        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return super().compute_logits(hidden_states)[:, : VOCAB // 2]

    with pytest.raises(ValueError, match="yields"):
        make_mirror(make_engine_provider(ShardedTarget())).verify_parity()


def test_a_batch_on_the_training_device_scores_through_the_mirror(
    resolved_config, rollout_records
):
    """The end of the path: the head, the batch and the teacher all on the training
    device, with the engine's module untouched."""
    target_model = FakeTarget()
    mirrored_provider = make_mirror(make_engine_provider(target_model))
    draft_head = head_factory.create(make_arch()).to(
        device=REMOTE, dtype=torch.float32
    )
    head_weights.load_shared(
        draft_head,
        mirrored_provider.embedding_weight(),
        mirrored_provider.lm_head_weight(),
    )
    replay_batch = make_collator(resolved_config, device=REMOTE).collate(
        rollout_records
    )

    sft_objective = make_objective(
        resolved_config, draft_head, mirrored_provider.teacher_logits
    )
    loss, metrics = sft_objective(
        student_hidden=draft_head.replay(replay_batch), replay_batch=replay_batch
    )

    assert torch.isfinite(loss)
    assert metrics["count/tokens"] > 0
    assert target_model.compute_logits_calls == 0


@pytest.fixture(scope="module")
def assembler() -> SessionAssembler:
    return SessionAssembler()


@pytest.mark.parametrize("requested_device", [None, "cpu"])
def test_the_assembler_keeps_the_engine_provider_on_one_device(
    assembler, requested_device
):
    """Training on the serving device must stay exactly what it was: the engine's own
    projection, no second copy of the LM head."""
    resolved_config = make_config(make_settings(train_device=requested_device))
    engine_provider = make_engine_provider()
    assert (
        assembler.build_provider(resolved_config, engine_provider)
        is engine_provider
    )


def test_the_assembler_mirrors_onto_a_second_device(assembler):
    resolved_config = make_config(make_settings(train_device="cpu:0"))
    state_provider = assembler.build_provider(
        resolved_config, make_engine_provider()
    )
    assert isinstance(state_provider, MirroredStateProvider)
    assert state_provider.device == REMOTE


def test_the_head_is_built_on_the_training_device(assembler):
    """The head is what drags the gradients and the AdamW moments along with it, so it
    following the provider is the whole of what `train_device` buys."""
    resolved_config = make_config(make_settings(train_device="cpu:0"))
    state_provider = assembler.build_provider(
        resolved_config, make_engine_provider(serving_dtype=torch.bfloat16)
    )

    draft_head = assembler.build_head(
        fake_vllm_config(), resolved_config, state_provider
    )

    assert draft_head.model.fc.weight.device == placed(REMOTE)
    assert draft_head.model.fc.weight.dtype == torch.float32
    assert draft_head.lm_head.weight.device == placed(REMOTE)
    assert draft_head.lm_head.weight.dtype == torch.bfloat16
    assert not draft_head.lm_head.weight.requires_grad
    assert not draft_head.model.embed_tokens.weight.requires_grad


def test_the_mirrored_projection_matches_the_heads_shared_precision(assembler):
    """The head's borrowed LM head and the teacher's copy of it are the same matrix; a
    KL between two precisions of one projection measures the cast, not the head."""
    engine_provider = make_engine_provider(serving_dtype=torch.bfloat16)

    resolved_config = make_config(make_settings(train_device="cpu:0"))
    state_provider = assembler.build_provider(resolved_config, engine_provider)
    assert state_provider.projection_dtype == torch.bfloat16

    resolved_config = make_config(
        make_settings(train_device="cpu:0", shared_dtype="float32")
    )
    state_provider = assembler.build_provider(resolved_config, engine_provider)
    assert state_provider.projection_dtype == torch.float32
