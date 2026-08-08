"""What `build_head` places on the training device.

The head holds two vocabulary-sized tensors and every borrow is another one, so these
pin the precision each is placed at and the number of copies alive at once.
"""

from typing import Any

import torch

from tests.conftest import (
    CPU,
    REMOTE,
    FakeTarget,
    fake_vllm_config,
    make_config,
    make_settings,
    placed,
)
from vllm_online_train.assembler import SessionAssembler
from vllm_online_train.contracts.provider import StateProvider
from vllm_online_train.engine.state.engine import EngineStateProvider
from vllm_online_train.engine.state.target_locator import TargetLocator
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead


class RecordingProvider:
    """Passes every call through, recording the dtype each borrow is asked for."""

    def __init__(self, inner: StateProvider) -> None:
        self.inner = inner
        self.asked: list[torch.dtype | None] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def embedding_weight(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        self.asked.append(dtype)
        return self.inner.embedding_weight(device, dtype)

    def lm_head_weight(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        self.asked.append(dtype)
        return self.inner.lm_head_weight(device, dtype)


def build(
    target: FakeTarget | None = None,
    device: torch.device = CPU,
    **settings_overrides: Any,
) -> tuple[TrainableDFlashHead, RecordingProvider]:
    """Build a head the way a session does.

    Args:
        target: The model to borrow from. Defaults to a bf16 `FakeTarget`.
        device: The training device the provider reports.
        **settings_overrides: Flat settings fields.

    Returns:
        The head and the provider it borrowed through.
    """
    engine = EngineStateProvider(
        TargetLocator(),
        target if target is not None else FakeTarget().to(torch.bfloat16),
        device,
        torch.bfloat16,
    )
    provider = RecordingProvider(engine)
    config = make_config(settings=make_settings(**settings_overrides))
    head = SessionAssembler().build_head(fake_vllm_config(), config, provider)
    return head, provider


def test_the_borrowed_tensors_are_held_at_the_shared_dtype():
    head, _ = build()
    assert head.model.embed_tokens.weight.dtype == torch.bfloat16
    assert head.lm_head.weight.dtype == torch.bfloat16
    assert not head.model.embed_tokens.weight.requires_grad
    assert not head.lm_head.weight.requires_grad


def test_the_trained_tensors_stay_at_fp32():
    head, _ = build()
    assert head.model.fc.weight.dtype == torch.float32
    assert head.model.fc.weight.requires_grad
    assert head.compute_dtype == torch.float32


def test_each_borrow_is_taken_at_the_dtype_the_head_holds_it_at():
    """An fp32 borrow that is cast afterwards costs twice the bytes of the tensor it
    ends up as, on the device with the least room."""
    _, provider = build()
    assert provider.asked == [torch.bfloat16, torch.bfloat16]


def test_fp32_shared_weights_are_borrowed_at_fp32():
    head, provider = build(shared_dtype="float32")
    assert provider.asked == [torch.float32, torch.float32]
    assert head.model.embed_tokens.weight.dtype == torch.float32
    assert head.lm_head.weight.dtype == torch.float32


def test_the_targets_weights_land_in_the_head():
    target = FakeTarget().to(torch.bfloat16)
    head, _ = build(target)
    torch.testing.assert_close(head.model.embed_tokens.weight, target.embed.weight)
    torch.testing.assert_close(head.lm_head.weight, target.lm_head.weight)


def test_the_head_lands_on_the_training_device():
    head, _ = build(device=REMOTE)
    assert head.model.embed_tokens.weight.device == placed(REMOTE)
    assert head.model.fc.weight.device == placed(REMOTE)
