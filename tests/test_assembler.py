"""What `build_head` places on the training device.

The head holds two vocabulary-sized tensors and every borrow is another one, so these
pin the precision each is placed at and the number of copies alive at once.
"""

from types import SimpleNamespace
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
from vllm_online_train.config.shapes import ResolvedConfig
from vllm_online_train.contracts.provider import StateProvider
from vllm_online_train.engine.capture.rollout_buffer import RolloutBuffer
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


class StubRunner:
    """A model runner exposing only what `build` reads off it."""

    def __init__(self, target: FakeTarget, device: torch.device) -> None:
        """
        Args:
            target: The model to borrow the embedding table and LM head from.
            device: The engine's device.
        """
        self.vllm_config = fake_vllm_config()
        self.device = device
        self._target = target

    def get_model(self) -> FakeTarget:
        """The model the engine is serving."""
        return self._target


class CpuAssembler(SessionAssembler):
    """A `SessionAssembler` whose capture allocates no CUDA stream, so the rest of
    `build` runs on the CPU the suite runs on."""

    def build_capture(
        self, config: ResolvedConfig, buffer: RolloutBuffer, device: torch.device
    ) -> Any:
        return SimpleNamespace(observe=lambda step: None, abort=lambda req_ids: None)


def build_session() -> Any:
    """Assemble a session the way the capture hook does, from inside the engine's
    `inference_mode` region.

    Returns:
        The started session. The caller stops it.
    """
    runner = StubRunner(FakeTarget(), CPU)
    with torch.inference_mode():
        return CpuAssembler().build(runner, make_settings())


def test_the_session_is_built_outside_the_engines_inference_mode():
    """`build` is reached from inside `execute_model`, which vLLM decorates with
    `torch.inference_mode()`. A tensor allocated there carries the inference flag for
    life, `.clone()` inherits it, and the first backward raises -- so every training
    step fails and the thread retries forever while serving looks healthy."""
    session = build_session()
    try:
        head = session.manager.head
        assert not head.model.fc.weight.is_inference()
        assert not head.model.hidden_norm.weight.is_inference()
        assert not head.model.embed_tokens.weight.is_inference()
        assert not head.lm_head.weight.is_inference()
    finally:
        session.stop()


def test_a_session_built_under_inference_mode_can_take_a_backward():
    """The property the inference flag actually costs."""
    session = build_session()
    try:
        head = session.manager.head
        features = torch.randn(1, 4, head.arch.feature_width, dtype=head.compute_dtype)
        head.project_context(features).sum().backward()
        assert head.model.fc.weight.grad is not None
    finally:
        session.stop()
