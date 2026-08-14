"""What `build_head` places on the training device.

The head holds two vocabulary-sized tensors and every borrow is another one, so these
pin the precision each is placed at and the number of copies alive at once.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.conftest import (
    CPU,
    REMOTE,
    FakeTarget,
    checkpoint_writer,
    fake_vllm_config,
    head_factory,
    head_weights,
    make_arch,
    make_config,
    make_settings,
    placed,
)
from vllm_online_train.assembler import SessionAssembler
from vllm_online_train.config.shapes import ResolvedConfig
from vllm_online_train.contracts.provider import StateProvider
from vllm_online_train.engine.capture.rollouts.rollout_buffer import RolloutBuffer
from vllm_online_train.engine.state.engine import EngineStateProvider
from vllm_online_train.engine.state.target_locator import TargetLocator
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead


class RecordingProvider:
    """Passes every call through, recording the dtype each borrow is asked for."""

    def __init__(self, inner_provider: StateProvider) -> None:
        self.inner_provider = inner_provider
        self.asked_dtypes: list[torch.dtype | None] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner_provider, name)

    def embedding_weight(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        self.asked_dtypes.append(dtype)
        return self.inner_provider.embedding_weight(device, dtype)

    def lm_head_weight(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        self.asked_dtypes.append(dtype)
        return self.inner_provider.lm_head_weight(device, dtype)


NO_SERVED_HEAD = Path("/nonexistent/head")
"""A served directory holding no weights, so `build_head` keeps the random head."""


def build(
    target_model: FakeTarget | None = None,
    device: torch.device = CPU,
    source_dir: Path = NO_SERVED_HEAD,
    **settings_overrides: Any,
) -> tuple[TrainableDFlashHead, RecordingProvider]:
    """Build a head the way a session does.

    Args:
        target_model: The model to borrow from. Defaults to a bf16 `FakeTarget`.
        device: The training device the provider reports.
        source_dir: The served head's directory.
        **settings_overrides: Flat settings fields.

    Returns:
        The head and the provider it borrowed through.
    """
    engine_provider = EngineStateProvider(
        TargetLocator(),
        target_model if target_model is not None else FakeTarget().to(torch.bfloat16),
        device,
        torch.bfloat16,
    )
    state_provider = RecordingProvider(engine_provider)
    resolved_config = make_config(
        online_train_settings=make_settings(**settings_overrides)
    )
    draft_head = SessionAssembler().build_head(
        fake_vllm_config(), resolved_config, state_provider, source_dir
    )
    return draft_head, state_provider


def test_the_borrowed_tensors_are_held_at_the_shared_dtype():
    draft_head, _ = build()
    assert draft_head.model.embed_tokens.weight.dtype == torch.bfloat16
    assert draft_head.lm_head.weight.dtype == torch.bfloat16
    assert not draft_head.model.embed_tokens.weight.requires_grad
    assert not draft_head.lm_head.weight.requires_grad


def test_the_trained_tensors_stay_at_fp32():
    draft_head, _ = build()
    assert draft_head.model.fc.weight.dtype == torch.float32
    assert draft_head.model.fc.weight.requires_grad
    assert draft_head.compute_dtype == torch.float32


def test_each_borrow_is_taken_at_the_dtype_the_head_holds_it_at():
    """An fp32 borrow that is cast afterwards costs twice the bytes of the tensor it
    ends up as, on the device with the least room."""
    _, state_provider = build()
    assert state_provider.asked_dtypes == [torch.bfloat16, torch.bfloat16]


def test_fp32_shared_weights_are_borrowed_at_fp32():
    draft_head, state_provider = build(shared_dtype="float32")
    assert state_provider.asked_dtypes == [torch.float32, torch.float32]
    assert draft_head.model.embed_tokens.weight.dtype == torch.float32
    assert draft_head.lm_head.weight.dtype == torch.float32


def test_the_targets_weights_land_in_the_head():
    target_model = FakeTarget().to(torch.bfloat16)
    draft_head, _ = build(target_model)
    torch.testing.assert_close(
        draft_head.model.embed_tokens.weight, target_model.embed.weight
    )
    torch.testing.assert_close(
        draft_head.lm_head.weight, target_model.lm_head.weight
    )


def test_the_head_lands_on_the_training_device():
    draft_head, _ = build(device=REMOTE)
    assert draft_head.model.embed_tokens.weight.device == placed(REMOTE)
    assert draft_head.model.fc.weight.device == placed(REMOTE)


def write_served_head(directory: Path) -> Any:
    """Write a trained head into a directory the way an export does.

    Args:
        directory: Destination.

    Returns:
        The head that was written.
    """
    head_arch = make_arch()
    served_head = head_factory.create(
        head_arch, generator=torch.Generator().manual_seed(11)
    )
    checkpoint_writer.write(
        directory,
        head_weights.export(served_head),
        head_arch,
        dtype=torch.bfloat16,
    )
    return served_head


def test_the_served_checkpoint_is_adopted_by_default(tmp_path):
    """The bug this closes: serving v1 while training a fresh random head throws away
    every step of the previous run and restarts top-1 agreement at chance."""
    served_head = write_served_head(tmp_path)
    draft_head, _ = build(source_dir=tmp_path)

    torch.testing.assert_close(
        draft_head.model.fc.weight,
        served_head.model.fc.weight,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        draft_head.model.layers[0].self_attn.qkv_proj.weight,
        served_head.model.layers[0].self_attn.qkv_proj.weight,
        atol=1e-2,
        rtol=1e-2,
    )
    assert draft_head.model.fc.weight.dtype == torch.float32
    assert draft_head.model.fc.weight.requires_grad


def test_a_random_head_is_available_on_request(tmp_path):
    """The from-scratch baseline the README's A/B needs."""
    served_head = write_served_head(tmp_path)
    draft_head, _ = build(source_dir=tmp_path, head_init_source="random")

    assert not torch.allclose(
        draft_head.model.fc.weight,
        served_head.model.fc.weight.to(torch.float32),
        atol=1e-2,
    )


def test_a_served_directory_with_no_weights_leaves_the_head_random(tmp_path):
    """A draft served from a format this reader cannot open must not take the engine
    down; serving is unaffected by what the trainer starts from."""
    draft_head, _ = build(source_dir=tmp_path)
    assert draft_head.model.fc.weight.requires_grad


def test_a_served_checkpoint_of_another_shape_is_refused(tmp_path):
    """Loading it would train against a projection the serving path never applies, and
    the sag in acceptance is indistinguishable from a badly-trained head."""
    wide_arch = make_arch(hidden_size=64)
    checkpoint_writer.write(
        tmp_path,
        head_weights.export(head_factory.create(wide_arch)),
        wide_arch,
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="different shape"):
        build(source_dir=tmp_path)


class StubRunner:
    """A model runner exposing only what `build` reads off it."""

    def __init__(self, target_model: FakeTarget, device: torch.device) -> None:
        """
        Args:
            target_model: The model to borrow the embedding table and LM head from.
            device: The engine's device.
        """
        self.vllm_config = fake_vllm_config()
        self.device = device
        self._target_model = target_model

    def get_model(self) -> FakeTarget:
        """The model the engine is serving."""
        return self._target_model


class CpuAssembler(SessionAssembler):
    """A `SessionAssembler` whose capture allocates no CUDA stream, so the rest of
    `build` runs on the CPU the suite runs on."""

    def build_capture(
        self,
        resolved_config: ResolvedConfig,
        rollout_buffer: RolloutBuffer,
        device: torch.device,
    ) -> Any:
        return SimpleNamespace(
            observe=lambda engine_step: None, abort=lambda req_ids: None
        )


def build_session() -> Any:
    """Assemble a session the way the capture hook does, from inside the engine's
    `inference_mode` region.

    Returns:
        The started session. The caller stops it.
    """
    model_runner = StubRunner(FakeTarget(), CPU)
    with torch.inference_mode():
        return CpuAssembler().build(model_runner, make_settings())


def test_the_session_is_built_outside_the_engines_inference_mode():
    """`build` is reached from inside `execute_model`, which vLLM decorates with
    `torch.inference_mode()`. A tensor allocated there carries the inference flag for
    life, `.clone()` inherits it, and the first backward raises -- so every training
    step fails and the thread retries forever while serving looks healthy."""
    online_train_session = build_session()
    try:
        draft_head = online_train_session.online_train_manager.draft_head
        assert not draft_head.model.fc.weight.is_inference()
        assert not draft_head.model.hidden_norm.weight.is_inference()
        assert not draft_head.model.embed_tokens.weight.is_inference()
        assert not draft_head.lm_head.weight.is_inference()
    finally:
        online_train_session.stop()


def test_the_session_reports_the_pool_on_a_timer():
    """The status thread reads the same gate and pool the trainer does, and writes to
    the same sink, so the two views of a run cannot disagree."""
    online_train_session = build_session()
    try:
        status_thread = online_train_session.status_thread
        assert status_thread.enabled
        assert status_thread.idle_gate is online_train_session.idle_gate
        assert status_thread.idle_gate is online_train_session.trainer_thread.idle_gate
        assert status_thread.step_runner is online_train_session.online_train_manager
        assert status_thread.metrics_sink is online_train_session.metrics_sink
    finally:
        online_train_session.stop()


def test_a_session_built_under_inference_mode_can_take_a_backward():
    """The property the inference flag actually costs."""
    online_train_session = build_session()
    try:
        draft_head = online_train_session.online_train_manager.draft_head
        target_features = torch.randn(
            1, 4, draft_head.head_arch.feature_width, dtype=draft_head.compute_dtype
        )
        draft_head.project_context(target_features).sum().backward()
        assert draft_head.model.fc.weight.grad is not None
    finally:
        online_train_session.stop()
