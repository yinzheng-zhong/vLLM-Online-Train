"""Shared fixtures for the online-training tests.

Everything is deliberately tiny (32-wide, 2 layers, 64-token vocabulary) and runs on
CPU. The builders here construct the same collaborators the composition root does, so
the wiring is exercised rather than mocked; only the engine and the target model are
stubbed.
"""

import random
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tests.instances import (
    arch_factory,
    block_builder,
    block_mask_builder,
    checkpoint_writer,
    config_resolver,
    device_placement,
    draft_config_builder,
    drafter_locator,
    head_factory,
    head_weights,
    kl_divergence,
    schedule_factory,
    settings_factory,
    target_layer_planner,
    teacher_scorer,
    valid_positions,
    weight_name_rewriter,
    weight_publisher,
)
from vllm_online_train.config.settings import OnlineTrainSettings
from vllm_online_train.config.shapes import EngineShapes, ResolvedConfig
from vllm_online_train.engine.capture.pool_sampler import PoolSampler
from vllm_online_train.engine.capture.records import BufferStats, RolloutRecord
from vllm_online_train.engine.capture.rollout_buffer import RolloutBuffer
from vllm_online_train.training.collate.anchor_sampler import AnchorSampler
from vllm_online_train.training.collate.device_transfer import DeviceTransfer
from vllm_online_train.training.collate.rollout_collator import RolloutCollator
from vllm_online_train.training.head.arch import DFlashHeadArch
from vllm_online_train.training.loss.position_decay import PositionDecay
from vllm_online_train.training.loss.sft_objective import SFTObjective
from vllm_online_train.training.optim.trainer import OnlineTrainer

__all__ = [
    "arch_factory",
    "block_builder",
    "block_mask_builder",
    "checkpoint_writer",
    "config_resolver",
    "device_placement",
    "draft_config_builder",
    "drafter_locator",
    "head_factory",
    "head_weights",
    "kl_divergence",
    "schedule_factory",
    "settings_factory",
    "target_layer_planner",
    "teacher_scorer",
    "valid_positions",
    "weight_name_rewriter",
    "weight_publisher",
]

HIDDEN = 32
NUM_FEATURES = 3
BLOCK_SIZE = 8
VOCAB = 64
NUM_LAYERS = 2

CPU = torch.device("cpu")

# Stands in for a second device: `.to()` copies onto it and it compares unequal to
# `CPU`, so the two-device paths run on a machine with one CUDA device or none.
REMOTE = torch.device("cpu", 0)


def placed(device: torch.device) -> torch.device:
    """The device a tensor placed on `device` reports back.

    Args:
        device: The requested device.

    Returns:
        The device of a tensor allocated there; CPU drops the index.
    """
    return torch.empty(0, device=device).device


def make_settings(**overrides: Any) -> OnlineTrainSettings:
    """Small but valid settings, built from the flat mapping an operator writes.

    Args:
        **overrides: Flat field names, exactly as they appear in the JSON config.

    Returns:
        The sectioned settings.
    """
    field_values = dict(
        enabled=True,
        anchors_per_sequence=4,
        sequences_per_step=2,
        min_rollout_tokens=8,
        max_rollout_tokens=128,
        buffer_capacity_tokens=256,
        kl_chunk_size=8,
    )
    field_values.update(overrides)
    return settings_factory.create(field_values)


def make_shapes(**overrides: Any) -> EngineShapes:
    """Shapes matching the toy head, bypassing engine resolution.

    Args:
        **overrides: Field names on `EngineShapes`.

    Returns:
        The shapes.
    """
    shape_fields = dict(
        aux_layer_ids=tuple(range(1, NUM_FEATURES + 1)),
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        block_size=BLOCK_SIZE,
        mask_token_id=0,
        feature_dtype=torch.float32,
    )
    shape_fields.update(overrides)
    return EngineShapes(**shape_fields)


def make_config(
    online_train_settings: OnlineTrainSettings | None = None,
    position_decay_gamma: float = 4.0,
    **shape_overrides: Any,
) -> ResolvedConfig:
    """A `ResolvedConfig` built directly.

    Args:
        online_train_settings: Operator settings. Defaults to `make_settings()`.
        position_decay_gamma: The resolved decay constant.
        **shape_overrides: Passed to `make_shapes`.

    Returns:
        The resolved config.
    """
    return ResolvedConfig(
        online_train_settings=online_train_settings or make_settings(),
        engine_shapes=make_shapes(**shape_overrides),
        position_decay_gamma=position_decay_gamma,
    )


def make_arch(**overrides: Any) -> DFlashHeadArch:
    """The toy head's shape.

    Args:
        **overrides: Field names on `DFlashHeadArch`.

    Returns:
        The arch.
    """
    arch_fields = dict(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_heads=4,
        num_kv_heads=2,
        head_dim=HIDDEN // 4,
        num_features=NUM_FEATURES,
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB,
        draft_vocab_size=VOCAB,
        mask_token_id=0,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        attention_bias=False,
    )
    arch_fields.update(overrides)
    return DFlashHeadArch(**arch_fields)


def make_buffer(resolved_config: ResolvedConfig) -> RolloutBuffer:
    """A pool wired the way the composition root wires it.

    Args:
        resolved_config: Supplies the buffer settings, the shapes and the seed.

    Returns:
        The pool.
    """
    online_train_settings = resolved_config.online_train_settings
    pool_sampler = PoolSampler(
        online_train_settings.buffer,
        random.Random(online_train_settings.bookkeeping.seed),
    )
    return RolloutBuffer(
        online_train_settings.buffer,
        resolved_config.engine_shapes,
        pool_sampler,
        BufferStats(),
    )


def make_collator(
    resolved_config: ResolvedConfig, seed: int = 0, device: torch.device = CPU
) -> RolloutCollator:
    """A collator wired the way the composition root wires it.

    Args:
        resolved_config: Supplies the anchor settings and the shapes.
        seed: Seeds the anchor subsample.
        device: Where the collated batch is sent.

    Returns:
        The collator.
    """
    return RolloutCollator(
        block_builder,
        AnchorSampler(
            resolved_config.online_train_settings.anchors, random.Random(seed)
        ),
        DeviceTransfer(device),
        resolved_config.engine_shapes,
    )


def make_objective(
    resolved_config: ResolvedConfig,
    draft_head: Any,
    teacher_project: Any,
) -> SFTObjective:
    """An objective wired the way the composition root wires it.

    Args:
        resolved_config: Supplies the objective settings and the decay constant.
        draft_head: Supplies the student projection.
        teacher_project: `[N, H] -> [N, vocab]`.

    Returns:
        The objective.
    """
    return SFTObjective(
        resolved_config.online_train_settings.objective,
        kl_divergence,
        teacher_scorer,
        PositionDecay(resolved_config.position_decay_gamma),
        draft_head.project_to_vocab,
        teacher_project,
    )


def make_trainer(
    resolved_config: ResolvedConfig,
    draft_head: Any,
    sft_objective: SFTObjective,
    rollout_collator: RolloutCollator | None = None,
) -> OnlineTrainer:
    """A trainer wired the way the composition root wires it.

    Args:
        resolved_config: Supplies the optimizer settings and the batch size.
        draft_head: The head to train.
        sft_objective: Scores a replayed batch.
        rollout_collator: Defaults to `make_collator(resolved_config)`.

    Returns:
        The trainer.
    """
    online_train_settings = resolved_config.online_train_settings
    optimizer = torch.optim.AdamW(
        draft_head.trainable_parameters(),
        lr=online_train_settings.optim.learning_rate,
        weight_decay=online_train_settings.optim.weight_decay,
    )
    return OnlineTrainer(
        online_train_settings.optim,
        draft_head,
        head_weights,
        sft_objective,
        rollout_collator or make_collator(resolved_config),
        schedule_factory,
        optimizer,
        online_train_settings.anchors.sequences_per_step,
    )


def make_record(
    num_tokens: int = 40,
    prompt_len: int = 10,
    seed: int = 0,
    hidden_size: int = HIDDEN,
) -> RolloutRecord:
    """A random rollout of the toy shape.

    Args:
        num_tokens: Positions the record covers.
        prompt_len: Tokens belonging to the prompt.
        seed: Seeds the random tensors.
        hidden_size: Hidden width.

    Returns:
        The record.
    """
    generator = torch.Generator().manual_seed(seed)
    return RolloutRecord(
        token_ids=torch.randint(
            0, VOCAB, (num_tokens,), generator=generator, dtype=torch.int32
        ),
        features=torch.randn(
            num_tokens, NUM_FEATURES * hidden_size, generator=generator
        ),
        final_hidden=torch.randn(num_tokens, hidden_size, generator=generator),
        prompt_len=prompt_len,
    )


def fake_vllm_config(
    *,
    num_speculative_tokens: int = BLOCK_SIZE - 1,
    target_layer_ids: list[int] | None = None,
    hidden_size: int = HIDDEN,
    num_hidden_layers: int = NUM_LAYERS,
    method: str = "dflash",
    online_train_settings: OnlineTrainSettings | None = None,
    **dflash_extra: Any,
) -> SimpleNamespace:
    """A stand-in exposing only what the resolver reads.

    Args:
        num_speculative_tokens: `K - 1`.
        target_layer_ids: DFlash-numbered feature layers.
        hidden_size: Target hidden width.
        num_hidden_layers: Draft layer count.
        method: The speculative method the engine reports.
        online_train_settings: Hung off the config as `online_train_config`.
        **dflash_extra: Merged into the `dflash_config` block.

    Returns:
        The stand-in.
    """
    # `is None` rather than falsy: an explicit empty list is a distinct case the
    # resolver must reject.
    if target_layer_ids is None:
        target_layer_ids = list(range(NUM_FEATURES))
    dflash_config = {"target_layer_ids": target_layer_ids}
    dflash_config.update(dflash_extra)
    hf_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=2 * hidden_size,
        dflash_config=dflash_config,
        vocab_size=VOCAB,
    )
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            use_dflash=lambda: method == "dflash",
            num_speculative_tokens=num_speculative_tokens,
            draft_model_config=SimpleNamespace(hf_config=hf_config),
            model="/nonexistent/head",
        ),
        model_config=SimpleNamespace(
            get_hidden_size=lambda: hidden_size,
            get_vocab_size=lambda: VOCAB,
            dtype=torch.float32,
        ),
        online_train_config=online_train_settings,
    )


class FakeTargetBody(nn.Module):
    """The decoder stack, holding the embedding table where vLLM holds it."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        """
        Args:
            vocab_size: Vocabulary size.
            hidden_size: Hidden width.
        """
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)


class FakeTarget(nn.Module):
    """A stand-in target laid out the way vLLM lays out a causal LM."""

    def __init__(self, vocab_size: int = VOCAB, hidden_size: int = HIDDEN) -> None:
        """
        Args:
            vocab_size: Vocabulary size.
            hidden_size: Hidden width.
        """
        super().__init__()
        self.model = FakeTargetBody(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.compute_logits_calls = 0

    @property
    def embed(self) -> nn.Embedding:
        """The embedding table."""
        return self.model.embed_tokens

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project through the vocabulary head.

        Args:
            hidden_states: `[N, H]`.

        Returns:
            `[N, vocab]`.
        """
        self.compute_logits_calls += 1
        return self.lm_head(hidden_states)


class FakeMultiModalTarget(nn.Module):
    """A stand-in target laid out the way vLLM lays out a multimodal model."""

    def __init__(self, vocab_size: int = VOCAB, hidden_size: int = HIDDEN) -> None:
        """
        Args:
            vocab_size: Vocabulary size.
            hidden_size: Hidden width.
        """
        super().__init__()
        self.language_model = FakeTarget(vocab_size, hidden_size)

    def get_language_model(self) -> FakeTarget:
        """The text stack this delegates generation to."""
        return self.language_model

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project through the language model's vocabulary head.

        Args:
            hidden_states: `[N, H]`.

        Returns:
            `[N, vocab]`.
        """
        return self.language_model.compute_logits(hidden_states)


@pytest.fixture
def online_train_settings() -> OnlineTrainSettings:
    return make_settings()


@pytest.fixture
def resolved_config() -> ResolvedConfig:
    return make_config()


@pytest.fixture
def head_arch() -> DFlashHeadArch:
    return make_arch()


@pytest.fixture
def draft_head(head_arch):
    torch.manual_seed(0)
    return head_factory.create(head_arch)


@pytest.fixture
def rollout_records() -> list[RolloutRecord]:
    return [make_record(40, 10, seed=1), make_record(36, 8, seed=2)]


@pytest.fixture
def rollout_collator(resolved_config) -> RolloutCollator:
    return make_collator(resolved_config)


@pytest.fixture
def replay_batch(rollout_collator, rollout_records):
    return rollout_collator.collate(rollout_records)


@pytest.fixture
def teacher_projection(draft_head):
    """A frozen stand-in for the target's LM head, shared with the student.

    Sharing the projection means a perfectly trained head reaches zero loss, which is
    what makes an overfit test meaningful.
    """
    projection = nn.Linear(HIDDEN, VOCAB, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.randn(VOCAB, HIDDEN) * 0.1)
    projection.requires_grad_(False)
    head_weights.load_shared(
        draft_head, torch.randn(VOCAB, HIDDEN) * 0.05, projection.weight
    )
    return projection


@pytest.fixture
def sft_objective(resolved_config, draft_head, teacher_projection) -> SFTObjective:
    return make_objective(resolved_config, draft_head, teacher_projection)
