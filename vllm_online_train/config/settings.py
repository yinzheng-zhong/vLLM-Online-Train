from dataclasses import dataclass, field
from typing import Literal

from vllm.config.utils import config

KLDirection = Literal["forward", "reverse"]
SampleStrategy = Literal["fifo", "reservoir"]
SharedDtype = Literal["model", "float32"]
FeatureDtype = Literal["bfloat16", "float16", "float32"]
PublishMode = Literal["off", "hot"]


@config(frozen=True)
class ObjectiveSettings:
    """Loss settings: KL direction, in-block weighting, auxiliary CE, chunking."""

    kl_direction: KLDirection = "forward"
    """`"forward"` is KL(teacher || student); `"reverse"` is KL(student || teacher)."""

    position_decay_gamma: float | None = None
    """`w_k = exp(-(k-1)/gamma)` in-block weighting. `None` derives it from the block
    size; 0 selects a uniform mean."""

    label_smoothing_ce_weight: float = 0.0
    """Weight on an auxiliary cross-entropy against the emitted token."""

    kl_chunk_size: int = 512
    """Rows scored per KL chunk."""

    def __post_init__(self) -> None:
        """Reject values the loss cannot be computed with.

        Raises:
            ValueError: If `kl_chunk_size` is below 1 or `position_decay_gamma` is
                negative.
        """
        if self.kl_chunk_size < 1:
            raise ValueError(f"kl_chunk_size must be >= 1, got {self.kl_chunk_size}")
        if self.position_decay_gamma is not None and self.position_decay_gamma < 0:
            raise ValueError(
                "position_decay_gamma must be non-negative (0 for a uniform mean), "
                f"got {self.position_decay_gamma}"
            )

    @staticmethod
    def default_gamma(block_size: int) -> float:
        """Decay constant for a block size, when none is set.

        Args:
            block_size: `K`, query slots per block.

        Returns:
            7.0 at 16, 5.0 at 10 and 4.0 at 8; `0.375 * block_size + 1.0` elsewhere.
        """
        exact = {16: 7.0, 10: 5.0, 8: 4.0}
        if block_size in exact:
            return exact[block_size]
        return 0.375 * block_size + 1.0


@config(frozen=True)
class OptimSettings:
    """Optimizer and learning-rate schedule settings."""

    learning_rate: float = 6e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0

    total_steps: int | None = None
    """Cosine horizon. `None` selects linear warmup followed by a flat rate."""

    grad_accum_steps: int = 1
    """Micro-batches accumulated per optimizer step."""

    def __post_init__(self) -> None:
        """Reject values the optimizer loop cannot run with.

        Raises:
            ValueError: If `grad_accum_steps` is below 1, `warmup_ratio` is outside
                [0, 1], or `total_steps` is set below 1.
        """
        if self.grad_accum_steps < 1:
            raise ValueError(
                f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}"
            )
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(
                f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}"
            )
        if self.total_steps is not None and self.total_steps < 1:
            raise ValueError(
                f"total_steps must be >= 1 when set, got {self.total_steps}"
            )


@config(frozen=True)
class AnchorSettings:
    """How many anchors a rollout contributes and how they are chosen."""

    anchors_per_sequence: int = 16
    """Anchors kept per rollout."""

    anchor_stride: int = 1
    """Spacing of candidate anchors over the response."""

    random_anchor_subsample: bool = True
    """Sample the anchor cap at random rather than taking the first N."""

    sequences_per_step: int = 4
    """Rollouts drawn from the buffer per training batch."""

    def __post_init__(self) -> None:
        """Reject values that would yield no anchors.

        Raises:
            ValueError: If any count is below 1.
        """
        if self.anchors_per_sequence < 1:
            raise ValueError(
                f"anchors_per_sequence must be >= 1, got {self.anchors_per_sequence}"
            )
        if self.anchor_stride < 1:
            raise ValueError(f"anchor_stride must be >= 1, got {self.anchor_stride}")
        if self.sequences_per_step < 1:
            raise ValueError(
                f"sequences_per_step must be >= 1, got {self.sequences_per_step}"
            )


@config(frozen=True)
class BufferSettings:
    """Rollout pool capacity, length bounds and draw strategy."""

    buffer_capacity_tokens: int = 200_000
    """Pool capacity in tokens."""

    max_rollout_tokens: int = 2048
    """Rollouts longer than this are dropped. Also bounds the staging buffer."""

    min_rollout_tokens: int = 64
    """Rollouts shorter than this are dropped."""

    sample_strategy: SampleStrategy = "reservoir"
    """`"reservoir"` draws uniformly over the pool; `"fifo"` takes the oldest."""

    max_draws_per_rollout: int = 0
    """Training batches a single rollout may appear in before it stops being drawn;
    0 draws it for as long as it stays pooled."""

    def __post_init__(self) -> None:
        """Reject capacities that cannot hold what they admit.

        Raises:
            ValueError: If the length bounds are inverted, the pool cannot hold one
                maximum-length rollout, or the draw cap is negative.
        """
        if self.max_draws_per_rollout < 0:
            raise ValueError(
                f"max_draws_per_rollout must be >= 0, got "
                f"{self.max_draws_per_rollout}"
            )
        if self.min_rollout_tokens > self.max_rollout_tokens:
            raise ValueError(
                f"min_rollout_tokens ({self.min_rollout_tokens}) exceeds "
                f"max_rollout_tokens ({self.max_rollout_tokens})"
            )
        if self.buffer_capacity_tokens < self.max_rollout_tokens:
            raise ValueError(
                f"buffer_capacity_tokens ({self.buffer_capacity_tokens}) cannot hold "
                f"a single max_rollout_tokens ({self.max_rollout_tokens}) rollout"
            )


@config(frozen=True)
class CaptureSettings:
    """Which requests are captured, and at what precision their features are held."""

    skip_prefix_cache_hits: bool = True
    """Drop requests that reused cached blocks."""

    capture_sample_rate: float = 1.0
    """Fraction of requests to capture, decided once per request from a hash of its
    id."""

    feature_dtype: FeatureDtype | None = None
    """Storage dtype for buffered features. `None` follows the model dtype."""

    shared_dtype: SharedDtype = "model"
    """Precision for the borrowed embedding table and LM head. `"model"` holds them at
    the serving dtype; `"float32"` matches an fp32 offline reference."""

    def __post_init__(self) -> None:
        """Reject a sample rate that captures nothing or more than everything.

        Raises:
            ValueError: If `capture_sample_rate` is outside (0, 1].
        """
        if not 0.0 < self.capture_sample_rate <= 1.0:
            raise ValueError(
                f"capture_sample_rate must be in (0, 1], got {self.capture_sample_rate}"
            )


@config(frozen=True)
class GateSettings:
    """When the idle gate opens and how long a single opening may hold the GPU."""

    idle_ms: float = 50.0
    """Quiet time after the last engine step before training may start."""

    busy_tokens: int = 0
    """Also require the last step to have scheduled no more than this many tokens.
    0 disables the check."""

    poll_ms: float = 10.0
    """Gate polling interval."""

    max_micro_steps_per_gate: int = 0
    """Micro-batches to run per gate opening; 0 means until the gate closes."""

    def __post_init__(self) -> None:
        """Reject timings the polling loop cannot use.

        Raises:
            ValueError: If any duration or bound is negative, or `poll_ms` is zero.
        """
        if self.idle_ms < 0:
            raise ValueError(f"idle_ms must be non-negative, got {self.idle_ms}")
        if self.poll_ms <= 0:
            raise ValueError(f"poll_ms must be positive, got {self.poll_ms}")
        if self.busy_tokens < 0:
            raise ValueError(f"busy_tokens must be >= 0, got {self.busy_tokens}")
        if self.max_micro_steps_per_gate < 0:
            raise ValueError(
                "max_micro_steps_per_gate must be >= 0 (0 for unlimited), got "
                f"{self.max_micro_steps_per_gate}"
            )

    @property
    def idle_seconds(self) -> float:
        """`idle_ms` in seconds."""
        return self.idle_ms / 1000.0

    @property
    def poll_seconds(self) -> float:
        """`poll_ms` in seconds."""
        return self.poll_ms / 1000.0


@config(frozen=True)
class PlacementSettings:
    """Which device the training copy of the head occupies."""

    train_device: str | None = None
    """Device the head, its gradients, its optimizer state and its batches live on,
    e.g. `"cuda:1"`. `None` trains on the engine's own device. A bare `"cuda"` also
    means the engine's device; only an explicit index moves the training off it."""


@config(frozen=True)
class PublishSettings:
    """Whether trained weights reach the live drafter without a restart."""

    publish_mode: PublishMode = "off"
    """`"off"` writes checkpoints only; `"hot"` also loads them into the drafter."""

    @property
    def hot(self) -> bool:
        """Whether a publish follows every checkpoint export."""
        return self.publish_mode == "hot"


@config(frozen=True)
class BookkeepingSettings:
    """Seed, logging cadence, checkpoint destination and the metrics sink."""

    seed: int = 42
    log_every: int = 10
    checkpoint_dir: str | None = None

    checkpoint_every: int = 0
    """Steps between checkpoint exports; 0 disables. Also the hot-publish cadence."""

    metrics_path: str | None = None
    """JSONL sink for per-step training metrics."""

    def __post_init__(self) -> None:
        """Reject negative cadences.

        Raises:
            ValueError: If `log_every` or `checkpoint_every` is negative.
        """
        if self.log_every < 0:
            raise ValueError(f"log_every must be >= 0, got {self.log_every}")
        if self.checkpoint_every < 0:
            raise ValueError(
                f"checkpoint_every must be >= 0, got {self.checkpoint_every}"
            )


@dataclass(frozen=True)
class OnlineTrainSettings:
    """Every section an operator can set, plus the master switch.

    Each section validates its own fields; this object validates only the rules that
    span two sections.
    """

    enabled: bool = False
    objective: ObjectiveSettings = field(default_factory=ObjectiveSettings)
    optim: OptimSettings = field(default_factory=OptimSettings)
    anchors: AnchorSettings = field(default_factory=AnchorSettings)
    buffer: BufferSettings = field(default_factory=BufferSettings)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    gate: GateSettings = field(default_factory=GateSettings)
    placement: PlacementSettings = field(default_factory=PlacementSettings)
    publish: PublishSettings = field(default_factory=PublishSettings)
    bookkeeping: BookkeepingSettings = field(default_factory=BookkeepingSettings)

    def __post_init__(self) -> None:
        """Check the rules that span two sections.

        Raises:
            ValueError: If a hot publish is configured without a checkpoint cadence.
        """
        if self.publish.hot and not self.bookkeeping.checkpoint_every:
            raise ValueError(
                "publish_mode='hot' needs checkpoint_every > 0; the checkpoint "
                "cadence is also the publish cadence, and a hot publish with no "
                "checkpoint beside it leaves nothing to restart from."
            )
