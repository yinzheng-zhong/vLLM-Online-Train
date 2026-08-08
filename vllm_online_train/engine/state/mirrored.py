import torch
import torch.nn.functional as F

from vllm_online_train.contracts.provider import StateProvider
from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class MirroredStateProvider:
    PARITY_TOLERANCE: float = 0.02
    """Largest deviation from the engine's own projection that `verify_parity` accepts,
    as a fraction of the engine's largest logit."""

    def __init__(
        self, source: StateProvider, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Serves the borrowed target state from a device other than the engine's.

        Holds its own copy of the target's vocabulary projection on the training device
        and scores the teacher there, so no part of a training step runs on the serving
        device. The wrapped provider stays the only holder of the target handle.

        Args:
            source: Holds the target handle, on the engine's device.
            device: Where the head trains.
            dtype: Precision the mirrored vocabulary projection is held at.
        """
        self._source = source
        self._device = device
        self._projection = source.lm_head_weight(device).to(dtype)
        logger.debug(
            "mirrored state provider built on %s at %s", device, self._projection.dtype
        )

    @property
    def device(self) -> torch.device:
        """Where the state this serves lives, i.e. the training device."""
        return self._device

    @property
    def target_dtype(self) -> torch.dtype:
        """The target's parameter dtype."""
        return self._source.target_dtype

    @property
    def serving_dtype(self) -> torch.dtype:
        """The dtype the engine is configured to serve at."""
        return self._source.serving_dtype

    @property
    def projection_dtype(self) -> torch.dtype:
        """The precision the mirrored vocabulary projection is held at."""
        return self._projection.dtype

    def embedding_weight(self, device: torch.device | None = None) -> torch.Tensor:
        """The target's input embedding table, detached in fp32.

        Args:
            device: Where to place the copy. `None` selects the training device.

        Returns:
            `[vocab, hidden]` fp32.
        """
        return self._source.embedding_weight(device or self._device)

    def lm_head_weight(self, device: torch.device | None = None) -> torch.Tensor:
        """The target's vocabulary projection, detached in fp32.

        Args:
            device: Where to place the copy. `None` selects the training device.

        Returns:
            `[vocab, hidden]` fp32.
        """
        return self._source.lm_head_weight(device or self._device)

    def teacher_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states through the mirrored copy of the target's head.

        Args:
            hidden: `[N, hidden_size]` post-norm final hidden states, on the training
                device.

        Returns:
            `[N, vocab_size]` fp32 logits.
        """
        return F.linear(hidden.to(self._projection.dtype), self._projection).float()

    def verify_parity(self, rows: int = 4) -> float:
        """Score the same states through both projections and compare.

        Args:
            rows: Hidden states to probe with.

        Returns:
            The largest deviation, as a fraction of the engine's largest logit.

        Raises:
            ValueError: If the two projections disagree in shape, or by more than
                `PARITY_TOLERANCE`.
        """
        generator = torch.Generator().manual_seed(0)
        hidden = torch.randn(
            rows,
            self._projection.shape[1],
            generator=generator,
            dtype=torch.float32,
        )
        with torch.no_grad():
            engine = self._source.teacher_logits(hidden.to(self._source.device))
            mirrored = self.teacher_logits(hidden.to(self._device)).to(engine.device)
            deviation = self._deviation(mirrored, engine)

        if deviation > self.PARITY_TOLERANCE:
            raise ValueError(
                f"the mirrored teacher projection deviates from the engine's own by "
                f"{deviation:.3g} of its largest logit, above the "
                f"{self.PARITY_TOLERANCE:.3g} this accepts. A plain matmul against the "
                "LM head is not this target's projection: a logit scale, a soft cap or "
                "a quantised head would each do this. Leave train_device unset to "
                "score the teacher through the engine's own projection."
            )
        return deviation

    @staticmethod
    def _deviation(mirrored: torch.Tensor, engine: torch.Tensor) -> float:
        """Compare two sets of logits for the same states.

        Args:
            mirrored: `[N, vocab]` from the mirrored projection.
            engine: `[N, vocab]` from the engine's own projection.

        Returns:
            The largest absolute difference, as a fraction of the engine's largest
            logit.

        Raises:
            ValueError: If the two disagree in shape.
        """
        if mirrored.shape != engine.shape:
            raise ValueError(
                f"the mirrored projection yields {tuple(mirrored.shape)} logits where "
                f"the engine's own yields {tuple(engine.shape)}. Leave train_device "
                "unset to score the teacher through the engine's projection instead."
            )

        scale = engine.abs().amax().clamp_min(1.0)
        return float((mirrored - engine).abs().amax() / scale)
