import math

from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class CosineWithWarmup:
    def __init__(self, total_steps: int, warmup_ratio: float) -> None:
        """Learning-rate multiplier: linear warmup, then cosine decay to zero.

        Args:
            total_steps: The horizon the cosine decays towards.
            warmup_ratio: Fraction of `total_steps` spent warming up.
        """
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = max(1, round(total_steps * warmup_ratio))
        logger.debug(
            "CosineWithWarmup: total_steps=%d, warmup_steps=%d",
            total_steps,
            self.warmup_steps,
        )

    def __call__(self, optimizer_step: int) -> float:
        """Compute the multiplier for one step.

        Args:
            optimizer_step: Optimizer step, zero-based.

        Returns:
            `(optimizer_step + 1) / warmup_steps` during warmup, then the cosine
            from 1.0 to 0.0 over the remaining steps.
        """
        if optimizer_step < self.warmup_steps:
            return (optimizer_step + 1) / self.warmup_steps
        progress = (optimizer_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
