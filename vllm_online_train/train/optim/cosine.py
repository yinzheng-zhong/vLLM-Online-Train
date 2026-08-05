import math


class CosineWithWarmup:
    """Learning-rate multiplier: linear warmup, then cosine decay to zero."""

    def __init__(self, total_steps: int, warmup_ratio: float) -> None:
        """
        Args:
            total_steps: The horizon the cosine decays towards.
            warmup_ratio: Fraction of `total_steps` spent warming up.
        """
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = max(1, round(total_steps * warmup_ratio))

    def __call__(self, step: int) -> float:
        """Compute the multiplier for one step.

        Args:
            step: Optimizer step, zero-based.

        Returns:
            `(step + 1) / warmup_steps` during warmup, then the cosine from 1.0 to
            0.0 over the remaining steps.
        """
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
