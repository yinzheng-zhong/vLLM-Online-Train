class WarmupOnly:
    """Learning-rate multiplier: linear warmup, then flat."""

    def __init__(self, warmup_steps: int) -> None:
        """
        Args:
            warmup_steps: Steps spent warming up.
        """
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        """Compute the multiplier for one step.

        Args:
            step: Optimizer step, zero-based.

        Returns:
            `(step + 1) / warmup_steps` during warmup, then 1.0.
        """
        if step >= self.warmup_steps:
            return 1.0
        return (step + 1) / max(1, self.warmup_steps)
