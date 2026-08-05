import random

from vllm_online_train.capture.records import RolloutRecord
from vllm_online_train.config.settings import BufferSettings


class PoolSampler:
    """Draws a training batch from the pool without consuming it."""

    def __init__(self, settings: BufferSettings, rng: random.Random) -> None:
        """
        Args:
            settings: Supplies the draw strategy.
            rng: Draws the reservoir sample.
        """
        self.settings = settings
        self.rng = rng

    def draw(self, pool: list[RolloutRecord], count: int) -> list[RolloutRecord]:
        """Take `count` rollouts, or the whole pool when it is smaller.

        Args:
            pool: The admitted rollouts, oldest first.
            count: How many to draw.

        Returns:
            The drawn rollouts. Empty when the pool is.
        """
        if not pool:
            return []
        count = min(count, len(pool))
        if self.settings.sample_strategy == "fifo":
            return pool[:count]
        return self.rng.sample(pool, count)
