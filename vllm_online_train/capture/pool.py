import random

from vllm.logger import init_logger

from vllm_online_train.capture.records import RolloutRecord
from vllm_online_train.config.settings import BufferSettings

logger = init_logger(__name__)


class PoolSampler:
    def __init__(self, settings: BufferSettings, rng: random.Random) -> None:
        """Draws a training batch from the pool without consuming it.

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
        strategy = self.settings.sample_strategy
        logger.debug("drew %d/%d rollouts via %s", count, len(pool), strategy)
        if strategy == "fifo":
            return pool[:count]
        return self.rng.sample(pool, count)
