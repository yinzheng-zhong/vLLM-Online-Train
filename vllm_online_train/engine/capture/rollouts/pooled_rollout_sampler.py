import random

from vllm_online_train.config.settings import BufferSettings
from vllm_online_train.engine.capture.rollouts.rollout_records import RolloutRecord
from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class PooledRolloutSampler:
    def __init__(self, buffer_settings: BufferSettings, rng: random.Random) -> None:
        """Draws a training batch from the pool without consuming it.

        Args:
            buffer_settings: Supplies the draw strategy.
            rng: Draws the reservoir sample.
        """
        self.buffer_settings = buffer_settings
        self.rng = rng

    def draw(
        self, pooled_rollouts: list[RolloutRecord], count: int
    ) -> list[RolloutRecord]:
        """Take `count` rollouts, or the whole pool when it is smaller.

        Args:
            pooled_rollouts: The admitted rollouts, oldest first.
            count: How many to draw.

        Returns:
            The drawn rollouts. Empty when the pool is.
        """
        if not pooled_rollouts:
            return []
        count = min(count, len(pooled_rollouts))
        sample_strategy = self.buffer_settings.sample_strategy
        logger.debug(
            "drew %d/%d rollouts via %s",
            count,
            len(pooled_rollouts),
            sample_strategy,
        )
        if sample_strategy == "fifo":
            return pooled_rollouts[:count]
        return self.rng.sample(pooled_rollouts, count)
