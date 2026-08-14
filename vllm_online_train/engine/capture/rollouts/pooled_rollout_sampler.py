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

    def eligible(self, pooled_rollouts: list[RolloutRecord]) -> list[RolloutRecord]:
        """The pooled rollouts still under the draw cap, oldest first.

        Args:
            pooled_rollouts: The admitted rollouts, oldest first.

        Returns:
            Every record when the cap is 0, otherwise those drawn fewer than
            `max_draws_per_rollout` times.
        """
        max_draws = self.buffer_settings.max_draws_per_rollout
        if not max_draws:
            return pooled_rollouts
        return [record for record in pooled_rollouts if record.draws < max_draws]

    def draw(
        self, pooled_rollouts: list[RolloutRecord], count: int
    ) -> list[RolloutRecord]:
        """Take `count` eligible rollouts, or all of them when there are fewer.

        Every drawn record has its `draws` counter incremented, which is what retires it
        once it reaches `max_draws_per_rollout`.

        Args:
            pooled_rollouts: The admitted rollouts, oldest first.
            count: How many to draw.

        Returns:
            The drawn rollouts. Empty when nothing is eligible.
        """
        eligible_rollouts = self.eligible(pooled_rollouts)
        if not eligible_rollouts:
            return []
        count = min(count, len(eligible_rollouts))
        sample_strategy = self.buffer_settings.sample_strategy
        logger.debug(
            "drew %d/%d eligible rollouts (%d pooled) via %s",
            count,
            len(eligible_rollouts),
            len(pooled_rollouts),
            sample_strategy,
        )
        if sample_strategy == "fifo":
            drawn_records = eligible_rollouts[:count]
        else:
            drawn_records = self.rng.sample(eligible_rollouts, count)
        for record in drawn_records:
            record.draws += 1
        return drawn_records
