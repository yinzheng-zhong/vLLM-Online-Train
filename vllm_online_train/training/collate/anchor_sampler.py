import random

from vllm.logger import init_logger

from vllm_online_train.config.settings import AnchorSettings
from vllm_online_train.training.collate.replay_batch import BlockRecord

logger = init_logger(__name__)


class AnchorSampler:
    def __init__(self, settings: AnchorSettings, rng: random.Random) -> None:
        """Cuts a rollout's candidate blocks down to the per-sequence anchor cap.

        Args:
            settings: Anchor cap, stride and subsample mode.
            rng: Draws the subsample.
        """
        self.settings = settings
        self.rng = rng
        logger.debug(
            "anchor sampler: cap=%d stride=%d random_subsample=%s",
            settings.anchors_per_sequence,
            settings.anchor_stride,
            settings.random_anchor_subsample,
        )

    @property
    def stride(self) -> int:
        """Spacing between candidate anchors."""
        return self.settings.anchor_stride

    @property
    def sequences_per_step(self) -> int:
        """Rollouts drawn from the buffer per training batch."""
        return self.settings.sequences_per_step

    def select(self, blocks: list[BlockRecord]) -> list[BlockRecord]:
        """Keep at most `anchors_per_sequence` blocks, in anchor order.

        Args:
            blocks: Every candidate block for one rollout.

        Returns:
            The kept blocks, sorted by anchor. Returns the input when it is already
            within the cap.
        """
        cap = self.settings.anchors_per_sequence
        if len(blocks) <= cap:
            return blocks
        kept = (
            self.rng.sample(blocks, cap)
            if self.settings.random_anchor_subsample
            else blocks[:cap]
        )
        kept.sort(key=lambda block: block.anchor)
        return kept
