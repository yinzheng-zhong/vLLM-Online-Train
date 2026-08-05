import random

from vllm_online_train.collate.batch import BlockRecord
from vllm_online_train.config.settings import AnchorSettings


class AnchorSampler:
    """Cuts a rollout's candidate blocks down to the per-sequence anchor cap."""

    def __init__(self, settings: AnchorSettings, rng: random.Random) -> None:
        """
        Args:
            settings: Anchor cap, stride and subsample mode.
            rng: Draws the subsample.
        """
        self.settings = settings
        self.rng = rng

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
