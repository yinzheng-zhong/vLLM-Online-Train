import random

from vllm_online_train.config.settings import AnchorSettings
from vllm_online_train.logger import init_logger
from vllm_online_train.training.collate.replay_batch import BlockRecord

logger = init_logger(__name__)


class AnchorSampler:
    def __init__(self, anchor_settings: AnchorSettings, rng: random.Random) -> None:
        """Cuts a rollout's candidate blocks down to the per-sequence anchor cap.

        Args:
            anchor_settings: Anchor cap, stride and subsample mode.
            rng: Draws the subsample.
        """
        self.anchor_settings = anchor_settings
        self.rng = rng
        logger.debug(
            "anchor sampler: cap=%d stride=%d random_subsample=%s",
            anchor_settings.anchors_per_sequence,
            anchor_settings.anchor_stride,
            anchor_settings.random_anchor_subsample,
        )

    @property
    def stride(self) -> int:
        """Spacing between candidate anchors."""
        return self.anchor_settings.anchor_stride

    @property
    def sequences_per_step(self) -> int:
        """Rollouts drawn from the buffer per training batch."""
        return self.anchor_settings.sequences_per_step

    def select(self, block_records: list[BlockRecord]) -> list[BlockRecord]:
        """Keep at most `anchors_per_sequence` blocks, in anchor order.

        Args:
            block_records: Every candidate block for one rollout.

        Returns:
            The kept blocks, sorted by anchor. Returns the input when it is already
            within the cap.
        """
        cap = self.anchor_settings.anchors_per_sequence
        if len(block_records) <= cap:
            return block_records
        kept_records = (
            self.rng.sample(block_records, cap)
            if self.anchor_settings.random_anchor_subsample
            else block_records[:cap]
        )
        kept_records.sort(key=lambda block_record: block_record.anchor)
        return kept_records
