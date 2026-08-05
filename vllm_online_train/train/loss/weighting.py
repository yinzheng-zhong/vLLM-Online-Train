import torch

from vllm_online_train.collate.batch import ReplayBatch


class PositionDecay:
    def __init__(self, gamma: float) -> None:
        """In-block position weighting, `w_k = exp(-(k-1)/gamma)`.

        Args:
            gamma: Decay constant. 0 selects a uniform mean.
        """
        self.gamma = gamma

    def weights(self, batch: ReplayBatch) -> torch.Tensor | None:
        """Weight every scored position by its offset inside its block.

        Args:
            batch: The batch being scored.

        Returns:
            `[B, A, D]` weights, or `None` for the uniform mean.
        """
        if not self.gamma:
            return None
        offsets = batch.block_offsets().float()
        return torch.exp(-(offsets - 1.0).clamp_min(0.0) / self.gamma)
