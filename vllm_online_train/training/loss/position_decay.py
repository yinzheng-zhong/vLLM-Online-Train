import torch

from vllm_online_train.training.collate.replay_batch import ReplayBatch


class PositionDecay:
    def __init__(self, gamma: float) -> None:
        """In-block position weighting, `w_k = exp(-(k-1)/gamma)`.

        Args:
            gamma: Decay constant. 0 selects a uniform mean.
        """
        self.gamma = gamma

    def weights(self, replay_batch: ReplayBatch) -> torch.Tensor | None:
        """Weight every scored position by its offset inside its block.

        Args:
            replay_batch: The batch being scored.

        Returns:
            `[B, A, D]` weights, or `None` for the uniform mean.
        """
        if not self.gamma:
            return None
        block_offsets = replay_batch.block_offsets().float()
        return torch.exp(-(block_offsets - 1.0).clamp_min(0.0) / self.gamma)
