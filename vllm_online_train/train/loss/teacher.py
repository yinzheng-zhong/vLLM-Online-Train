import torch

from vllm_online_train.collate.batch import ReplayBatch
from vllm_online_train.contracts.positions import BlockPositions


class TeacherScorer:
    def __init__(self, positions: BlockPositions) -> None:
        """Gathers the target activations that score each label.

        Args:
            positions: Maps anchors to the positions their block slots occupy.
        """
        self.positions = positions

    def hidden_for_labels(
        self, batch: ReplayBatch, num_draft_tokens: int
    ) -> torch.Tensor:
        """The target hidden states that produce each label's teacher distribution.

        The distribution over the label at position `a + j` is the target's output at
        `a + j - 1`, so block `m` reads positions `a_m .. a_m + D - 1`.

        Args:
            batch: The batch being scored.
            num_draft_tokens: `D`, labels per block.

        Returns:
            `[B, A, D, H]`.
        """
        hidden = batch.final_hidden
        batch_size, context_len, hidden_size = hidden.shape
        positions = self.positions.block_position_ids(
            batch.anchor_positions.clamp_min(0), num_draft_tokens
        ).clamp_max(context_len - 1)
        gathered = hidden.gather(
            1, positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        return gathered.view(batch_size, -1, num_draft_tokens, hidden_size)
