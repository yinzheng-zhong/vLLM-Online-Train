import torch

from vllm_online_train.contracts.positions import BlockPositions
from vllm_online_train.training.collate.replay_batch import ReplayBatch


class TeacherScorer:
    def __init__(self, block_positions: BlockPositions) -> None:
        """Gathers the target activations that score each label.

        Args:
            block_positions: Maps anchors to the positions their block slots occupy.
        """
        self.block_positions = block_positions

    def hidden_for_labels(
        self, replay_batch: ReplayBatch, num_draft_tokens: int
    ) -> torch.Tensor:
        """The target hidden states that produce each label's teacher distribution.

        The distribution over the label at position `a + j` is the target's output at
        `a + j - 1`, so block `m` reads positions `a_m .. a_m + D - 1`.

        Args:
            replay_batch: The batch being scored.
            num_draft_tokens: `D`, labels per block.

        Returns:
            `[B, A, D, H]`.
        """
        final_hidden = replay_batch.final_hidden
        batch_size, context_len, hidden_size = final_hidden.shape
        slot_position_ids = self.block_positions.block_position_ids(
            replay_batch.anchor_positions.clamp_min(0), num_draft_tokens
        ).clamp_max(context_len - 1)
        gathered = final_hidden.gather(
            1, slot_position_ids.unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        return gathered.view(batch_size, -1, num_draft_tokens, hidden_size)
