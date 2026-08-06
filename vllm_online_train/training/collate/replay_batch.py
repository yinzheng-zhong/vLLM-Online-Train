from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class BlockRecord:
    """One drafted block: the state the head acted in, and what it should emit."""

    anchor: int
    """Absolute index of the last known token before the block. The block covers
    positions `anchor + 1 .. anchor + num_draft_tokens`."""

    draft_token_ids: list[int]
    """`num_draft_tokens` labels: the tokens that occupy those positions."""

    accepted: int
    """`r`: how many leading labels the head got right."""


@dataclass(slots=True)
class ReplayBatch:
    """A padded batch of rollouts prepared for anchored replay.

    The context is kept once and every block is appended after it, so a single forward
    scores every anchor:

    ```
    [ context tokens (T) ][ block_0 (K) ][ block_1 (K) ] ... [ block_{A-1} (K) ]
    ```
    """

    input_ids: torch.Tensor
    """`[B, T]` context tokens, right-padded."""

    attention_mask: torch.Tensor
    """`[B, T]` 1 for real context tokens."""

    prompt_lengths: torch.Tensor
    """`[B]`."""

    anchor_positions: torch.Tensor
    """`[B, A]` absolute anchor index per block, clamped to `>= 0`."""

    block_keep_mask: torch.Tensor
    """`[B, A]` bool; False marks padding blocks."""

    block_token_ids: torch.Tensor
    """`[B, A, D]` labels per block, where `D = block_size - 1`."""

    accepted_counts: torch.Tensor
    """`[B, A]`."""

    target_features: torch.Tensor
    """`[B, T, num_features * hidden_size]` captured target features, concatenated in
    aux-layer order."""

    final_hidden: torch.Tensor
    """`[B, T, hidden_size]` captured post-norm final activations."""

    meta: dict[str, object] = field(default_factory=dict)

    @property
    def num_draft_tokens(self) -> int:
        """`D`: scored positions per block."""
        return int(self.block_token_ids.shape[-1])

    @property
    def num_anchors(self) -> int:
        """`A`: blocks per row, padding included."""
        return int(self.anchor_positions.shape[1])

    def acceptance_index_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The `I_acc` / `I_rej` masks.

        Returns:
            Two `[B, A, D]` bool tensors. Position `k` is accepted when `k < r` and
            rejected otherwise; padding blocks appear in neither.
        """
        k_index = torch.arange(
            self.num_draft_tokens, device=self.accepted_counts.device
        ).view(1, 1, -1)
        counts = self.accepted_counts.unsqueeze(-1)
        keep = self.block_keep_mask.unsqueeze(-1)
        return (k_index < counts) & keep, (k_index >= counts) & keep

    def block_offsets(self) -> torch.Tensor:
        """`[B, A, D]` 1-based position of each label inside its block."""
        k_index = torch.arange(
            self.num_draft_tokens, device=self.accepted_counts.device
        )
        return (k_index + 1).view(1, 1, -1).expand_as(self.block_token_ids)
