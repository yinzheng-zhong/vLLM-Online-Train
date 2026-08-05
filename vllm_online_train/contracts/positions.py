from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class BlockPositions(Protocol):
    """Maps anchors to the absolute positions their block slots occupy."""

    def block_position_ids(
        self, anchor_positions: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Position ids for every slot of every block.

        Args:
            anchor_positions: `[B, A]` absolute anchor index per block.
            block_size: Slots per block.

        Returns:
            `[B, A * block_size]` running `a_m, a_m+1, ..., a_m+block_size-1`.
        """
        ...
