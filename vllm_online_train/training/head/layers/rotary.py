import torch
from torch import nn

from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        """Neox-style RoPE evaluated at explicit positions.

        Positions are explicit because a replayed block sits at absolute positions
        `a_m .. a_m + K - 1` while occupying contiguous slots in the packed sequence.

        Args:
            head_dim: Width of one attention head.
            base: RoPE theta.
        """
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        logger.debug(
            "built RoPE inv_freq buffer: head_dim=%d, base=%.1f, size=%d",
            head_dim,
            base,
            inv_freq.numel(),
        )

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the cosine and sine tables for a set of positions.

        Args:
            positions: `[B, T]` absolute positions.

        Returns:
            Two `[B, T, head_dim]` tables.
        """
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        angles = torch.cat([freqs, freqs], dim=-1)
        return angles.cos(), angles.sin()

    @staticmethod
    def rotate_half(projected_heads: torch.Tensor) -> torch.Tensor:
        """Swap the halves of the last dimension, negating the second.

        Args:
            projected_heads: `[..., head_dim]`.

        Returns:
            `[-second_half, first_half]` concatenated.
        """
        half = projected_heads.shape[-1] // 2
        return torch.cat(
            [-projected_heads[..., half:], projected_heads[..., :half]], dim=-1
        )

    @classmethod
    def rotate(
        cls,
        projected_heads: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the rotation to projected heads.

        Named `rotate` rather than `apply`, which would shadow `nn.Module.apply` and
        silently stop any `parent.apply(fn)` recursion at this node.

        Args:
            projected_heads: `[B, heads, T, head_dim]`.
            cos_table: `[B, T, head_dim]` cosine table.
            sin_table: `[B, T, head_dim]` sine table.

        Returns:
            `[B, heads, T, head_dim]` rotated.
        """
        cos_table = cos_table.unsqueeze(1).to(projected_heads.dtype)
        sin_table = sin_table.unsqueeze(1).to(projected_heads.dtype)
        return (
            projected_heads * cos_table + cls.rotate_half(projected_heads) * sin_table
        )
