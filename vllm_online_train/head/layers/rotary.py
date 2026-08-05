import torch
from torch import nn
from vllm.logger import init_logger

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
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Swap the halves of the last dimension, negating the second.

        Args:
            x: `[..., head_dim]`.

        Returns:
            `[-x_second_half, x_first_half]` concatenated.
        """
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    @classmethod
    def rotate(
        cls, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply the rotation to projected heads.

        Named `rotate` rather than `apply`, which would shadow `nn.Module.apply` and
        silently stop any `parent.apply(fn)` recursion at this node.

        Args:
            x: `[B, heads, T, head_dim]`.
            cos: `[B, T, head_dim]` cosine table.
            sin: `[B, T, head_dim]` sine table.

        Returns:
            `[B, heads, T, head_dim]` rotated.
        """
        cos = cos.unsqueeze(1).to(x.dtype)
        sin = sin.unsqueeze(1).to(x.dtype)
        return x * cos + cls.rotate_half(x) * sin
