import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS norm accumulated in fp32, matching vLLM's `RMSNorm` numerics."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        """
        Args:
            hidden_size: Width to normalise over.
            eps: Added to the variance before the reciprocal square root.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension and scale by the weight.

        Args:
            x: `[..., hidden_size]`.

        Returns:
            The normalised tensor, back in `x`'s dtype.
        """
        dtype = x.dtype
        x32 = x.float()
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.variance_epsilon)
        return (x32 * self.weight.float()).to(dtype)
