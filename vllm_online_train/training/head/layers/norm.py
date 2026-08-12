import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        """RMS norm accumulated in fp32, matching vLLM's `RMSNorm` numerics.

        Args:
            hidden_size: Width to normalise over.
            eps: Added to the variance before the reciprocal square root.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension and scale by the weight.

        Args:
            hidden_states: `[..., hidden_size]`.

        Returns:
            The normalised tensor, back in the input's dtype.
        """
        input_dtype = hidden_states.dtype
        hidden_fp32 = hidden_states.float()
        variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
        hidden_fp32 = hidden_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_fp32 * self.weight.float()).to(input_dtype)
