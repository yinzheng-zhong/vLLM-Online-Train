import torch
import torch.nn.functional as F
from torch import nn

from vllm_online_train.head.arch import DFlashHeadArch


class DFlashMLP(nn.Module):
    def __init__(self, arch: DFlashHeadArch) -> None:
        """SwiGLU with a fused gate/up projection, matching the serving parameter names.

        Args:
            arch: Supplies the hidden and intermediate widths.
        """
        super().__init__()
        self.gate_up_proj = nn.Linear(
            arch.hidden_size, 2 * arch.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(arch.intermediate_size, arch.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SwiGLU.

        Args:
            x: `[..., hidden_size]`.

        Returns:
            `[..., hidden_size]`.
        """
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
