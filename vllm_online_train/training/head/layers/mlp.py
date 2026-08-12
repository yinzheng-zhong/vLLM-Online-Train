import torch
import torch.nn.functional as F
from torch import nn

from vllm_online_train.training.head.arch import DFlashHeadArch


class DFlashMLP(nn.Module):
    def __init__(self, head_arch: DFlashHeadArch) -> None:
        """SwiGLU with a fused gate/up projection, matching the serving parameter names.

        Args:
            head_arch: Supplies the hidden and intermediate widths.
        """
        super().__init__()
        self.gate_up_proj = nn.Linear(
            head_arch.hidden_size, 2 * head_arch.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            head_arch.intermediate_size, head_arch.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the SwiGLU.

        Args:
            hidden_states: `[..., hidden_size]`.

        Returns:
            `[..., hidden_size]`.
        """
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
