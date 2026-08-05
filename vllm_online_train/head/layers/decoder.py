import torch
from torch import nn

from vllm_online_train.head.arch import DFlashHeadArch
from vllm_online_train.head.layers.attention import DFlashAttention
from vllm_online_train.head.layers.mlp import DFlashMLP
from vllm_online_train.head.layers.norm import RMSNorm


class DFlashLayer(nn.Module):
    def __init__(self, arch: DFlashHeadArch) -> None:
        """One pre-norm decoder layer: cross attention over the context, then an MLP.

        Args:
            arch: Supplies every width the sublayers need.
        """
        super().__init__()
        self.self_attn = DFlashAttention(arch)
        self.mlp = DFlashMLP(arch)
        self.input_layernorm = RMSNorm(arch.hidden_size, arch.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(arch.hidden_size, arch.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        context_kv: torch.Tensor,
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the block slots by one layer.

        Args:
            hidden: `[B, Q, H]` block slots.
            context_kv: `[B, T, H]` projected context, the same at every depth.
            query_positions: `[B, Q]` absolute positions of the block slots.
            context_positions: `[B, T]` absolute positions of the context.
            attn_bias: Additive mask broadcastable to `[B, heads, Q, T + Q]`.

        Returns:
            `[B, Q, H]`.
        """
        # Explicit pre-norm form. The serving layer uses vLLM's fused
        # `RMSNorm(hidden, residual)` variant, which is algebraically this.
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden),
            context_kv,
            query_positions,
            context_positions,
            attn_bias,
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))
