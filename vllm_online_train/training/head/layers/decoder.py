import torch
from torch import nn

from vllm_online_train.training.head.arch import DFlashHeadArch
from vllm_online_train.training.head.layers.attention import DFlashAttention
from vllm_online_train.training.head.layers.mlp import DFlashMLP
from vllm_online_train.training.head.layers.norm import RMSNorm


class DFlashLayer(nn.Module):
    def __init__(self, head_arch: DFlashHeadArch) -> None:
        """One pre-norm decoder layer: cross attention over the context, then an MLP.

        Args:
            head_arch: Supplies every width the sublayers need.
        """
        super().__init__()
        self.self_attn = DFlashAttention(head_arch)
        self.mlp = DFlashMLP(head_arch)
        self.input_layernorm = RMSNorm(
            head_arch.hidden_size, head_arch.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            head_arch.hidden_size, head_arch.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_kv_input: torch.Tensor,
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the block slots by one layer.

        Args:
            hidden_states: `[B, Q, H]` block slots.
            context_kv_input: `[B, T, H]` projected context, the same at every depth.
            query_positions: `[B, Q]` absolute positions of the block slots.
            context_positions: `[B, T]` absolute positions of the context.
            attn_bias: Additive mask broadcastable to `[B, heads, Q, T + Q]`.

        Returns:
            `[B, Q, H]`.
        """
        # Explicit pre-norm form. The serving layer uses vLLM's fused
        # `RMSNorm(hidden, residual)` variant, which is algebraically this.
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            context_kv_input,
            query_positions,
            context_positions,
            attn_bias,
        )
        return hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
