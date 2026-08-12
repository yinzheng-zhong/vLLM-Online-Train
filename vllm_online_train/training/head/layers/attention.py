import torch
import torch.nn.functional as F
from torch import nn

from vllm_online_train.training.head.arch import DFlashHeadArch
from vllm_online_train.training.head.layers.norm import RMSNorm
from vllm_online_train.training.head.layers.rotary import RotaryEmbedding


class DFlashAttention(nn.Module):
    def __init__(self, head_arch: DFlashHeadArch) -> None:
        """Grouped-query cross attention with Qwen3 per-head Q/K norm.

        Uses fused `qkv_proj` and `o_proj` rather than split projections, so the
        parameter names match the serving module.

        Args:
            head_arch: Supplies head counts, widths, bias and the RoPE base.
        """
        super().__init__()
        self.head_arch = head_arch
        self.scaling = head_arch.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            head_arch.hidden_size,
            head_arch.q_size + 2 * head_arch.kv_size,
            bias=head_arch.attention_bias,
        )
        self.o_proj = nn.Linear(
            head_arch.q_size, head_arch.hidden_size, bias=head_arch.attention_bias
        )
        self.q_norm = RMSNorm(head_arch.head_dim, head_arch.rms_norm_eps)
        self.k_norm = RMSNorm(head_arch.head_dim, head_arch.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(head_arch.head_dim, head_arch.rope_theta)

    def forward(
        self,
        query_input: torch.Tensor,
        context_kv_input: torch.Tensor,
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Attend block slots over `[context ; block slots]`.

        Args:
            query_input: `[B, Q, H]` post-`input_layernorm` block slots.
            context_kv_input: `[B, T, H]` projected context, already normed by
                `hidden_norm` and therefore fed without `input_layernorm`.
            query_positions: `[B, Q]` absolute positions of the block slots.
            context_positions: `[B, T]` absolute positions of the context.
            attn_bias: Additive mask broadcastable to `[B, heads, Q, T + Q]`.

        Returns:
            `[B, Q, H]`.
        """
        head_arch = self.head_arch
        batch_size, q_len, _ = query_input.shape
        context_len = context_kv_input.shape[1]

        q, k_q, v_q = self._project(query_input)
        # The context reuses the same qkv_proj, which is what the serving path's fused
        # KV GEMM does with `qkv_proj.weight[q_size:]`.
        _, k_c, v_c = self._project(context_kv_input)

        q = self.q_norm(
            self._heads(q, head_arch.num_heads, batch_size, q_len)
        ).transpose(1, 2)
        k_q = self.k_norm(
            self._heads(k_q, head_arch.num_kv_heads, batch_size, q_len)
        ).transpose(1, 2)
        k_c = self.k_norm(
            self._heads(k_c, head_arch.num_kv_heads, batch_size, context_len)
        ).transpose(1, 2)
        v_q = self._heads(
            v_q, head_arch.num_kv_heads, batch_size, q_len
        ).transpose(1, 2)
        v_c = self._heads(
            v_c, head_arch.num_kv_heads, batch_size, context_len
        ).transpose(1, 2)

        cos_q, sin_q = self.rotary_emb(query_positions)
        cos_c, sin_c = self.rotary_emb(context_positions)
        q = RotaryEmbedding.rotate(q, cos_q, sin_q)
        k_q = RotaryEmbedding.rotate(k_q, cos_q, sin_q)
        k_c = RotaryEmbedding.rotate(k_c, cos_c, sin_c)

        # KV order matches the mask's `[context ; blocks]` layout.
        k = self.repeat_kv(torch.cat([k_c, k_q], dim=2), head_arch.n_rep)
        v = self.repeat_kv(torch.cat([v_c, v_q], dim=2), head_arch.n_rep)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, q_len, head_arch.q_size
        )
        return self.o_proj(attn_output)

    def _project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the fused QKV projection into its three parts.

        Args:
            hidden_states: `[B, T, H]`.

        Returns:
            `[B, T, q_size]`, `[B, T, kv_size]`, `[B, T, kv_size]`.
        """
        qkv = self.qkv_proj(hidden_states)
        return qkv.split(
            [
                self.head_arch.q_size,
                self.head_arch.kv_size,
                self.head_arch.kv_size,
            ],
            dim=-1,
        )

    def _heads(
        self,
        projection: torch.Tensor,
        num_heads: int,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Reshape a projection into per-head views.

        Args:
            projection: `[B, T, num_heads * head_dim]`.
            num_heads: Heads to split into.
            batch_size: `B`.
            seq_len: `T`.

        Returns:
            `[B, T, num_heads, head_dim]`.
        """
        return projection.view(
            batch_size, seq_len, num_heads, self.head_arch.head_dim
        )

    @staticmethod
    def repeat_kv(kv_heads: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand key/value heads to match the query head count.

        Args:
            kv_heads: `[B, kv_heads, T, head_dim]`.
            n_rep: Query heads per key/value head.

        Returns:
            `[B, kv_heads * n_rep, T, head_dim]`, or the input when `n_rep` is 1.
        """
        if n_rep == 1:
            return kv_heads
        batch_size, num_kv_heads, seq_len, head_dim = kv_heads.shape
        return (
            kv_heads[:, :, None]
            .expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
            .reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
        )
