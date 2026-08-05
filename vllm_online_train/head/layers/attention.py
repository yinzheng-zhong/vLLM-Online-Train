import torch
import torch.nn.functional as F
from torch import nn

from vllm_online_train.head.arch import DFlashHeadArch
from vllm_online_train.head.layers.norm import RMSNorm
from vllm_online_train.head.layers.rotary import RotaryEmbedding


class DFlashAttention(nn.Module):
    def __init__(self, arch: DFlashHeadArch) -> None:
        """Grouped-query cross attention with Qwen3 per-head Q/K norm.

        Uses fused `qkv_proj` and `o_proj` rather than split projections, so the
        parameter names match the serving module.

        Args:
            arch: Supplies head counts, widths, bias and the RoPE base.
        """
        super().__init__()
        self.arch = arch
        self.scaling = arch.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            arch.hidden_size,
            arch.q_size + 2 * arch.kv_size,
            bias=arch.attention_bias,
        )
        self.o_proj = nn.Linear(arch.q_size, arch.hidden_size, bias=arch.attention_bias)
        self.q_norm = RMSNorm(arch.head_dim, arch.rms_norm_eps)
        self.k_norm = RMSNorm(arch.head_dim, arch.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(arch.head_dim, arch.rope_theta)

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
        arch = self.arch
        batch, q_len, _ = query_input.shape
        ctx_len = context_kv_input.shape[1]

        q, k_q, v_q = self._project(query_input)
        # The context reuses the same qkv_proj, which is what the serving path's fused
        # KV GEMM does with `qkv_proj.weight[q_size:]`.
        _, k_c, v_c = self._project(context_kv_input)

        q = self.q_norm(self._heads(q, arch.num_heads, batch, q_len)).transpose(1, 2)
        k_q = self.k_norm(
            self._heads(k_q, arch.num_kv_heads, batch, q_len)
        ).transpose(1, 2)
        k_c = self.k_norm(
            self._heads(k_c, arch.num_kv_heads, batch, ctx_len)
        ).transpose(1, 2)
        v_q = self._heads(v_q, arch.num_kv_heads, batch, q_len).transpose(1, 2)
        v_c = self._heads(v_c, arch.num_kv_heads, batch, ctx_len).transpose(1, 2)

        cos_q, sin_q = self.rotary_emb(query_positions)
        cos_c, sin_c = self.rotary_emb(context_positions)
        q = RotaryEmbedding.rotate(q, cos_q, sin_q)
        k_q = RotaryEmbedding.rotate(k_q, cos_q, sin_q)
        k_c = RotaryEmbedding.rotate(k_c, cos_c, sin_c)

        # KV order matches the mask's `[context ; blocks]` layout.
        k = self.repeat_kv(torch.cat([k_c, k_q], dim=2), arch.n_rep)
        v = self.repeat_kv(torch.cat([v_c, v_q], dim=2), arch.n_rep)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, scale=self.scaling
        )
        out = out.transpose(1, 2).reshape(batch, q_len, arch.q_size)
        return self.o_proj(out)

    def _project(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the fused QKV projection into its three parts.

        Args:
            hidden: `[B, T, H]`.

        Returns:
            `[B, T, q_size]`, `[B, T, kv_size]`, `[B, T, kv_size]`.
        """
        qkv = self.qkv_proj(hidden)
        return qkv.split(
            [self.arch.q_size, self.arch.kv_size, self.arch.kv_size], dim=-1
        )

    def _heads(
        self, x: torch.Tensor, count: int, batch: int, length: int
    ) -> torch.Tensor:
        """Reshape a projection into per-head views.

        Args:
            x: `[B, T, count * head_dim]`.
            count: Heads to split into.
            batch: `B`.
            length: `T`.

        Returns:
            `[B, T, count, head_dim]`.
        """
        return x.view(batch, length, count, self.arch.head_dim)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand key/value heads to match the query head count.

        Args:
            x: `[B, kv_heads, T, head_dim]`.
            n_rep: Query heads per key/value head.

        Returns:
            `[B, kv_heads * n_rep, T, head_dim]`, or `x` when `n_rep` is 1.
        """
        if n_rep == 1:
            return x
        batch, heads, seq, dim = x.shape
        return (
            x[:, :, None]
            .expand(batch, heads, n_rep, seq, dim)
            .reshape(batch, heads * n_rep, seq, dim)
        )
