"""The head's building blocks, each testable now that it has its own file."""

import pytest
import torch
from torch import nn

from tests.conftest import make_arch
from vllm_online_train.training.head.layers.attention import DFlashAttention
from vllm_online_train.training.head.layers.decoder import DFlashLayer
from vllm_online_train.training.head.layers.mlp import DFlashMLP
from vllm_online_train.training.head.layers.norm import RMSNorm
from vllm_online_train.training.head.layers.rotary import RotaryEmbedding


def test_rms_norm_returns_unit_scale_rows():
    rms_norm = RMSNorm(8, 1e-6)
    hidden_states = torch.randn(4, 8) * 10.0
    normed = rms_norm(hidden_states)
    torch.testing.assert_close(
        normed.pow(2).mean(-1).sqrt(), torch.ones(4), atol=1e-4, rtol=1e-4
    )


def test_rms_norm_accumulates_in_fp32():
    """Matching vLLM's numerics means the reduction happens in fp32 even when the
    input is bf16, and the result comes back in the input dtype."""
    rms_norm = RMSNorm(8, 1e-6)
    normed = rms_norm(torch.randn(4, 8, dtype=torch.bfloat16))
    assert normed.dtype == torch.bfloat16


def test_rotary_does_not_shadow_module_apply():
    """`nn.Module.apply` must keep recursing: a method named `apply` here would
    silently stop weight init, dtype casts and compile passes at this node."""
    rotary_emb = RotaryEmbedding(8, 1e6)
    assert rotary_emb.apply.__func__ is nn.Module.apply

    visited_names = []
    parent = nn.Sequential(rotary_emb, nn.Linear(4, 4))
    parent.apply(lambda module: visited_names.append(type(module).__name__))
    assert "RotaryEmbedding" in visited_names


def test_rotate_half_swaps_and_negates():
    projected_heads = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(
        RotaryEmbedding.rotate_half(projected_heads),
        torch.tensor([[-3.0, -4.0, 1.0, 2.0]]),
    )


def test_rotation_preserves_norm():
    """RoPE is a rotation, so it cannot change a vector's length."""
    rotary_emb = RotaryEmbedding(8, 1e6)
    cos_table, sin_table = rotary_emb(torch.arange(5).view(1, 5))
    projected_heads = torch.randn(1, 2, 5, 8)
    rotated = RotaryEmbedding.rotate(projected_heads, cos_table, sin_table)
    torch.testing.assert_close(
        rotated.norm(dim=-1), projected_heads.norm(dim=-1), atol=1e-5, rtol=1e-5
    )


def test_rotation_at_position_zero_is_the_identity():
    rotary_emb = RotaryEmbedding(8, 1e6)
    cos_table, sin_table = rotary_emb(torch.zeros(1, 3, dtype=torch.long))
    projected_heads = torch.randn(1, 2, 3, 8)
    torch.testing.assert_close(
        RotaryEmbedding.rotate(projected_heads, cos_table, sin_table),
        projected_heads,
    )


def test_repeat_kv_is_a_no_op_for_matched_heads():
    kv_heads = torch.randn(1, 4, 6, 8)
    assert DFlashAttention.repeat_kv(kv_heads, 1) is kv_heads


def test_repeat_kv_expands_each_head_contiguously():
    kv_heads = torch.arange(2 * 3, dtype=torch.float32).view(1, 2, 3, 1)
    expanded = DFlashAttention.repeat_kv(kv_heads, 2)
    assert expanded.shape == (1, 4, 3, 1)
    torch.testing.assert_close(expanded[0, 0], expanded[0, 1])
    torch.testing.assert_close(expanded[0, 2], expanded[0, 3])


def test_mlp_preserves_the_hidden_width():
    head_arch = make_arch()
    mlp = DFlashMLP(head_arch)
    out = mlp(torch.randn(2, 5, head_arch.hidden_size))
    assert out.shape == (2, 5, head_arch.hidden_size)


def test_attention_projections_match_the_fused_serving_names():
    head_arch = make_arch()
    attention = DFlashAttention(head_arch)
    parameters_by_name = dict(attention.named_parameters())
    assert "qkv_proj.weight" in parameters_by_name
    assert "o_proj.weight" in parameters_by_name
    assert parameters_by_name["qkv_proj.weight"].shape == (
        head_arch.q_size + 2 * head_arch.kv_size,
        head_arch.hidden_size,
    )


@pytest.mark.parametrize("layer_cls", [DFlashAttention, DFlashLayer])
def test_layers_accept_the_replay_shapes(layer_cls):
    head_arch = make_arch()
    layer = layer_cls(head_arch)
    batch_size, q_len, context_len = 2, head_arch.block_size, 6
    query_input = torch.randn(batch_size, q_len, head_arch.hidden_size)
    context_kv_input = torch.randn(
        batch_size, context_len, head_arch.hidden_size
    )
    attn_bias = torch.zeros(batch_size, 1, q_len, context_len + q_len)
    out = layer(
        query_input,
        context_kv_input,
        torch.arange(q_len).expand(batch_size, q_len),
        torch.arange(context_len).expand(batch_size, context_len),
        attn_bias,
    )
    assert out.shape == (batch_size, q_len, head_arch.hidden_size)
    assert torch.isfinite(out).all()


def test_decoder_layer_is_residual():
    """A zeroed attention and MLP output must leave the input unchanged, which is
    what makes the pre-norm form algebraically the fused serving variant."""
    head_arch = make_arch()
    layer = DFlashLayer(head_arch)
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()

    hidden_states = torch.randn(1, head_arch.block_size, head_arch.hidden_size)
    out = layer(
        hidden_states,
        torch.randn(1, 4, head_arch.hidden_size),
        torch.arange(head_arch.block_size).view(1, -1),
        torch.arange(4).view(1, -1),
        torch.zeros(1, 1, head_arch.block_size, 4 + head_arch.block_size),
    )
    torch.testing.assert_close(out, hidden_states)
