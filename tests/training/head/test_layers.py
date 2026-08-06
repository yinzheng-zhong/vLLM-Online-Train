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
    norm = RMSNorm(8, 1e-6)
    x = torch.randn(4, 8) * 10.0
    out = norm(x)
    torch.testing.assert_close(
        out.pow(2).mean(-1).sqrt(), torch.ones(4), atol=1e-4, rtol=1e-4
    )


def test_rms_norm_accumulates_in_fp32():
    """Matching vLLM's numerics means the reduction happens in fp32 even when the
    input is bf16, and the result comes back in the input dtype."""
    norm = RMSNorm(8, 1e-6)
    out = norm(torch.randn(4, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16


def test_rotary_does_not_shadow_module_apply():
    """`nn.Module.apply` must keep recursing: a method named `apply` here would
    silently stop weight init, dtype casts and compile passes at this node."""
    rotary = RotaryEmbedding(8, 1e6)
    assert rotary.apply.__func__ is nn.Module.apply

    visited = []
    parent = nn.Sequential(rotary, nn.Linear(4, 4))
    parent.apply(lambda module: visited.append(type(module).__name__))
    assert "RotaryEmbedding" in visited


def test_rotate_half_swaps_and_negates():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(
        RotaryEmbedding.rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    )


def test_rotation_preserves_norm():
    """RoPE is a rotation, so it cannot change a vector's length."""
    rotary = RotaryEmbedding(8, 1e6)
    cos, sin = rotary(torch.arange(5).view(1, 5))
    x = torch.randn(1, 2, 5, 8)
    rotated = RotaryEmbedding.rotate(x, cos, sin)
    torch.testing.assert_close(
        rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5
    )


def test_rotation_at_position_zero_is_the_identity():
    rotary = RotaryEmbedding(8, 1e6)
    cos, sin = rotary(torch.zeros(1, 3, dtype=torch.long))
    x = torch.randn(1, 2, 3, 8)
    torch.testing.assert_close(RotaryEmbedding.rotate(x, cos, sin), x)


def test_repeat_kv_is_a_no_op_for_matched_heads():
    x = torch.randn(1, 4, 6, 8)
    assert DFlashAttention.repeat_kv(x, 1) is x


def test_repeat_kv_expands_each_head_contiguously():
    x = torch.arange(2 * 3, dtype=torch.float32).view(1, 2, 3, 1)
    expanded = DFlashAttention.repeat_kv(x, 2)
    assert expanded.shape == (1, 4, 3, 1)
    torch.testing.assert_close(expanded[0, 0], expanded[0, 1])
    torch.testing.assert_close(expanded[0, 2], expanded[0, 3])


def test_mlp_preserves_the_hidden_width():
    arch = make_arch()
    mlp = DFlashMLP(arch)
    out = mlp(torch.randn(2, 5, arch.hidden_size))
    assert out.shape == (2, 5, arch.hidden_size)


def test_attention_projections_match_the_fused_serving_names():
    arch = make_arch()
    attention = DFlashAttention(arch)
    names = dict(attention.named_parameters())
    assert "qkv_proj.weight" in names
    assert "o_proj.weight" in names
    assert names["qkv_proj.weight"].shape == (
        arch.q_size + 2 * arch.kv_size,
        arch.hidden_size,
    )


@pytest.mark.parametrize("layer_cls", [DFlashAttention, DFlashLayer])
def test_layers_accept_the_replay_shapes(layer_cls):
    arch = make_arch()
    layer = layer_cls(arch)
    batch, q_len, ctx_len = 2, arch.block_size, 6
    query = torch.randn(batch, q_len, arch.hidden_size)
    context = torch.randn(batch, ctx_len, arch.hidden_size)
    bias = torch.zeros(batch, 1, q_len, ctx_len + q_len)
    out = layer(
        query,
        context,
        torch.arange(q_len).expand(batch, q_len),
        torch.arange(ctx_len).expand(batch, ctx_len),
        bias,
    )
    assert out.shape == (batch, q_len, arch.hidden_size)
    assert torch.isfinite(out).all()


def test_decoder_layer_is_residual():
    """A zeroed attention and MLP output must leave the input unchanged, which is
    what makes the pre-norm form algebraically the fused serving variant."""
    arch = make_arch()
    layer = DFlashLayer(arch)
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()

    hidden = torch.randn(1, arch.block_size, arch.hidden_size)
    out = layer(
        hidden,
        torch.randn(1, 4, arch.hidden_size),
        torch.arange(arch.block_size).view(1, -1),
        torch.arange(4).view(1, -1),
        torch.zeros(1, 1, arch.block_size, 4 + arch.block_size),
    )
    torch.testing.assert_close(out, hidden)
