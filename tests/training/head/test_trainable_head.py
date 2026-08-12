"""Conventions the trainable head has to get right.

These are the ones that "still train" when wrong -- the head converges to something,
just not to a head the serving path will accept.
"""

import dataclasses

import pytest
import torch

from tests.conftest import mask_builder


def test_replay_drops_slot_zero(head, batch, config):
    """A K-slot block yields K-1 scored positions.

    Keeping slot 0 -- which re-feeds the anchor, a token already known -- trains the
    head as a parallel next-token predictor that appears to draft one extra token per
    block and flatters the acceptance rate.
    """
    shapes = config.shapes
    hidden = head.replay(batch)
    batch_size, num_anchors = batch.anchor_positions.shape
    assert hidden.shape == (
        batch_size,
        num_anchors,
        shapes.num_draft_tokens,
        shapes.hidden_size,
    )
    assert shapes.num_draft_tokens == shapes.block_size - 1

    anchors = batch.anchor_positions.clamp_min(0)
    full = head.forward_blocks(
        context_features=batch.target_features,
        anchor_positions=anchors,
        block_keep_mask=batch.block_keep_mask,
        anchor_token_ids=batch.input_ids.gather(1, anchors),
        context_valid_mask=batch.attention_mask,
    )
    assert full.shape[2] == shapes.block_size
    torch.testing.assert_close(hidden, full[:, :, 1:])


def test_shared_tensors_are_frozen(head):
    """The target's embedding table and LM head are borrowed, not trained."""
    trainable = {id(p) for p in head.trainable_parameters()}
    assert id(head.model.embed_tokens.weight) not in trainable
    assert id(head.lm_head.weight) not in trainable
    assert not head.model.embed_tokens.weight.requires_grad
    assert not head.lm_head.weight.requires_grad
    assert trainable, "nothing left to train"


def test_trainable_set_is_exactly_the_dflash_set(head):
    """`fc`, `hidden_norm`, `layers.*` and `norm`."""
    trained = {
        name for name, parameter in head.named_parameters() if parameter.requires_grad
    }
    assert trained
    for name in trained:
        assert name.startswith(
            ("model.fc", "model.hidden_norm", "model.layers.", "model.norm")
        ), f"unexpected trained parameter {name}"
    assert "model.fc.weight" in trained
    assert "model.norm.weight" in trained


def test_gradients_reach_every_trained_parameter(head, batch, objective):
    """A parameter with no gradient is silently not being trained at all."""
    loss, _ = objective(student_hidden=head.replay(batch), batch=batch)
    loss.backward()
    missing = [
        name
        for name, parameter in head.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
    ]
    assert not missing, f"no finite gradient for {missing}"


def test_project_context_rejects_the_wrong_feature_width(head, batch):
    """Capturing a different number of target layers than the head expects is the
    layer-id off-by-one, and must not be absorbed silently."""
    with pytest.raises(ValueError, match="capture is reading a different number"):
        head.project_context(batch.target_features[..., :-1])


def test_block_attention_is_bidirectional_and_stops_at_the_anchor(config):
    anchors = torch.tensor([[5, 9]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    context_len = 12
    k = config.shapes.block_size

    allow = mask_builder.anchored_block_mask(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_len=context_len,
        block_size=k,
        intra_block_causal=False,
    )[0, 0]

    # Block 0 occupies queries [0, k); its context reach is strictly < 5.
    assert allow[0, :5].all()
    assert not allow[0, 5:context_len].any()
    # Bidirectional inside the block, and never across blocks.
    assert allow[1, context_len + 2]
    assert not allow[1, context_len + k :].any()


def test_padding_blocks_never_leak_a_real_token(head, arch):
    """Padding blocks are all-MASK, so they cannot contribute a real token at a
    position meant to contribute nothing."""
    keep = torch.tensor([[True, False]])
    tokens = torch.tensor([[7, 11]])
    embedded = head.build_block_inputs(tokens, keep)

    mask_row = head.model.embed_tokens.weight[arch.mask_token_id].to(embedded.dtype)
    torch.testing.assert_close(embedded[0, arch.block_size], mask_row)
    assert not torch.allclose(embedded[0, 0], mask_row)


def test_block_inputs_are_cast_to_the_compute_dtype(head, arch):
    """The embedding table may be held at a lower precision than the layers."""
    from tests.conftest import head_weights

    head_weights.cast_shared(head, torch.bfloat16)
    embedded = head.build_block_inputs(
        torch.tensor([[3]]), torch.tensor([[True]])
    )
    assert embedded.dtype == head.compute_dtype == torch.float32


def test_no_nan_from_a_fully_masked_row(head, batch):
    """A softmax over an all -inf row yields NaN, which is why the mask re-opens each
    query's own slot. Padding blocks are the case that exercises it."""
    assert torch.isfinite(head.replay(batch)).all()


def test_the_head_borrows_its_mask_builder(head):
    """The objective reads position ids from the same instance, so the head must not
    construct its own."""
    assert head.masks is mask_builder


def test_project_context_takes_features_at_the_capture_dtype(head, batch):
    """The trained layers run at fp32 while the capture buffers features at the
    model's dtype, so the projection has to take the cast. Without it every step
    dies in `fc` on any bf16 target, which is every target vLLM serves by default."""
    features = batch.target_features.to(torch.bfloat16)
    assert features.dtype != head.compute_dtype

    projected = head.project_context(features)

    assert projected.dtype == head.compute_dtype
    torch.testing.assert_close(
        projected,
        head.project_context(features.to(head.compute_dtype)),
        rtol=0,
        atol=0,
    )


def test_replay_takes_features_at_the_capture_dtype(head, batch):
    """The whole replay, not just the projection: a cast that lands only in
    `project_context` would move the failure one layer down."""
    bf16 = dataclasses.replace(
        batch, target_features=batch.target_features.to(torch.bfloat16)
    )
    hidden = head.replay(bf16)
    assert hidden.dtype == head.compute_dtype
    hidden.sum().backward()
    assert head.model.fc.weight.grad is not None
