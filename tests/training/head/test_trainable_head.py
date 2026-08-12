"""Conventions the trainable head has to get right.

These are the ones that "still train" when wrong -- the head converges to something,
just not to a head the serving path will accept.
"""

import dataclasses

import pytest
import torch

from tests.conftest import block_mask_builder


def test_replay_drops_slot_zero(draft_head, replay_batch, resolved_config):
    """A K-slot block yields K-1 scored positions.

    Keeping slot 0 -- which re-feeds the anchor, a token already known -- trains the
    head as a parallel next-token predictor that appears to draft one extra token per
    block and flatters the acceptance rate.
    """
    engine_shapes = resolved_config.engine_shapes
    scored_hidden = draft_head.replay(replay_batch)
    batch_size, num_anchors = replay_batch.anchor_positions.shape
    assert scored_hidden.shape == (
        batch_size,
        num_anchors,
        engine_shapes.num_draft_tokens,
        engine_shapes.hidden_size,
    )
    assert engine_shapes.num_draft_tokens == engine_shapes.block_size - 1

    anchor_positions = replay_batch.anchor_positions.clamp_min(0)
    full_hidden = draft_head.forward_blocks(
        context_features=replay_batch.target_features,
        anchor_positions=anchor_positions,
        block_keep_mask=replay_batch.block_keep_mask,
        anchor_token_ids=replay_batch.input_ids.gather(1, anchor_positions),
        context_valid_mask=replay_batch.attention_mask,
    )
    assert full_hidden.shape[2] == engine_shapes.block_size
    torch.testing.assert_close(scored_hidden, full_hidden[:, :, 1:])


def test_shared_tensors_are_frozen(draft_head):
    """The target's embedding table and LM head are borrowed, not trained."""
    trainable_ids = {
        id(parameter) for parameter in draft_head.trainable_parameters()
    }
    assert id(draft_head.model.embed_tokens.weight) not in trainable_ids
    assert id(draft_head.lm_head.weight) not in trainable_ids
    assert not draft_head.model.embed_tokens.weight.requires_grad
    assert not draft_head.lm_head.weight.requires_grad
    assert trainable_ids, "nothing left to train"


def test_trainable_set_is_exactly_the_dflash_set(draft_head):
    """`fc`, `hidden_norm`, `layers.*` and `norm`."""
    trained_names = {
        name
        for name, parameter in draft_head.named_parameters()
        if parameter.requires_grad
    }
    assert trained_names
    for name in trained_names:
        assert name.startswith(
            ("model.fc", "model.hidden_norm", "model.layers.", "model.norm")
        ), f"unexpected trained parameter {name}"
    assert "model.fc.weight" in trained_names
    assert "model.norm.weight" in trained_names


def test_gradients_reach_every_trained_parameter(
    draft_head, replay_batch, sft_objective
):
    """A parameter with no gradient is silently not being trained at all."""
    loss, _ = sft_objective(
        student_hidden=draft_head.replay(replay_batch), replay_batch=replay_batch
    )
    loss.backward()
    missing_names = [
        name
        for name, parameter in draft_head.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
    ]
    assert not missing_names, f"no finite gradient for {missing_names}"


def test_project_context_rejects_the_wrong_feature_width(draft_head, replay_batch):
    """Capturing a different number of target layers than the head expects is the
    layer-id off-by-one, and must not be absorbed silently."""
    with pytest.raises(ValueError, match="capture is reading a different number"):
        draft_head.project_context(replay_batch.target_features[..., :-1])


def test_block_attention_is_bidirectional_and_stops_at_the_anchor(resolved_config):
    anchor_positions = torch.tensor([[5, 9]])
    block_keep_mask = torch.ones_like(anchor_positions, dtype=torch.bool)
    context_len = 12
    block_size = resolved_config.engine_shapes.block_size

    allow_mask = block_mask_builder.anchored_block_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        context_len=context_len,
        block_size=block_size,
        intra_block_causal=False,
    )[0, 0]

    # Block 0 occupies queries [0, K); its context reach is strictly < 5.
    assert allow_mask[0, :5].all()
    assert not allow_mask[0, 5:context_len].any()
    # Bidirectional inside the block, and never across blocks.
    assert allow_mask[1, context_len + 2]
    assert not allow_mask[1, context_len + block_size :].any()


def test_padding_blocks_never_leak_a_real_token(draft_head, head_arch):
    """Padding blocks are all-MASK, so they cannot contribute a real token at a
    position meant to contribute nothing."""
    block_keep_mask = torch.tensor([[True, False]])
    anchor_token_ids = torch.tensor([[7, 11]])
    embedded = draft_head.build_block_inputs(anchor_token_ids, block_keep_mask)

    mask_row = draft_head.model.embed_tokens.weight[head_arch.mask_token_id].to(
        embedded.dtype
    )
    torch.testing.assert_close(embedded[0, head_arch.block_size], mask_row)
    assert not torch.allclose(embedded[0, 0], mask_row)


def test_block_inputs_are_cast_to_the_compute_dtype(draft_head):
    """The embedding table may be held at a lower precision than the layers."""
    from tests.conftest import head_weights

    head_weights.cast_shared(draft_head, torch.bfloat16)
    embedded = draft_head.build_block_inputs(
        torch.tensor([[3]]), torch.tensor([[True]])
    )
    assert embedded.dtype == draft_head.compute_dtype == torch.float32


def test_no_nan_from_a_fully_masked_row(draft_head, replay_batch):
    """A softmax over an all -inf row yields NaN, which is why the mask re-opens each
    query's own slot. Padding blocks are the case that exercises it."""
    assert torch.isfinite(draft_head.replay(replay_batch)).all()


def test_the_head_borrows_its_mask_builder(draft_head):
    """The objective reads position ids from the same instance, so the head must not
    construct its own."""
    assert draft_head.block_mask_builder is block_mask_builder


def test_project_context_takes_features_at_the_capture_dtype(
    draft_head, replay_batch
):
    """The trained layers run at fp32 while the capture buffers features at the
    model's dtype, so the projection has to take the cast. Without it every step
    dies in `fc` on any bf16 target, which is every target vLLM serves by default."""
    target_features = replay_batch.target_features.to(torch.bfloat16)
    assert target_features.dtype != draft_head.compute_dtype

    projected = draft_head.project_context(target_features)

    assert projected.dtype == draft_head.compute_dtype
    torch.testing.assert_close(
        projected,
        draft_head.project_context(
            target_features.to(draft_head.compute_dtype)
        ),
        rtol=0,
        atol=0,
    )


def test_replay_takes_features_at_the_capture_dtype(draft_head, replay_batch):
    """The whole replay, not just the projection: a cast that lands only in
    `project_context` would move the failure one layer down."""
    bf16_batch = dataclasses.replace(
        replay_batch,
        target_features=replay_batch.target_features.to(torch.bfloat16),
    )
    scored_hidden = draft_head.replay(bf16_batch)
    assert scored_hidden.dtype == draft_head.compute_dtype
    scored_hidden.sum().backward()
    assert draft_head.model.fc.weight.grad is not None
