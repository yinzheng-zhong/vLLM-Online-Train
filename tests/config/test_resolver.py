"""Resolving settings against a live engine config.

Resolution is the only place the draft/target pair is checked, so what it rejects
matters as much as what it computes: a head trained on the wrong feature layers still
converges, just to something that cannot be served.
"""

import pytest
import torch

from tests.conftest import (
    HIDDEN,
    NUM_FEATURES,
    config_resolver,
    fake_vllm_config,
    make_config,
    make_settings,
)


def test_resolve_applies_the_dflash_aux_layer_offset():
    """`target_layer_ids` are offset by +1 to become vLLM aux-layer ids.

    Reading the raw ids would capture layers one shallower than the drafter is fed.
    """
    resolved_config = config_resolver.resolve(
        fake_vllm_config(target_layer_ids=[2, 18, 33]), make_settings()
    )
    assert resolved_config.engine_shapes.aux_layer_ids == (3, 19, 34)


def test_resolve_derives_shapes():
    resolved_config = config_resolver.resolve(
        fake_vllm_config(num_speculative_tokens=15), make_settings()
    )
    engine_shapes = resolved_config.engine_shapes
    # K = num_speculative_tokens + 1: one anchor slot plus one slot per draft token.
    assert engine_shapes.block_size == 16
    assert engine_shapes.num_draft_tokens == 15
    assert engine_shapes.num_features == NUM_FEATURES
    assert engine_shapes.feature_width == NUM_FEATURES * HIDDEN


def test_resolve_rejects_non_dflash_methods():
    with pytest.raises(ValueError, match="method == 'dflash'"):
        config_resolver.resolve(fake_vllm_config(method="eagle3"), make_settings())


def test_resolve_rejects_unresolvable_feature_layers():
    with pytest.raises(ValueError, match="feature layers"):
        config_resolver.resolve(
            fake_vllm_config(target_layer_ids=[]), make_settings()
        )


def test_resolve_rejects_disabled_aux_hidden_state():
    """Without aux hidden states the drafter sees only the final layer, so there are
    no per-layer features to train on."""
    with pytest.raises(ValueError, match="use_aux_hidden_state"):
        config_resolver.resolve(
            fake_vllm_config(use_aux_hidden_state=False), make_settings()
        )


def test_resolve_rejects_zero_speculative_tokens():
    """A block needs at least one predicted slot beside the anchor."""
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        config_resolver.resolve(
            fake_vllm_config(num_speculative_tokens=0), make_settings()
        )


def test_resolve_rejects_feature_width_mismatch():
    """A hidden size the drafter's `fc` cannot consume means an incompatible pair."""
    vllm_config = fake_vllm_config(target_layer_ids=[0, 1, 2])
    vllm_config.model_config.get_hidden_size = lambda: HIDDEN * 2
    with pytest.raises(ValueError, match="fc expects"):
        config_resolver.resolve(vllm_config, make_settings())


def test_resolve_rejects_a_rollout_bound_below_the_block():
    with pytest.raises(ValueError, match="max_rollout_tokens"):
        config_resolver.resolve(
            fake_vllm_config(num_speculative_tokens=15),
            make_settings(max_rollout_tokens=16, min_rollout_tokens=8),
        )


def test_resolve_defaults_gamma_from_block_size():
    resolved_config = config_resolver.resolve(
        fake_vllm_config(num_speculative_tokens=15),
        make_settings(position_decay_gamma=None),
    )
    assert resolved_config.position_decay_gamma == pytest.approx(7.0)


def test_resolve_honours_explicit_gamma_zero():
    """0 must survive resolution as "uniform mean", not be treated as unset."""
    resolved_config = config_resolver.resolve(
        fake_vllm_config(), make_settings(position_decay_gamma=0.0)
    )
    assert resolved_config.position_decay_gamma == 0.0


def test_resolve_falls_back_to_the_engine_attribute():
    """Nothing sets `online_train_config` unless this subsystem is in-tree, but the
    attribute is honoured when present."""
    online_train_settings = make_settings(idle_ms=33.0)
    resolved_config = config_resolver.resolve(
        fake_vllm_config(online_train_settings=online_train_settings)
    )
    assert resolved_config.online_train_settings.gate.idle_ms == 33.0


def test_feature_dtype_follows_the_model_by_default():
    resolved_config = config_resolver.resolve(fake_vllm_config(), make_settings())
    assert resolved_config.engine_shapes.feature_dtype == torch.float32


def test_feature_dtype_can_be_overridden():
    resolved_config = config_resolver.resolve(
        fake_vllm_config(), make_settings(feature_dtype="bfloat16")
    )
    assert resolved_config.engine_shapes.feature_dtype == torch.bfloat16


def test_bytes_per_token_covers_features_and_final_hidden():
    resolved_config = make_config()
    engine_shapes = resolved_config.engine_shapes
    item_size = torch.empty((), dtype=engine_shapes.feature_dtype).element_size()
    expected_bytes = (
        engine_shapes.feature_width + engine_shapes.hidden_size
    ) * item_size + 4
    assert engine_shapes.bytes_per_token == expected_bytes
    assert resolved_config.buffer_capacity_bytes == (
        resolved_config.online_train_settings.buffer.buffer_capacity_tokens
        * expected_bytes
    )
