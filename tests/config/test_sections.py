"""What each settings section refuses to be constructed with.

Every section is frozen and forbids extras, so a typo or an unusable value fails at
startup rather than after a night of training.
"""

import dataclasses

import pytest

from vllm_online_train.config.settings import (
    AnchorSettings,
    BookkeepingSettings,
    BufferSettings,
    CaptureSettings,
    GateSettings,
    HeadInitSettings,
    ObjectiveSettings,
    OnlineTrainSettings,
    OptimSettings,
    PlacementSettings,
    PublishSettings,
)

SECTIONS = [
    AnchorSettings,
    BookkeepingSettings,
    BufferSettings,
    CaptureSettings,
    GateSettings,
    HeadInitSettings,
    ObjectiveSettings,
    OptimSettings,
    PlacementSettings,
    PublishSettings,
]


@pytest.mark.parametrize("section_cls", SECTIONS)
def test_every_section_is_frozen(section_cls):
    section = section_cls()
    field_name = dataclasses.fields(section_cls)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(section, field_name, getattr(section, field_name))


@pytest.mark.parametrize("section_cls", SECTIONS)
def test_every_section_forbids_unknown_fields(section_cls):
    with pytest.raises(ValueError):
        section_cls(definitely_not_a_field=1)


@pytest.mark.parametrize(
    "section_cls,overrides",
    [
        (ObjectiveSettings, {"kl_chunk_size": 0}),
        (ObjectiveSettings, {"position_decay_gamma": -1.0}),
        (AnchorSettings, {"anchors_per_sequence": 0}),
        (AnchorSettings, {"anchor_stride": 0}),
        (AnchorSettings, {"sequences_per_step": 0}),
        (OptimSettings, {"grad_accum_steps": 0}),
        (OptimSettings, {"warmup_ratio": 1.5}),
        (OptimSettings, {"total_steps": 0}),
        (CaptureSettings, {"capture_sample_rate": 0.0}),
        (CaptureSettings, {"capture_sample_rate": 1.5}),
        (BufferSettings, {"min_rollout_tokens": 100, "max_rollout_tokens": 50}),
        (BufferSettings, {"buffer_capacity_tokens": 10, "max_rollout_tokens": 2048}),
        (GateSettings, {"idle_ms": -1}),
        (GateSettings, {"poll_ms": 0}),
        (GateSettings, {"busy_tokens": -1}),
        (GateSettings, {"max_micro_steps_per_gate": -1}),
        (BookkeepingSettings, {"checkpoint_every": -1}),
    ],
)
def test_sections_reject_unusable_values(section_cls, overrides):
    with pytest.raises(ValueError):
        section_cls(**overrides)


@pytest.mark.parametrize(
    "block_size,expected", [(16, 7.0), (10, 5.0), (8, 4.0), (12, 5.5)]
)
def test_position_decay_gamma_follows_dflash_a1(block_size, expected):
    """A.1 pins 7/5/4 at block 16/10/8; anything else interpolates linearly."""
    assert ObjectiveSettings.default_gamma(block_size) == pytest.approx(expected)


def test_gate_exposes_its_durations_in_seconds():
    gate_settings = GateSettings(idle_ms=50.0, poll_ms=10.0)
    assert gate_settings.idle_seconds == pytest.approx(0.05)
    assert gate_settings.poll_seconds == pytest.approx(0.01)


def test_publish_mode_exposes_a_hot_predicate():
    assert not PublishSettings().hot
    assert PublishSettings(publish_mode="hot").hot


def test_hot_publish_requires_a_checkpoint_cadence():
    """The checkpoint cadence is also the publish cadence; a hot publish with no
    checkpoint beside it leaves nothing to restart from."""
    with pytest.raises(ValueError, match="checkpoint_every"):
        OnlineTrainSettings(
            publish=PublishSettings(publish_mode="hot"),
            bookkeeping=BookkeepingSettings(checkpoint_every=0),
        )


def test_hot_publish_is_accepted_with_a_cadence():
    online_train_settings = OnlineTrainSettings(
        publish=PublishSettings(publish_mode="hot"),
        bookkeeping=BookkeepingSettings(checkpoint_every=10),
    )
    assert online_train_settings.publish.hot


def test_placement_defaults_to_the_engines_device():
    """Nothing about the plugin's placement may change unless it is asked for: an
    installed-but-unconfigured trainer stays on the serving GPU."""
    assert PlacementSettings().train_device is None
    assert OnlineTrainSettings().placement.train_device is None


def test_the_head_defaults_to_continuing_the_served_checkpoint():
    """A restart that quietly retrained from scratch is indistinguishable from a head
    that learned nothing, so continuing is the default and starting over is opt-in."""
    assert HeadInitSettings().from_served
    assert OnlineTrainSettings().head_init.from_served
    assert not HeadInitSettings(head_init_source="random").from_served


def test_the_aggregate_is_frozen():
    online_train_settings = OnlineTrainSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        online_train_settings.enabled = True
