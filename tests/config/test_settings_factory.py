"""Routing the flat mapping an operator writes into typed sections.

The JSON schema is flat and stays flat: what changes below the plugin boundary is
that `IdleGate` receives gate settings rather than the whole config. The factory is
the one place that mapping lives, so an unrouted field is a silently ignored setting.
"""

import json
from dataclasses import fields
from pathlib import Path

import pytest

from tests.conftest import settings_factory
from vllm_online_train.config.factory import SettingsFactory


def test_flat_keys_reach_their_sections():
    settings = settings_factory.create(
        {
            "enabled": True,
            "idle_ms": 25.0,
            "kl_chunk_size": 128,
            "sequences_per_step": 3,
            "buffer_capacity_tokens": 60_000,
            "capture_sample_rate": 0.5,
            "learning_rate": 1e-4,
            "train_device": "cuda:1",
            "publish_mode": "off",
            "seed": 7,
        }
    )
    assert settings.enabled
    assert settings.gate.idle_ms == 25.0
    assert settings.objective.kl_chunk_size == 128
    assert settings.anchors.sequences_per_step == 3
    assert settings.buffer.buffer_capacity_tokens == 60_000
    assert settings.capture.capture_sample_rate == 0.5
    assert settings.optim.learning_rate == pytest.approx(1e-4)
    assert settings.placement.train_device == "cuda:1"
    assert settings.publish.publish_mode == "off"
    assert settings.bookkeeping.seed == 7


def test_every_section_field_is_routable():
    """A field no section claims would be silently dropped by `create`."""
    factory = SettingsFactory()
    for name, section in factory.SECTIONS.items():
        for field in fields(section):
            assert factory.field_owner(field.name) == name, (
                f"{field.name} is declared by {name} but routes elsewhere"
            )


def test_no_field_name_is_claimed_by_two_sections():
    """Two sections sharing a flat name would make routing order-dependent."""
    factory = SettingsFactory()
    seen: dict[str, str] = {}
    for name, section in factory.SECTIONS.items():
        for field in fields(section):
            assert field.name not in seen, (
                f"{field.name} is declared by both {seen.get(field.name)} and {name}"
            )
            seen[field.name] = name


def test_unknown_fields_are_rejected_by_name():
    """A typo must fail loudly rather than leave the operator waiting for a setting
    that never took effect."""
    with pytest.raises(ValueError, match="anchors_per_sequenc"):
        settings_factory.create({"anchors_per_sequenc": 7})


def test_the_error_lists_what_is_accepted():
    with pytest.raises(ValueError, match="idle_ms"):
        settings_factory.create({"nonsense": 1})


def test_defaults_apply_to_unmentioned_sections():
    settings = settings_factory.create({"idle_ms": 5.0})
    assert settings.buffer.buffer_capacity_tokens == 200_000
    assert settings.optim.grad_accum_steps == 1
    assert not settings.enabled


def test_known_fields_covers_the_master_switch():
    assert "enabled" in settings_factory.known_fields()
    assert settings_factory.field_owner("enabled") == "enabled"


def test_the_shipped_template_still_routes():
    """`train.example.json` is what an operator copies, so it must parse unchanged."""
    template = Path(__file__).resolve().parents[2] / "train.example.json"
    if not template.is_file():
        pytest.skip("template not present")
    flat = {
        key: value
        for key, value in json.loads(template.read_text()).items()
        if not key.startswith("_")
    }
    flat.setdefault("enabled", True)
    settings = settings_factory.create(flat)
    assert settings.enabled
    assert settings.objective.kl_chunk_size == 128
    assert settings.publish.publish_mode == "off"
