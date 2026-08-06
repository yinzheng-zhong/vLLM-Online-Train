"""Reading the operator's settings out of the environment."""

import json

import pytest

from tests.instances import REPO_ROOT
from vllm_online_train.engine.hook import capture_hook

settings_loader = capture_hook.loader

CONFIG_ENV = settings_loader.CONFIG_ENV


def test_no_config_means_completely_inert(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    assert settings_loader.load() is None


def test_an_empty_value_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, "")
    assert settings_loader.load() is None


def test_reads_inline_json(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"anchors_per_sequence": 7}))
    settings = settings_loader.load()
    assert settings.anchors.anchors_per_sequence == 7
    # Pointing the variable at a config is itself the opt-in.
    assert settings.enabled


def test_reads_a_file(tmp_path, monkeypatch):
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"sequences_per_step": 3, "idle_ms": 25}))
    monkeypatch.setenv(CONFIG_ENV, str(path))
    settings = settings_loader.load()
    assert settings.anchors.sequences_per_step == 3
    assert settings.gate.idle_ms == 25


def test_an_explicit_source_overrides_the_environment(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"idle_ms": 1}))
    settings = settings_loader.load(json.dumps({"idle_ms": 99}))
    assert settings.gate.idle_ms == 99


def test_enabled_false_survives(monkeypatch):
    """The default is only applied when the key is absent."""
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"enabled": False}))
    assert not settings_loader.load().enabled


def test_the_shipped_template_loads(monkeypatch):
    """`train.example.json` is what an operator copies, so it must actually parse."""
    template = REPO_ROOT / "train.example.json"
    if not template.is_file():
        pytest.skip("template not present")
    monkeypatch.setenv(CONFIG_ENV, str(template))
    settings = settings_loader.load()
    assert settings.enabled
    assert settings.objective.kl_chunk_size == 128
    assert settings.publish.publish_mode == "off"


def test_annotation_keys_are_ignored(monkeypatch):
    """JSON has no comments, so `_`-prefixed keys are reserved for notes."""
    monkeypatch.setenv(
        CONFIG_ENV, json.dumps({"_comment": "why", "_note": 1, "idle_ms": 33})
    )
    assert settings_loader.load().gate.idle_ms == 33


def test_unknown_config_keys_are_rejected(monkeypatch):
    """A typo must fail loudly instead of being ignored while the operator waits for
    a setting that never took effect."""
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"anchors_per_sequenc": 7}))
    with pytest.raises(ValueError, match="anchors_per_sequenc"):
        settings_loader.load()


def test_missing_config_file_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(ValueError, match="is not a file and is not JSON"):
        settings_loader.load()


def test_a_file_holding_a_json_array_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "train.json"
    path.write_text(json.dumps([1, 2, 3]))
    monkeypatch.setenv(CONFIG_ENV, str(path))
    with pytest.raises(TypeError, match="JSON object"):
        settings_loader.load()


def test_an_inline_value_that_is_not_an_object_is_treated_as_a_path(monkeypatch):
    """Only a leading `{` selects the inline branch, so anything else has to name a
    file -- and says so rather than failing on a JSON decode deeper in."""
    monkeypatch.setenv(CONFIG_ENV, json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="is not a file and is not JSON"):
        settings_loader.load()


def test_malformed_json_is_reported(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, "{not json")
    with pytest.raises(json.JSONDecodeError):
        settings_loader.load()


def test_hot_publish_requires_a_checkpoint_cadence(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"publish_mode": "hot"}))
    with pytest.raises(ValueError, match="checkpoint_every"):
        settings_loader.load()
