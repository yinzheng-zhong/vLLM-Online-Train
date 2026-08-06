"""The JSONL metrics sink."""

import json

from vllm_online_train.training.jsonl_metrics import JsonlMetricsSink


def test_writes_one_object_per_line(tmp_path):
    sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    assert sink.enabled
    sink.write("train", {"loss/total": 0.5})
    sink.write("train", {"loss/total": 0.25})
    sink.close()

    lines = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["event"] == "train"
    assert lines[1]["loss/total"] == 0.25
    assert "t" in lines[0]


def test_the_path_carries_the_pid(tmp_path):
    """Under TP>1 every worker writes its own metrics; a shared path would
    interleave them."""
    import os

    sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    try:
        assert str(os.getpid()) in sink.path.name
        assert sink.path.suffix == ".jsonl"
    finally:
        sink.close()


def test_without_a_path_it_is_a_no_op():
    sink = JsonlMetricsSink(None)
    assert not sink.enabled
    assert sink.path is None
    sink.write("train", {"loss/total": 1.0})
    sink.close()


def test_an_unwritable_path_disables_rather_than_raises(tmp_path):
    """A metrics sink that can take down the worker is worse than no metrics."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    sink = JsonlMetricsSink(blocker / "nested" / "metrics.jsonl")
    assert not sink.enabled
    sink.write("train", {"loss/total": 1.0})
    sink.close()


def test_writing_after_close_is_a_no_op(tmp_path):
    sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    sink.write("train", {"a": 1.0})
    sink.close()
    sink.write("train", {"b": 2.0})
    assert len(sink.path.read_text().splitlines()) == 1


def test_close_is_idempotent(tmp_path):
    sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    sink.close()
    sink.close()


def test_non_serialisable_values_fall_back_to_float(tmp_path):
    """Records come from several sources; one odd value must not lose the line."""
    import torch

    sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    sink.write("train", {"loss": torch.tensor(0.5)})
    sink.close()
    record = json.loads(sink.path.read_text().splitlines()[0])
    assert record["loss"] == 0.5
