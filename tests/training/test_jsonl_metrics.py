"""The JSONL metrics metrics_sink."""

import json

from vllm_online_train.training.jsonl_metrics import JsonlMetricsSink


def test_writes_one_object_per_line(tmp_path):
    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    assert metrics_sink.enabled
    metrics_sink.write("train", {"loss/total": 0.5})
    metrics_sink.write("train", {"loss/total": 0.25})
    metrics_sink.close()

    records = [
        json.loads(line) for line in metrics_sink.path.read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["event"] == "train"
    assert records[1]["loss/total"] == 0.25
    assert "t" in records[0]


def test_the_path_carries_the_pid(tmp_path):
    """Under TP>1 every worker writes its own metrics; a shared path would
    interleave them."""
    import os

    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    try:
        assert str(os.getpid()) in metrics_sink.path.name
        assert metrics_sink.path.suffix == ".jsonl"
    finally:
        metrics_sink.close()


def test_without_a_path_it_is_a_no_op():
    metrics_sink = JsonlMetricsSink(None)
    assert not metrics_sink.enabled
    assert metrics_sink.path is None
    metrics_sink.write("train", {"loss/total": 1.0})
    metrics_sink.close()


def test_an_unwritable_path_disables_rather_than_raises(tmp_path):
    """A metrics sink that can take down the worker is worse than no metrics."""
    blocking_file = tmp_path / "blocker"
    blocking_file.write_text("not a directory")
    metrics_sink = JsonlMetricsSink(blocking_file / "nested" / "metrics.jsonl")
    assert not metrics_sink.enabled
    metrics_sink.write("train", {"loss/total": 1.0})
    metrics_sink.close()


def test_writing_after_close_is_a_no_op(tmp_path):
    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    metrics_sink.write("train", {"a": 1.0})
    metrics_sink.close()
    metrics_sink.write("train", {"b": 2.0})
    assert len(metrics_sink.path.read_text().splitlines()) == 1


def test_close_is_idempotent(tmp_path):
    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    metrics_sink.close()
    metrics_sink.close()


def test_non_serialisable_values_fall_back_to_float(tmp_path):
    """Records come from several sources; one odd value must not lose the line."""
    import torch

    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    metrics_sink.write("train", {"loss": torch.tensor(0.5)})
    metrics_sink.close()
    record = json.loads(metrics_sink.path.read_text().splitlines()[0])
    assert record["loss"] == 0.5
