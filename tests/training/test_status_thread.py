"""The status thread: reporting the pool even when nothing is training.

The case that matters is the one the `train` records cannot cover -- a gate that never
opens, or a pool that never fills -- because a run in that state writes nothing at all
and looks identical to a run that is not installed.
"""

import json
import threading
import time

from tests.conftest import make_settings
from vllm_online_train.training.idle_gate import IdleGate
from vllm_online_train.training.jsonl_metrics import JsonlMetricsSink
from vllm_online_train.training.status_thread import StatusThread


class FakeRunner:
    """Reports counters and records that it was asked. Never trains."""

    def __init__(self) -> None:
        self.calls = 0
        self.asked = threading.Event()

    def train_step(self):
        raise AssertionError("the status thread must never run a training step")

    def metrics(self):
        self.calls += 1
        self.asked.set()
        return {
            "buffer/rollouts": 7.0,
            "buffer/tokens": 800.0,
            "buffer/fill": 0.4,
            "buffer/eligible": 5.0,
            "buffer/pending": 2.0,
            "buffer/admitted": 9.0,
            "buffer/dropped_too_short": 3.0,
            "buffer/dropped_too_long": 4.0,
            "train/step": 0.0,
        }


def build(step_runner, metrics_sink=None, **overrides) -> StatusThread:
    """A status thread on a fast cadence.

    Args:
        step_runner: Supplies the counters.
        metrics_sink: Receives the records. Defaults to a disabled sink.
        **overrides: Flat settings fields.

    Returns:
        The thread, not started.
    """
    field_values = {"status_every_ms": 10.0}
    field_values.update(overrides)
    online_train_settings = make_settings(**field_values)
    return StatusThread(
        IdleGate(online_train_settings.gate),
        step_runner,
        online_train_settings.bookkeeping,
        metrics_sink or JsonlMetricsSink(None),
    )


def test_reports_without_a_training_step(tmp_path):
    """The whole point: a shut gate and an empty pool still produce records."""
    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    step_runner = FakeRunner()
    status_thread = build(step_runner, metrics_sink=metrics_sink, idle_ms=10_000)
    status_thread.idle_gate.note_step(1)
    status_thread.start()
    try:
        assert step_runner.asked.wait(timeout=5.0), "status thread never reported"
        time.sleep(0.05)
    finally:
        status_thread.stop()
    metrics_sink.close()

    records = [
        json.loads(line) for line in metrics_sink.path.read_text().splitlines()
    ]
    status_records = [record for record in records if record["event"] == "status"]
    assert status_records
    assert status_records[0]["buffer/rollouts"] == 7.0
    assert status_records[0]["buffer/pending"] == 2.0
    assert status_records[0]["gate/open_fraction"] == 0.0
    assert status_records[0]["engine/steps"] == 1.0


def test_reports_on_the_configured_cadence(tmp_path):
    """A cadence that is ignored either floods the sink or reports once and stops."""
    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    duration, status_every_ms = 0.3, 25.0
    status_thread = build(
        FakeRunner(), metrics_sink=metrics_sink, status_every_ms=status_every_ms
    )
    status_thread.start()
    try:
        time.sleep(duration)
    finally:
        status_thread.stop()
    metrics_sink.close()

    expected = duration / (status_every_ms / 1000.0)
    assert 2 <= status_thread.reports <= expected + 4, (
        f"{status_thread.reports} reports in {duration}s at "
        f"status_every_ms={status_every_ms} (expected about {expected:.0f})"
    )


def test_a_zero_cadence_disables_the_thread(tmp_path):
    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    step_runner = FakeRunner()
    status_thread = build(
        step_runner, metrics_sink=metrics_sink, status_every_ms=0.0
    )
    assert not status_thread.enabled
    status_thread.start()
    try:
        time.sleep(0.1)
    finally:
        status_thread.stop()
    metrics_sink.close()

    assert step_runner.calls == 0
    assert metrics_sink.path.read_text() == ""


def test_a_raising_snapshot_is_contained():
    """A counter this cannot read must not end the reporting, which would take the
    only view of the pool with it."""

    class Exploding(FakeRunner):
        def metrics(self):
            self.calls += 1
            self.asked.set()
            raise RuntimeError("kaboom")

    step_runner = Exploding()
    status_thread = build(step_runner)
    status_thread.start()
    try:
        assert step_runner.asked.wait(timeout=5.0)
        time.sleep(0.1)
    finally:
        status_thread.stop()

    assert status_thread.errors > 0
    assert status_thread.reports == 0
    assert step_runner.calls > 1, "reporting stopped at the first failure"


def test_start_is_idempotent():
    step_runner = FakeRunner()
    status_thread = build(step_runner)
    status_thread.start()
    status_thread.start()
    try:
        assert step_runner.asked.wait(timeout=5.0)
    finally:
        status_thread.stop()


def test_stopping_a_thread_that_never_started_is_harmless():
    build(FakeRunner()).stop()


def test_the_drop_total_sums_every_reason():
    status_thread = build(FakeRunner())
    assert status_thread._num_dropped(FakeRunner().metrics()) == 7.0
