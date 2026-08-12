"""The trainer thread: training in the gaps, and getting out of the way."""

import threading
import time

from tests.conftest import make_settings
from vllm_online_train.training.idle_gate import IdleGate
from vllm_online_train.training.jsonl_metrics import JsonlMetricsSink
from vllm_online_train.training.trainer_thread import TrainerThread


class FakeRunner:
    """Counts micro-batches and can starve, to test that the gate is re-checked."""

    def __init__(self, thin_after: int | None = None) -> None:
        self.calls = 0
        self.thin_after = thin_after
        self.entered = threading.Event()

    def train_step(self):
        self.calls += 1
        self.entered.set()
        if self.thin_after is not None and self.calls > self.thin_after:
            return None
        return {"loss/total": 1.0 / self.calls, "step": float(self.calls)}

    def metrics(self):
        return {"buffer/rollouts": 1.0}


def build(step_runner, metrics_sink=None, **overrides) -> TrainerThread:
    field_values = {"idle_ms": 1, "poll_ms": 1, "log_every": 1}
    field_values.update(overrides)
    online_train_settings = make_settings(**field_values)
    return TrainerThread(
        IdleGate(online_train_settings.gate),
        step_runner,
        online_train_settings.gate,
        online_train_settings.bookkeeping,
        metrics_sink or JsonlMetricsSink(None),
    )


def test_trains_while_the_gate_is_open():
    step_runner = FakeRunner()
    trainer_thread = build(step_runner)
    trainer_thread.start()
    try:
        assert step_runner.entered.wait(timeout=5.0), "trainer never ran"
    finally:
        trainer_thread.stop()
    assert trainer_thread.micro_steps > 0


def test_start_is_idempotent():
    step_runner = FakeRunner()
    trainer_thread = build(step_runner)
    trainer_thread.start()
    trainer_thread.start()
    try:
        assert step_runner.entered.wait(timeout=5.0)
    finally:
        trainer_thread.stop()


def test_stops_when_the_pool_is_thin_without_spinning():
    """A `None` from `train_step` means nothing to train on, and must not become a
    busy loop that burns the idle window it was given.

    The bound is the poll interval: retries are allowed, but only at `poll_ms`.
    """
    step_runner = FakeRunner(thin_after=2)
    duration, poll_ms = 0.3, 20.0
    trainer_thread = build(step_runner, poll_ms=poll_ms)
    trainer_thread.start()
    try:
        time.sleep(duration)
    finally:
        trainer_thread.stop()

    assert trainer_thread.micro_steps == 2
    call_budget = duration / (poll_ms / 1000.0)
    assert step_runner.calls <= call_budget + 6, (
        f"{step_runner.calls} calls in {duration}s at poll_ms={poll_ms} "
        f"(budget {call_budget:.0f}) -- backoff is not being applied"
    )


def test_a_shut_gate_prevents_training_entirely():
    step_runner = FakeRunner()
    trainer_thread = build(step_runner, idle_ms=10_000)
    trainer_thread.idle_gate.note_step(1)
    trainer_thread.start()
    try:
        time.sleep(0.15)
    finally:
        trainer_thread.stop()
    assert step_runner.calls == 0
    assert trainer_thread.micro_steps == 0


def test_micro_step_cap_bounds_one_opening():
    """A cap bounds how long a single opening can hold the GPU even if traffic never
    returns."""
    step_runner = FakeRunner()
    trainer_thread = build(step_runner, max_micro_steps_per_gate=3)
    trainer_thread.start()
    try:
        time.sleep(0.25)
    finally:
        trainer_thread.stop()
    assert trainer_thread.micro_steps >= 3
    assert trainer_thread.idle_gate.as_metrics()["gate/openings"] >= 2


def test_a_raising_train_step_is_contained():
    class Exploding(FakeRunner):
        def train_step(self):
            self.calls += 1
            self.entered.set()
            raise RuntimeError("kaboom")

    step_runner = Exploding()
    trainer_thread = build(step_runner)
    trainer_thread.start()
    try:
        assert step_runner.entered.wait(timeout=5.0)
        time.sleep(0.1)
    finally:
        trainer_thread.stop()
    assert trainer_thread.errors > 0
    assert trainer_thread.micro_steps == 0


def test_errors_reach_the_sink(tmp_path):
    import json

    class Exploding(FakeRunner):
        def train_step(self):
            self.calls += 1
            self.entered.set()
            raise RuntimeError("kaboom")

    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    step_runner = Exploding()
    trainer_thread = build(step_runner, metrics_sink=metrics_sink)
    trainer_thread.start()
    try:
        assert step_runner.entered.wait(timeout=5.0)
        time.sleep(0.05)
    finally:
        trainer_thread.stop()
    metrics_sink.close()

    events = {
        json.loads(line)["event"]
        for line in metrics_sink.path.read_text().splitlines()
    }
    assert "train_error" in events


def test_logged_records_combine_gate_and_runner_metrics(tmp_path):
    import json

    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    step_runner = FakeRunner()
    trainer_thread = build(step_runner, metrics_sink=metrics_sink, log_every=1)
    trainer_thread.start()
    try:
        assert step_runner.entered.wait(timeout=5.0)
        time.sleep(0.05)
    finally:
        trainer_thread.stop()
    metrics_sink.close()

    records = [
        json.loads(line) for line in metrics_sink.path.read_text().splitlines()
    ]
    train_records = [record for record in records if record["event"] == "train"]
    assert train_records
    assert "gate/open_fraction" in train_records[0]
    assert "buffer/rollouts" in train_records[0]


def test_log_every_zero_writes_nothing(tmp_path):
    metrics_sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    step_runner = FakeRunner()
    trainer_thread = build(step_runner, metrics_sink=metrics_sink, log_every=0)
    trainer_thread.start()
    try:
        assert step_runner.entered.wait(timeout=5.0)
        time.sleep(0.05)
    finally:
        trainer_thread.stop()
    metrics_sink.close()
    assert metrics_sink.path.read_text() == ""


def test_stopping_a_thread_that_never_started_is_harmless():
    build(FakeRunner()).stop()
