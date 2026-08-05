"""The trainer thread: training in the gaps, and getting out of the way."""

import threading
import time

from tests.conftest import make_settings
from vllm_online_train.train.gate import IdleGate
from vllm_online_train.train.metrics import JsonlMetricsSink
from vllm_online_train.train.thread import TrainerThread


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


def build(runner, sink=None, **overrides) -> TrainerThread:
    flat = {"idle_ms": 1, "poll_ms": 1, "log_every": 1}
    flat.update(overrides)
    settings = make_settings(**flat)
    return TrainerThread(
        IdleGate(settings.gate),
        runner,
        settings.gate,
        settings.bookkeeping,
        sink or JsonlMetricsSink(None),
    )


def test_trains_while_the_gate_is_open():
    runner = FakeRunner()
    thread = build(runner)
    thread.start()
    try:
        assert runner.entered.wait(timeout=5.0), "trainer never ran"
    finally:
        thread.stop()
    assert thread.micro_steps > 0


def test_start_is_idempotent():
    runner = FakeRunner()
    thread = build(runner)
    thread.start()
    thread.start()
    try:
        assert runner.entered.wait(timeout=5.0)
    finally:
        thread.stop()


def test_stops_when_the_pool_is_thin_without_spinning():
    """A `None` from `train_step` means nothing to train on, and must not become a
    busy loop that burns the idle window it was given.

    The bound is the poll interval: retries are allowed, but only at `poll_ms`.
    """
    runner = FakeRunner(thin_after=2)
    duration, poll_ms = 0.3, 20.0
    thread = build(runner, poll_ms=poll_ms)
    thread.start()
    try:
        time.sleep(duration)
    finally:
        thread.stop()

    assert thread.micro_steps == 2
    budget = duration / (poll_ms / 1000.0)
    assert runner.calls <= budget + 6, (
        f"{runner.calls} calls in {duration}s at poll_ms={poll_ms} "
        f"(budget {budget:.0f}) -- backoff is not being applied"
    )


def test_a_shut_gate_prevents_training_entirely():
    runner = FakeRunner()
    thread = build(runner, idle_ms=10_000)
    thread.gate.note_step(1)
    thread.start()
    try:
        time.sleep(0.15)
    finally:
        thread.stop()
    assert runner.calls == 0
    assert thread.micro_steps == 0


def test_micro_step_cap_bounds_one_opening():
    """A cap bounds how long a single opening can hold the GPU even if traffic never
    returns."""
    runner = FakeRunner()
    thread = build(runner, max_micro_steps_per_gate=3)
    thread.start()
    try:
        time.sleep(0.25)
    finally:
        thread.stop()
    assert thread.micro_steps >= 3
    assert thread.gate.as_metrics()["gate/openings"] >= 2


def test_a_raising_train_step_is_contained():
    class Exploding(FakeRunner):
        def train_step(self):
            self.calls += 1
            self.entered.set()
            raise RuntimeError("kaboom")

    runner = Exploding()
    thread = build(runner)
    thread.start()
    try:
        assert runner.entered.wait(timeout=5.0)
        time.sleep(0.1)
    finally:
        thread.stop()
    assert thread.errors > 0
    assert thread.micro_steps == 0


def test_errors_reach_the_sink(tmp_path):
    import json

    class Exploding(FakeRunner):
        def train_step(self):
            self.calls += 1
            self.entered.set()
            raise RuntimeError("kaboom")

    sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    runner = Exploding()
    thread = build(runner, sink=sink)
    thread.start()
    try:
        assert runner.entered.wait(timeout=5.0)
        time.sleep(0.05)
    finally:
        thread.stop()
    sink.close()

    events = {
        json.loads(line)["event"] for line in sink.path.read_text().splitlines()
    }
    assert "train_error" in events


def test_logged_records_combine_gate_and_runner_metrics(tmp_path):
    import json

    sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    runner = FakeRunner()
    thread = build(runner, sink=sink, log_every=1)
    thread.start()
    try:
        assert runner.entered.wait(timeout=5.0)
        time.sleep(0.05)
    finally:
        thread.stop()
    sink.close()

    records = [json.loads(line) for line in sink.path.read_text().splitlines()]
    trained = [r for r in records if r["event"] == "train"]
    assert trained
    assert "gate/open_fraction" in trained[0]
    assert "buffer/rollouts" in trained[0]


def test_log_every_zero_writes_nothing(tmp_path):
    sink = JsonlMetricsSink(tmp_path / "metrics.jsonl")
    runner = FakeRunner()
    thread = build(runner, sink=sink, log_every=0)
    thread.start()
    try:
        assert runner.entered.wait(timeout=5.0)
        time.sleep(0.05)
    finally:
        thread.stop()
    sink.close()
    assert sink.path.read_text() == ""


def test_stopping_a_thread_that_never_started_is_harmless():
    build(FakeRunner()).stop()
