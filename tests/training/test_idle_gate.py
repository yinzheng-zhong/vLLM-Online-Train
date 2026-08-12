"""The idle gate.

Gate-open share under real traffic is the first number to watch -- ahead of the loss,
because if the gate never opens then nothing downstream of it matters. The property
that makes the whole thing shippable is that the gate closes promptly when a step
lands, since a training step straddling a request arrival puts its whole duration
onto that request's latency.
"""

import time

from tests.conftest import make_settings
from vllm_online_train.training.idle_gate import IdleGate


def make_gate(**overrides) -> IdleGate:
    return IdleGate(make_settings(**overrides).gate)


def test_gate_is_shut_immediately_after_a_step():
    idle_gate = make_gate(idle_ms=10_000)
    idle_gate.note_step(128)
    assert not idle_gate.is_open()


def test_gate_opens_once_the_engine_goes_quiet():
    idle_gate = make_gate(idle_ms=10)
    idle_gate.note_step(1)
    assert not idle_gate.is_open()
    time.sleep(0.03)
    assert idle_gate.is_open()


def test_a_new_step_closes_an_open_gate():
    """This is the latency guarantee: the trainer checks between micro-batches and
    must see the gate shut the moment traffic returns."""
    idle_gate = make_gate(idle_ms=10)
    time.sleep(0.03)
    assert idle_gate.is_open()
    idle_gate.note_step(64)
    assert not idle_gate.is_open()


def test_busy_tokens_keeps_training_out_of_a_prefill_gap():
    """A large step suggests more of the same batch is arriving, so the quiet that
    follows it is not really idle."""
    idle_gate = make_gate(idle_ms=1, busy_tokens=100)
    idle_gate.note_step(4096)
    time.sleep(0.02)
    assert not idle_gate.is_open(), "opened in the gap after a big prefill"

    idle_gate.note_step(8)
    time.sleep(0.02)
    assert idle_gate.is_open()


def test_busy_tokens_zero_disables_the_token_check():
    idle_gate = make_gate(idle_ms=1, busy_tokens=0)
    idle_gate.note_step(1_000_000)
    time.sleep(0.02)
    assert idle_gate.is_open()


def test_zero_idle_ms_opens_immediately():
    idle_gate = make_gate(idle_ms=0)
    idle_gate.note_step(1)
    assert idle_gate.is_open()


def test_metrics_report_gate_occupancy_and_step_count():
    idle_gate = make_gate(idle_ms=5)
    idle_gate.note_step(10)
    idle_gate.note_step(20)
    idle_gate.mark_opened()
    time.sleep(0.02)
    metrics = idle_gate.as_metrics()

    assert metrics["engine/steps"] == 2.0
    assert metrics["engine/last_step_tokens"] == 20.0
    assert metrics["gate/openings"] == 1.0
    assert 0.0 < metrics["gate/open_fraction"] <= 1.0
    idle_gate.mark_closed()
    assert idle_gate.as_metrics()["gate/open_seconds"] > 0


def test_open_fraction_never_exceeds_one():
    """One clock reading for both terms: sampling `now` twice lets the open window
    advance past the elapsed window."""
    idle_gate = make_gate(idle_ms=0)
    idle_gate.mark_opened()
    for _ in range(20):
        assert idle_gate.as_metrics()["gate/open_fraction"] <= 1.0


def test_mark_opened_is_idempotent_within_one_opening():
    idle_gate = make_gate(idle_ms=0)
    idle_gate.mark_opened()
    idle_gate.mark_opened()
    assert idle_gate.as_metrics()["gate/openings"] == 1.0


def test_mark_closed_without_an_opening_is_harmless():
    idle_gate = make_gate(idle_ms=0)
    idle_gate.mark_closed()
    assert idle_gate.as_metrics()["gate/open_seconds"] == 0.0
