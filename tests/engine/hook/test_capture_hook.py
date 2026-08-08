"""Plugin and patch integrity.

We wrap a private vLLM method, so the two failure modes that matter are: the
signature drifts and we silently capture the wrong tensor, and capture raises and
takes the engine down with it.
"""

import json
from types import SimpleNamespace

import pytest

from vllm_online_train.engine.hook import capture_hook

capture_state = capture_hook.state
signature_guard = capture_hook.patcher.guard

CONFIG_ENV = capture_hook.loader.CONFIG_ENV


@pytest.fixture(autouse=True)
def clean_hook_state():
    """Every test starts unpatched and ends unpatched."""
    capture_hook.uninstall()
    yield
    capture_hook.uninstall()


def test_accepts_the_real_runner_method():
    """The pin must match the vLLM actually installed, or nothing else is valid."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    signature_guard.check(GPUModelRunner.propose_draft_token_ids)


def test_rejects_a_reordered_signature():
    """Reordering is the dangerous drift: every argument still binds, and capture
    would read `hidden_states` out of whatever now sits in that position."""

    def mutated(
        self,
        scheduler_output,
        sampled_token_ids,
        sampling_metadata,
        sample_hidden_states,  # swapped with hidden_states
        hidden_states,
        aux_hidden_states,
        spec_decode_metadata,
        common_attn_metadata,
        slot_mappings,
    ):
        return None

    with pytest.raises(RuntimeError, match="signature has changed"):
        signature_guard.check(mutated)


def test_rejects_a_renamed_or_dropped_parameter():
    def mutated(
        self,
        scheduler_output,
        sampled_token_ids,
        sampling_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        spec_decode_metadata,
        common_attn_metadata,
    ):
        return None

    with pytest.raises(RuntimeError, match="signature has changed"):
        signature_guard.check(mutated)


def test_refuses_to_install_over_a_mutated_method(monkeypatch):
    """A loud startup failure, not a silent bad capture."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    def mutated(self, scheduler_output, sampled_token_ids):
        return None

    monkeypatch.setattr(
        GPUModelRunner, "propose_draft_token_ids", mutated, raising=True
    )
    with pytest.raises(RuntimeError, match="signature has changed"):
        capture_hook.install()
    assert not capture_hook.installed


def test_install_is_idempotent_and_reversible():
    """`register()` runs in `arg_utils`, `engine/core`, `worker_base` and
    `models/registry`, so it must be safe to call repeatedly."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.propose_draft_token_ids
    assert capture_hook.install()
    patched = GPUModelRunner.propose_draft_token_ids
    assert patched is not original

    assert capture_hook.install()
    assert GPUModelRunner.propose_draft_token_ids is patched, "double-wrapped"

    capture_hook.uninstall()
    assert GPUModelRunner.propose_draft_token_ids is original
    assert not capture_hook.installed


def test_register_is_reentrant():
    import vllm_online_train

    vllm_online_train.register()
    vllm_online_train.register()
    vllm_online_train.register()
    assert capture_hook.installed


def test_register_does_not_import_the_training_stack():
    """`register()` is called in several processes that never build a runner, so it
    must not pay for torch's training path to find that out."""
    import subprocess
    import sys

    script = (
        "import sys, vllm_online_train; vllm_online_train.register();"
        "heavy = [m for m in sys.modules if m.startswith("
        "('vllm_online_train.assembler', 'vllm_online_train.training',"
        " 'vllm_online_train.engine.capture',"
        " 'vllm_online_train.engine.state.engine'))];"
        "print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    # `register()` logs its banner to the same stream, so only the last line is the
    # script's own output.
    assert result.stdout.strip().splitlines()[-1] == "[]", result.stdout


def test_wrapper_keeps_the_signature_it_asserted():
    """The wrapper is itself subject to the pin, so a future patch of the patch
    cannot drift from what the guard describes."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    capture_hook.install()
    signature_guard.check(GPUModelRunner.propose_draft_token_ids)


def test_the_wrapper_exposes_the_original():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.propose_draft_token_ids
    capture_hook.install()
    assert GPUModelRunner.propose_draft_token_ids.__wrapped__ is original


def runner(method: str = "dflash", **kwargs) -> SimpleNamespace:
    spec = SimpleNamespace(method=method, use_dflash=lambda: method == "dflash")
    return SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=spec),
        input_batch=SimpleNamespace(req_ids=[]),
        **kwargs,
    )


def observe(target) -> None:
    capture_hook.observe(
        target,
        SimpleNamespace(total_num_scheduled_tokens=4),
        None,
        None,
        None,
        None,
    )


def test_unset_config_disables_on_a_dflash_engine(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    observe(runner())
    assert capture_state.disabled
    assert CONFIG_ENV in capture_state.reason


def test_a_disabled_config_disables_the_hook(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"enabled": False}))
    observe(runner())
    assert capture_state.disabled
    assert "enabled=false" in capture_state.reason


def test_an_unreadable_config_disables_without_raising(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, "{not json")
    observe(runner())
    assert capture_state.disabled
    assert CONFIG_ENV in capture_state.reason


def test_a_declining_assembler_disables(monkeypatch):
    """A non-DFlash engine or a runner with no config makes the assembler decline,
    and the hook must stop asking rather than retry every step."""
    monkeypatch.setenv(CONFIG_ENV, json.dumps({}))
    monkeypatch.setattr(
        "vllm_online_train.assembler.SessionAssembler",
        lambda: SimpleNamespace(build=lambda *_: None),
    )
    observe(runner(method="eagle3"))
    assert capture_state.disabled
    assert "declined" in capture_state.reason


def test_a_raising_observe_disables_capture_instead_of_the_engine(monkeypatch):
    """A training feature must never be able to take down serving."""
    monkeypatch.setenv(CONFIG_ENV, json.dumps({}))

    class Exploding:
        def note_step(self, _tokens):
            raise RuntimeError("boom")

        def stop(self):
            pass

    capture_state.activate(Exploding())
    observe(runner())
    assert capture_state.disabled
    assert "boom" in capture_state.reason

    # And it stays disabled: one log, then a cheap flag check per step.
    observe(runner())


def test_disabled_state_short_circuits():
    capture_state.disable("test")
    calls = []

    class Counting:
        def note_step(self, tokens):
            calls.append(tokens)

        def stop(self):
            pass

    capture_state.session = Counting()
    observe(runner())
    assert calls == []


def test_disable_stops_the_session():
    stopped = []

    class Session:
        def note_step(self, tokens):
            pass

        def stop(self):
            stopped.append(True)

    capture_state.activate(Session())
    capture_state.disable("test")
    assert stopped == [True]
    assert capture_state.session is None


def test_disable_is_idempotent():
    capture_state.disable("first")
    capture_state.disable("second")
    assert capture_state.reason == "first"
