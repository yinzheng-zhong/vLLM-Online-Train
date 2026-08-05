"""Wrapping and unwrapping the private runner method."""

import pytest

from vllm_online_train.hook import capture_hook
from vllm_online_train.hook.patcher import MethodPatcher

signature_guard = capture_hook.patcher.guard


@pytest.fixture
def patcher():
    subject = MethodPatcher(signature_guard)
    yield subject
    subject.uninstall()


def test_install_wraps_and_uninstall_restores(patcher):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.propose_draft_token_ids
    assert patcher.install(lambda *_: None)
    assert patcher.installed
    assert GPUModelRunner.propose_draft_token_ids is not original

    patcher.uninstall()
    assert not patcher.installed
    assert GPUModelRunner.propose_draft_token_ids is original


def test_install_is_idempotent(patcher):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    patcher.install(lambda *_: None)
    wrapped = GPUModelRunner.propose_draft_token_ids
    patcher.install(lambda *_: None)
    assert GPUModelRunner.propose_draft_token_ids is wrapped, "double-wrapped"


def test_uninstalling_twice_is_harmless(patcher):
    patcher.install(lambda *_: None)
    patcher.uninstall()
    patcher.uninstall()


def test_uninstalling_without_installing_is_harmless(patcher):
    patcher.uninstall()


def test_the_wrapper_observes_before_delegating(patcher):
    """`sampled_token_ids` is still the raw padded tensor with `-1` in rejected slots
    at this point, which is what the valid-position rule reads."""
    order: list[str] = []

    def observe(*_args):
        order.append("observe")

    def original(self, *args):
        order.append("original")
        return "result"

    wrapped = MethodPatcher._wrap(original, observe)
    result = wrapped(*range(10))
    assert order == ["observe", "original"]
    assert result == "result"


def test_the_wrapper_passes_every_argument_through(patcher):
    seen = {}

    def original(
        self,
        scheduler_output,
        sampled_token_ids,
        sampling_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        spec_decode_metadata,
        common_attn_metadata,
        slot_mappings,
    ):
        seen.update(locals())
        return None

    wrapped = MethodPatcher._wrap(original, lambda *_: None)
    wrapped("runner", "sched", "sampled", "meta", "hidden", "sample", "aux", "spec",
            "attn", "slots")
    assert seen["hidden_states"] == "hidden"
    assert seen["sample_hidden_states"] == "sample"
    assert seen["slot_mappings"] == "slots"


def test_the_observer_receives_the_capture_arguments(patcher):
    seen: list[tuple] = []
    wrapped = MethodPatcher._wrap(
        lambda *_: None, lambda *args: seen.append(args)
    )
    wrapped("runner", "sched", "sampled", "meta", "hidden", "sample", "aux", "spec",
            "attn", "slots")
    assert seen == [("runner", "sched", "sampled", "hidden", "aux", "spec")]


def test_a_mutated_signature_is_refused(patcher, monkeypatch):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    monkeypatch.setattr(
        GPUModelRunner,
        "propose_draft_token_ids",
        lambda self, a, b: None,
        raising=True,
    )
    with pytest.raises(RuntimeError, match="signature has changed"):
        patcher.install(lambda *_: None)
    assert not patcher.installed
