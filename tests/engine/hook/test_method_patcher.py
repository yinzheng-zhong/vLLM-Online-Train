"""Wrapping and unwrapping the private runner method."""

import pytest

from vllm_online_train.engine.hook import capture_hook
from vllm_online_train.engine.hook.method_patcher import MethodPatcher

signature_guard = capture_hook.method_patcher.signature_guard


@pytest.fixture
def method_patcher():
    patcher = MethodPatcher(signature_guard)
    yield patcher
    patcher.uninstall()


def test_install_wraps_and_uninstall_restores(method_patcher):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_method = GPUModelRunner.propose_draft_token_ids
    assert method_patcher.install(lambda *_: None)
    assert method_patcher.installed
    assert GPUModelRunner.propose_draft_token_ids is not original_method

    method_patcher.uninstall()
    assert not method_patcher.installed
    assert GPUModelRunner.propose_draft_token_ids is original_method


def test_install_is_idempotent(method_patcher):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    method_patcher.install(lambda *_: None)
    wrapped_method = GPUModelRunner.propose_draft_token_ids
    method_patcher.install(lambda *_: None)
    assert (
        GPUModelRunner.propose_draft_token_ids is wrapped_method
    ), "double-wrapped"


def test_uninstalling_twice_is_harmless(method_patcher):
    method_patcher.install(lambda *_: None)
    method_patcher.uninstall()
    method_patcher.uninstall()


def test_uninstalling_without_installing_is_harmless(method_patcher):
    method_patcher.uninstall()


def test_the_wrapper_observes_before_delegating(method_patcher):
    """`sampled_token_ids` is still the raw padded tensor with `-1` in rejected slots
    at this point, which is what the valid-position rule reads."""
    call_order: list[str] = []

    def observe(*_args):
        call_order.append("observe")

    def original_method(self, *args):
        call_order.append("original")
        return "result"

    wrapped_method = MethodPatcher._wrap(original_method, observe)
    result = wrapped_method(*range(10))
    assert call_order == ["observe", "original"]
    assert result == "result"


def test_the_wrapper_passes_every_argument_through(method_patcher):
    seen = {}

    def original_method(
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

    wrapped_method = MethodPatcher._wrap(original_method, lambda *_: None)
    wrapped_method(
        "runner", "sched", "sampled", "meta", "hidden", "sample", "aux", "spec",
        "attn", "slots",
    )
    assert seen["hidden_states"] == "hidden"
    assert seen["sample_hidden_states"] == "sample"
    assert seen["slot_mappings"] == "slots"


def test_the_observer_receives_the_capture_arguments(method_patcher):
    seen: list[tuple] = []
    wrapped_method = MethodPatcher._wrap(
        lambda *_: None, lambda *args: seen.append(args)
    )
    wrapped_method(
        "runner", "sched", "sampled", "meta", "hidden", "sample", "aux", "spec",
        "attn", "slots",
    )
    assert seen == [("runner", "sched", "sampled", "hidden", "aux", "spec")]


def test_a_mutated_signature_is_refused(method_patcher, monkeypatch):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    monkeypatch.setattr(
        GPUModelRunner,
        "propose_draft_token_ids",
        lambda self, a, b: None,
        raising=True,
    )
    with pytest.raises(RuntimeError, match="signature has changed"):
        method_patcher.install(lambda *_: None)
    assert not method_patcher.installed
