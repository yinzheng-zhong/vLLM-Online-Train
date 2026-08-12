"""The capture hook and the collaborators it is wired from."""

from vllm_online_train.config.settings_factory import SettingsFactory
from vllm_online_train.engine.hook.capture_hook import CaptureHook
from vllm_online_train.engine.hook.capture_state import CaptureState
from vllm_online_train.engine.hook.method_patcher import MethodPatcher
from vllm_online_train.engine.hook.runner_guard import RunnerGuard
from vllm_online_train.engine.hook.settings_loader import SettingsLoader
from vllm_online_train.engine.hook.signature_guard import SignatureGuard

capture_hook = CaptureHook(
    MethodPatcher(SignatureGuard()),
    SettingsLoader(SettingsFactory()),
    CaptureState(),
    RunnerGuard(),
)

__all__ = ["capture_hook"]
