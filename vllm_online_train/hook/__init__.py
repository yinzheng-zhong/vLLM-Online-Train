"""The capture hook and the collaborators it is wired from."""

from vllm_online_train.config.factory import SettingsFactory
from vllm_online_train.hook.guard import SignatureGuard
from vllm_online_train.hook.hook import CaptureHook
from vllm_online_train.hook.loader import SettingsLoader
from vllm_online_train.hook.patcher import MethodPatcher
from vllm_online_train.hook.state import CaptureState

capture_hook = CaptureHook(
    MethodPatcher(SignatureGuard()),
    SettingsLoader(SettingsFactory()),
    CaptureState(),
)

__all__ = ["capture_hook"]
