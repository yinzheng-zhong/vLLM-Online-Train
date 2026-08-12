import json
from typing import Any

from vllm_online_train.config.settings import OnlineTrainSettings
from vllm_online_train.engine.hook.capture_state import CaptureState
from vllm_online_train.engine.hook.method_patcher import MethodPatcher
from vllm_online_train.engine.hook.runner_guard import RunnerGuard
from vllm_online_train.engine.hook.settings_loader import SettingsLoader
from vllm_online_train.logger import init_logger
from vllm_online_train.step import EngineStep

logger = init_logger(__name__)


class CaptureHook:
    def __init__(
        self,
        method_patcher: MethodPatcher,
        settings_loader: SettingsLoader,
        capture_state: CaptureState,
        runner_guard: RunnerGuard,
    ) -> None:
        """Tees one engine step's training features out of the model runner.

        The wrapped method receives everything the objective needs in a single call,
        once per engine step, at a point where the step's sampling has resolved but the
        next step has not begun. Any exception disables capture for the rest of the
        process rather than reaching the serving path.

        Args:
            method_patcher: Wraps and unwraps the runner method.
            settings_loader: Reads the operator's settings.
            capture_state: Holds the session and the disabled flag.
            runner_guard: Reports a model runner the patch cannot reach.
        """
        self.method_patcher = method_patcher
        self.settings_loader = settings_loader
        self.capture_state = capture_state
        self.runner_guard = runner_guard

    @property
    def installed(self) -> bool:
        """Whether the runner method is currently wrapped."""
        return self.method_patcher.installed

    def install(self) -> bool:
        """Wrap the runner method and guard the runner the patch cannot reach.
        Idempotent.

        Returns:
            Whether the method is now wrapped.

        Raises:
            RuntimeError: If the method's signature does not match the pin.
        """
        self.runner_guard.install()
        return self.method_patcher.install(self.observe)

    def uninstall(self) -> None:
        """Restore the original method and the guard, and reset the capture state."""
        self.method_patcher.uninstall()
        self.runner_guard.uninstall()
        self.capture_state.reset()

    def observe(
        self,
        model_runner: Any,
        scheduler_output: Any,
        sampled_token_ids: Any,
        hidden_states: Any,
        aux_hidden_states: Any,
        spec_decode_metadata: Any,
    ) -> None:
        """Tee one step. Swallows everything; never raises into the serving path.

        Args:
            model_runner: The `GPUModelRunner` the wrapped method was called on.
            scheduler_output: This step's schedule.
            sampled_token_ids: `[num_reqs, num_spec_tokens + 1]`, `-1` where rejected.
            hidden_states: `[>=T, hidden_size]` post-norm final activations.
            aux_hidden_states: Per-feature-layer activations, or `None`.
            spec_decode_metadata: Per-request draft counts, or `None`.
        """
        if self.capture_state.disabled:
            return
        try:
            if self.capture_state.training_session is None and not self._build(
                model_runner
            ):
                return

            training_session = self.capture_state.training_session
            training_session.note_step(scheduler_output.total_num_scheduled_tokens)

            input_ids = getattr(model_runner, "input_ids", None)
            if input_ids is None or aux_hidden_states is None:
                return
            training_session.observe(
                EngineStep(
                    scheduler_output=scheduler_output,
                    input_batch=model_runner.input_batch,
                    input_ids=input_ids.gpu,
                    aux_hidden_states=aux_hidden_states,
                    hidden_states=hidden_states,
                    sampled_token_ids=sampled_token_ids,
                    spec_decode_metadata=spec_decode_metadata,
                )
            )
        except Exception as exc:
            self.capture_state.disable(
                f"capture raised {type(exc).__name__}: {exc}", exc_info=True
            )

    def _build(self, model_runner: Any) -> bool:
        """Assemble the session on the first usable step.

        Deferred here because `register()` receives no arguments: the `VllmConfig` is
        only reachable as `model_runner.vllm_config`.

        Args:
            model_runner: The `GPUModelRunner` the wrapped method was called on.

        Returns:
            Whether a session is now active. Capture is disabled either way when the
            answer is no, so a declined build costs one attribute check per step
            rather than a rebuild attempt.
        """
        online_train_settings = self._read_settings()
        if online_train_settings is None:
            return False

        from vllm_online_train.assembler import SessionAssembler

        training_session = SessionAssembler().build(
            model_runner, online_train_settings
        )
        if training_session is None:
            self.capture_state.disable("the assembler declined to build a session")
            return False
        self.capture_state.activate(training_session)
        logger.debug("Online training session built and activated")
        return True

    def _read_settings(self) -> OnlineTrainSettings | None:
        """Read and validate the operator's settings.

        Returns:
            The settings, or `None` after disabling capture with a reason.
        """
        config_env = self.settings_loader.CONFIG_ENV
        try:
            online_train_settings = self.settings_loader.load()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.capture_state.disable(f"could not read ${config_env}: {exc}")
            return None

        if online_train_settings is None:
            self.capture_state.disable(f"${config_env} is unset")
            return None
        if not online_train_settings.enabled:
            self.capture_state.disable("config sets enabled=false")
            return None
        return online_train_settings
