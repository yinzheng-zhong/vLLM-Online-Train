import importlib
import os
import threading
from collections.abc import Callable
from typing import Any

from vllm_online_train.engine.hook.settings_loader import SettingsLoader
from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class RunnerGuard:
    MODULE = "vllm.v1.worker.gpu.model_runner"
    """Where vLLM keeps the V2 `GPUModelRunner`."""

    ENV_VAR = "VLLM_USE_V2_MODEL_RUNNER"
    """The engine switch that chooses between the two model runners."""

    def __init__(self) -> None:
        """Reports a V2 model runner, which the capture hook cannot reach.

        The hook wraps `propose_draft_token_ids` on the V1 `GPUModelRunner`. The V2
        runner drafts through a `BaseSpeculator` and never calls that method, so the
        patch installs, fires nothing, and reports nothing for the life of the
        process. This wraps the V2 runner's constructor to say so once.
        """
        self._lock = threading.Lock()
        self._installed = False
        self._original_init: Any = None
        self._reported = False

    @property
    def installed(self) -> bool:
        """Whether the V2 runner's constructor is currently wrapped."""
        return self._installed

    def install(self) -> bool:
        """Wrap the V2 runner's constructor. Idempotent.

        Returns:
            Whether the constructor is now wrapped. `False` when this vLLM ships no
            V2 model runner.
        """
        with self._lock:
            if self._installed:
                return True

            model_runner_class = self._runner_class()
            if model_runner_class is None:
                logger.debug("No V2 model runner in this vLLM; guard not installed")
                return False

            original_init = model_runner_class.__init__
            model_runner_class.__init__ = self._wrap(original_init, self._report)
            self._original_init = original_init
            self._installed = True
            logger.debug("V2 model runner guard installed")
            return True

    def uninstall(self) -> None:
        """Restore the V2 runner's constructor, if one was wrapped."""
        with self._lock:
            if not self._installed:
                return

            model_runner_class = self._runner_class()
            if model_runner_class is not None:
                model_runner_class.__init__ = self._original_init
            self._original_init = None
            self._installed = False
            self._reported = False
            logger.debug("V2 model runner guard uninstalled")

    @classmethod
    def _runner_class(cls) -> Any | None:
        """The V2 `GPUModelRunner`.

        Returns:
            The class, or `None` when this vLLM has no V2 model runner.
        """
        try:
            module = importlib.import_module(cls.MODULE)
        except ImportError:
            return None
        return getattr(module, "GPUModelRunner", None)

    def _report(self) -> None:
        """Log that capture cannot run here, once per process, and only when the
        operator asked for training."""
        if self._reported or not os.environ.get(SettingsLoader.CONFIG_ENV):
            return
        self._reported = True
        logger.error(
            "Online training is configured, but this engine is building the V2 model "
            "runner. The capture hook wraps GPUModelRunner.propose_draft_token_ids on "
            "the V1 runner, which the V2 runner never calls: no rollout will be "
            "captured and no training step will run. Serve with %s=0 to stay on the "
            "V1 runner, or unset %s to serve without online training.",
            self.ENV_VAR,
            SettingsLoader.CONFIG_ENV,
        )

    @staticmethod
    def _wrap(original_init: Any, report: Callable[[], None]) -> Any:
        """Build the replacement constructor.

        Args:
            original_init: The V2 runner's `__init__`.
            report: Called before the delegation.

        Returns:
            A constructor that reports, then delegates with every argument unchanged.
        """

        def runner_init(self, *args: Any, **kwargs: Any) -> None:
            report()
            original_init(self, *args, **kwargs)

        # Standard-library convention: makes `inspect.signature` and
        # `inspect.unwrap` resolve through the replacement to `original_init`.
        runner_init.__wrapped__ = original_init
        return runner_init
