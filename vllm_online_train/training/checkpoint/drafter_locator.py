from typing import Any

from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class DrafterLocator:
    """Finds the live draft module on a model runner."""

    def find(self, runner: Any) -> Any | None:
        """The `DFlashQwen3ForCausalLM` a runner is serving with.

        Args:
            runner: The model runner.

        Returns:
            The draft module, or `None` when the runner has none that can load
            weights.
        """
        drafter = getattr(runner, "drafter", None)
        model = getattr(drafter, "model", None)
        if model is None or not hasattr(model, "load_weights"):
            logger.debug("No loadable drafter model found on runner %r", runner)
            return None
        return model
