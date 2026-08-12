from typing import Any

from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class DrafterLocator:
    """Finds the live draft module on a model runner."""

    def find(self, model_runner: Any) -> Any | None:
        """The `DFlashQwen3ForCausalLM` a runner is serving with.

        Args:
            model_runner: The model runner.

        Returns:
            The draft module, or `None` when the runner has none that can load
            weights.
        """
        drafter = getattr(model_runner, "drafter", None)
        drafter_model = getattr(drafter, "model", None)
        if drafter_model is None or not hasattr(drafter_model, "load_weights"):
            logger.debug(
                "No loadable drafter model found on runner %r", model_runner
            )
            return None
        return drafter_model
