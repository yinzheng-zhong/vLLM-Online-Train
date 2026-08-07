from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)


class TargetLocator:
    """Finds the modules the head borrows weights from on a vLLM target model.

    The embedding table and the vocabulary projection are reached by attribute path,
    on the model itself and on its language model when it has a distinct one.
    """

    EMBED_PATHS: tuple[str, ...] = ("model.embed_tokens", "embed_tokens")
    LM_HEAD_PATHS: tuple[str, ...] = ("lm_head",)

    def find(self, target_model: nn.Module) -> tuple[nn.Module, nn.Module]:
        """The target's input embedding table and its vocabulary projection.

        Args:
            target_model: The model the engine is serving.

        Returns:
            `(embed_tokens, lm_head)`, both carrying a `weight` tensor.

        Raises:
            ValueError: If either module is not reachable.
        """
        roots = self._roots(target_model)
        embed = self._resolve(roots, self.EMBED_PATHS, "embedding table")
        lm_head = self._resolve(roots, self.LM_HEAD_PATHS, "LM head")
        return embed, lm_head

    @staticmethod
    def _roots(target_model: nn.Module) -> tuple[nn.Module, ...]:
        """The modules to search, outermost first.

        Args:
            target_model: The model the engine is serving.

        Returns:
            `target_model`, followed by the language model it delegates text
            generation to when that is a separate module.
        """
        accessor = getattr(target_model, "get_language_model", None)
        if accessor is None:
            return (target_model,)
        try:
            language_model = accessor()
        except (NotImplementedError, AttributeError) as exc:
            logger.debug("%s exposes no language model: %s", type(target_model), exc)
            return (target_model,)
        if language_model is None or language_model is target_model:
            return (target_model,)
        logger.debug(
            "searching %s and its language model %s",
            type(target_model).__name__,
            type(language_model).__name__,
        )
        return (target_model, language_model)

    def _resolve(
        self, roots: tuple[nn.Module, ...], paths: tuple[str, ...], label: str
    ) -> nn.Module:
        """The first module found at any of `paths` under any of `roots`.

        Args:
            roots: Modules to search, in order.
            paths: Dotted attribute paths to try under each root, in order.
            label: What is being looked for, for the error message.

        Returns:
            The module, which carries a `weight` tensor.

        Raises:
            ValueError: If no path resolves to a module with a `weight` tensor.
        """
        for root in roots:
            for path in paths:
                module = self._walk(root, path)
                if isinstance(getattr(module, "weight", None), torch.Tensor):
                    return module
        searched = ", ".join(paths)
        raise ValueError(
            f"Online training needs the target's {label}, but neither "
            f"{' nor '.join(type(root).__name__ for root in roots)} carries a weight "
            f"at any of: {searched}."
        )

    @staticmethod
    def _walk(root: nn.Module, path: str) -> Any | None:
        """Follow a dotted attribute path.

        Args:
            root: Where to start.
            path: Dotted attribute names.

        Returns:
            What the path points at, or `None` if any step is missing.
        """
        current: Any = root
        for name in path.split("."):
            current = getattr(current, name, None)
            if current is None:
                return None
        return current
