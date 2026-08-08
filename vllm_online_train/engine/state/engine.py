import torch
from torch import nn

from vllm_online_train.engine.state.target_locator import TargetLocator
from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class EngineStateProvider:
    def __init__(
        self,
        locator: TargetLocator,
        target_model: nn.Module,
        device: torch.device,
        serving_dtype: torch.dtype,
    ) -> None:
        """The only holder of the target model handle.

        Owns the borrowed embedding table, the borrowed vocabulary projection and the
        target's own distribution. The head borrows through it and the objective scores
        through it, so neither knows where the target lives.

        Serves everything on the engine's device. `MirroredStateProvider` wraps this to
        serve it from a second one.

        Args:
            locator: Finds the borrowed modules on the target.
            target_model: The model the engine is serving.
            device: Where its weights live.
            serving_dtype: The dtype the engine is configured to serve at.

        Raises:
            ValueError: If the model exposes no embedding table or no LM head.
        """
        self._target = target_model
        self._device = device
        self._serving_dtype = serving_dtype
        self._embed, self._lm_head = locator.find(target_model)
        logger.debug(
            "engine state provider bound to %s, serving dtype %s", device, serving_dtype
        )

    @property
    def device(self) -> torch.device:
        """Where the state this serves lives, i.e. the engine's device."""
        return self._device

    @property
    def target_dtype(self) -> torch.dtype:
        """The target's parameter dtype."""
        return self._embed.weight.dtype

    @property
    def serving_dtype(self) -> torch.dtype:
        """The dtype the engine is configured to serve at."""
        return self._serving_dtype

    def embedding_weight(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The target's input embedding table, detached.

        Args:
            device: Where to place the copy. `None` selects the target's device.
            dtype: Precision to hold the copy at. `None` selects fp32.

        Returns:
            `[vocab, hidden]` at `dtype`.
        """
        return self._borrow(self._embed.weight, device, dtype)

    def lm_head_weight(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The target's vocabulary projection, detached.

        Args:
            device: Where to place the copy. `None` selects the target's device.
            dtype: Precision to hold the copy at. `None` selects fp32.

        Returns:
            `[vocab, hidden]` at `dtype`.
        """
        return self._borrow(self._lm_head.weight, device, dtype)

    def _borrow(
        self,
        weight: torch.Tensor,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        """Copy one of the target's parameters out.

        The move and the cast are one step, so a vocabulary-sized parameter is never
        materialised at an intermediate precision. The result is always a copy, never a
        view of the engine's own parameter.

        Args:
            weight: The parameter to borrow.
            device: Destination. `None` selects the target's device.
            dtype: Precision of the copy. `None` selects fp32.

        Returns:
            A detached copy on `device`, at `dtype`.
        """
        moved = weight.detach().to(
            device=device or self._device, dtype=dtype or torch.float32
        )
        # Sharing storage with the parameter means both the move and the cast were
        # no-ops, so the copy has to be taken here.
        if moved.data_ptr() == weight.data_ptr():
            return moved.clone()
        return moved

    def teacher_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states through the target's own vocabulary head.

        This is what makes an exact full-vocabulary teacher affordable: the buffer
        holds `[T, hidden]` and the `[T, vocab]` logits are regenerated here.

        Args:
            hidden: `[N, hidden_size]` post-norm final hidden states.

        Returns:
            `[N, vocab_size]` fp32 logits, computed without gradients.
        """
        return self._target.compute_logits(hidden.to(self.target_dtype)).float()
