import torch
from torch import nn


class EngineStateProvider:
    """The only holder of the target model handle.

    Owns the borrowed embedding table, the borrowed vocabulary projection and the
    target's own distribution. The head borrows through it and the objective scores
    through it, so neither knows where the target lives.

    Serves everything on the engine's device. `MirroredStateProvider` wraps this to
    serve it from a second one.
    """

    def __init__(
        self,
        target_model: nn.Module,
        device: torch.device,
        serving_dtype: torch.dtype,
    ) -> None:
        """
        Args:
            target_model: The model the engine is serving.
            device: Where its weights live.
            serving_dtype: The dtype the engine is configured to serve at.

        Raises:
            ValueError: If the model exposes no embedding table or no LM head.
        """
        self._target = target_model
        self._device = device
        self._serving_dtype = serving_dtype

        embed = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if embed is None or lm_head is None:
            raise ValueError(
                "Online training needs the target's embedding table and LM head, "
                f"but {type(target_model).__name__} exposes "
                f"embed={embed is not None}, lm_head={lm_head is not None}."
            )
        self._embed = embed
        self._lm_head = lm_head

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

    def embedding_weight(self, device: torch.device | None = None) -> torch.Tensor:
        """The target's input embedding table, detached in fp32.

        Args:
            device: Where to place the copy. `None` selects the target's device.

        Returns:
            `[vocab, hidden]` fp32.
        """
        return self._borrow(self._embed.weight, device)

    def lm_head_weight(self, device: torch.device | None = None) -> torch.Tensor:
        """The target's vocabulary projection, detached in fp32.

        Args:
            device: Where to place the copy. `None` selects the target's device.

        Returns:
            `[vocab, hidden]` fp32.
        """
        return self._borrow(self._lm_head.weight, device)

    def _borrow(
        self, weight: torch.Tensor, device: torch.device | None
    ) -> torch.Tensor:
        """Copy one of the target's parameters out, in fp32.

        Moves at the target's own dtype and upcasts on the destination. The result is
        always a copy, never a view of the engine's own parameter.

        Args:
            weight: The parameter to borrow.
            device: Destination. `None` selects the target's device.

        Returns:
            A detached fp32 copy on `device`.
        """
        moved = weight.detach().to(device or self._device)
        if moved.dtype is not torch.float32:
            return moved.float()
        # Sharing storage with the parameter means the move was a no-op and the upcast
        # would be too, so the copy has to be taken here.
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
