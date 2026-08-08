from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class StateProvider(Protocol):
    """Serves the state the draft head borrows from one target model.

    The only holder of the target handle. Everything that needs the embedding table,
    the vocabulary projection or the target's own distribution asks this instead of
    reaching for the model.
    """

    @property
    def device(self) -> torch.device:
        """Where the state this serves lives, and so where the head trains."""
        ...

    @property
    def target_dtype(self) -> torch.dtype:
        """The target's parameter dtype."""
        ...

    @property
    def serving_dtype(self) -> torch.dtype:
        """The dtype the engine is configured to serve at."""
        ...

    def embedding_weight(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The target's input embedding table, `[vocab, hidden]`, detached.

        Args:
            device: Where to place the copy. `None` selects `device`.
            dtype: Precision to hold the copy at. `None` selects fp32.
        """
        ...

    def lm_head_weight(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The target's vocabulary projection, `[vocab, hidden]`, detached.

        Args:
            device: Where to place the copy. `None` selects `device`.
            dtype: Precision to hold the copy at. `None` selects fp32.
        """
        ...

    def teacher_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states through the target's own vocabulary head.

        Args:
            hidden: `[N, hidden_size]` post-norm final hidden states, on `device`.

        Returns:
            `[N, vocab_size]` fp32 logits, computed without gradients.
        """
        ...
