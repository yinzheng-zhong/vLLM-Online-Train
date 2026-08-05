import math
from collections.abc import Iterator

import torch
from torch import nn

from vllm_online_train.head.head import TrainableDFlashHead
from vllm_online_train.head.layers.norm import RMSNorm


class HeadWeights:
    """The weight lifecycle of a `TrainableDFlashHead`.

    Stateless: every method takes the head it operates on, so one instance serves any
    number of heads.
    """

    def freeze_shared(self, head: TrainableDFlashHead) -> None:
        """Clear `requires_grad` on the tensors borrowed from the target.

        Args:
            head: The head to freeze.
        """
        head.model.embed_tokens.weight.requires_grad_(False)
        head.lm_head.weight.requires_grad_(False)

    def load_shared(
        self,
        head: TrainableDFlashHead,
        embed_tokens: torch.Tensor,
        lm_head: torch.Tensor,
    ) -> None:
        """Copy the target's embedding table and LM head in, frozen.

        `copy_` converts dtype, so this works at whichever precision the shared
        tensors are held.

        Args:
            head: The head to load into.
            embed_tokens: `[vocab, H]` embedding table.
            lm_head: `[vocab, H]` vocabulary projection.
        """
        with torch.no_grad():
            head.model.embed_tokens.weight.copy_(embed_tokens)
            head.lm_head.weight.copy_(lm_head)

    def cast_shared(self, head: TrainableDFlashHead, dtype: torch.dtype) -> None:
        """Hold the borrowed embedding table and LM head at `dtype`.

        Args:
            head: The head to cast.
            dtype: Target precision for the two frozen tensors.
        """
        head.model.embed_tokens.to(dtype)
        head.lm_head.to(dtype)
        self.freeze_shared(head)

    def init_draft(
        self, head: TrainableDFlashHead, generator: torch.Generator | None = None
    ) -> None:
        """Small-init the trained tensors and re-freeze the borrowed ones.

        Args:
            head: The head to initialise.
            generator: Draws the normal samples, for a reproducible checkpoint.
        """
        std = math.sqrt(2.0 / (5.0 * head.arch.hidden_size))
        with torch.no_grad():
            for module in head.model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0.0, std, generator=generator)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, RMSNorm):
                    module.weight.fill_(1.0)
        self.freeze_shared(head)

    def export(
        self, head: TrainableDFlashHead
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield `(name, tensor)` pairs in DFlash checkpoint naming.

        `DFlashQwen3ForCausalLM.load_weights` prefixes everything that is not
        `lm_head` or `d2t` with `"model."`, so the checkpoint form drops the prefix
        `state_dict()` carries.

        Args:
            head: The head to export.

        Yields:
            One `(name, tensor)` pair per parameter and buffer.
        """
        for name, tensor in head.state_dict().items():
            if name.startswith("model."):
                yield name[len("model.") :], tensor
            else:
                yield name, tensor
