import math
from collections.abc import Iterator

import torch
from torch import nn

from vllm_online_train.training.head.layers.norm import RMSNorm
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead


class HeadWeights:
    """The weight lifecycle of a `TrainableDFlashHead`.

    Stateless: every method takes the head it operates on, so one instance serves any
    number of heads.
    """

    def freeze_shared(self, draft_head: TrainableDFlashHead) -> None:
        """Clear `requires_grad` on the tensors borrowed from the target.

        Args:
            draft_head: The head to freeze.
        """
        draft_head.model.embed_tokens.weight.requires_grad_(False)
        draft_head.lm_head.weight.requires_grad_(False)

    def load_embedding(
        self, draft_head: TrainableDFlashHead, embedding_weight: torch.Tensor
    ) -> None:
        """Copy the target's embedding table in.

        Args:
            draft_head: The head to load into.
            embedding_weight: `[vocab, H]` embedding table.
        """
        with torch.no_grad():
            draft_head.model.embed_tokens.weight.copy_(embedding_weight)

    def load_lm_head(
        self, draft_head: TrainableDFlashHead, lm_head_weight: torch.Tensor
    ) -> None:
        """Copy the target's vocabulary projection in.

        Args:
            draft_head: The head to load into.
            lm_head_weight: `[vocab, H]` vocabulary projection.
        """
        with torch.no_grad():
            draft_head.lm_head.weight.copy_(lm_head_weight)

    def load_shared(
        self,
        draft_head: TrainableDFlashHead,
        embedding_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> None:
        """Copy the target's embedding table and LM head in, frozen.

        `copy_` converts dtype, so this works at whichever precision the shared
        tensors are held.

        Args:
            draft_head: The head to load into.
            embedding_weight: `[vocab, H]` embedding table.
            lm_head_weight: `[vocab, H]` vocabulary projection.
        """
        self.load_embedding(draft_head, embedding_weight)
        self.load_lm_head(draft_head, lm_head_weight)

    def cast_shared(
        self, draft_head: TrainableDFlashHead, dtype: torch.dtype
    ) -> None:
        """Hold the borrowed embedding table and LM head at `dtype`.

        Args:
            draft_head: The head to cast.
            dtype: Target precision for the two frozen tensors.
        """
        draft_head.model.embed_tokens.to(dtype)
        draft_head.lm_head.to(dtype)
        self.freeze_shared(draft_head)

    def init_draft(
        self,
        draft_head: TrainableDFlashHead,
        generator: torch.Generator | None = None,
    ) -> None:
        """Small-init the trained tensors and re-freeze the borrowed ones.

        Args:
            draft_head: The head to initialise.
            generator: Draws the normal samples, for a reproducible checkpoint.
        """
        init_std = math.sqrt(2.0 / (5.0 * draft_head.head_arch.hidden_size))
        with torch.no_grad():
            for module in draft_head.model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0.0, init_std, generator=generator)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, RMSNorm):
                    module.weight.fill_(1.0)
        self.freeze_shared(draft_head)

    def export(
        self, draft_head: TrainableDFlashHead
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield `(name, tensor)` pairs in DFlash checkpoint naming.

        `DFlashQwen3ForCausalLM.load_weights` prefixes everything that is not
        `lm_head` or `d2t` with `"model."`, so the checkpoint form drops the prefix
        `state_dict()` carries.

        Args:
            draft_head: The head to export.

        Yields:
            One `(name, tensor)` pair per parameter and buffer.
        """
        for name, tensor in draft_head.state_dict().items():
            if name.startswith("model."):
                yield name[len("model.") :], tensor
            else:
                yield name, tensor
