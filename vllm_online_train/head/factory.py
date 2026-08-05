import torch

from vllm_online_train.head.arch import DFlashHeadArch
from vllm_online_train.head.head import TrainableDFlashHead
from vllm_online_train.head.masks import BlockMaskBuilder
from vllm_online_train.head.weights import HeadWeights


class HeadFactory:
    """Builds a `TrainableDFlashHead` that is initialised and correctly frozen."""

    def __init__(self, masks: BlockMaskBuilder, weights: HeadWeights) -> None:
        """
        Args:
            masks: Injected into every head this factory builds.
            weights: Performs the initialisation and the freeze.
        """
        self.masks = masks
        self.weights = weights

    def create(
        self,
        arch: DFlashHeadArch,
        *,
        generator: torch.Generator | None = None,
    ) -> TrainableDFlashHead:
        """Build a head with small-init trained tensors and frozen borrowed ones.

        Args:
            arch: Shape of the head.
            generator: Draws the initial weights.

        Returns:
            The head, on the default device at the default dtype.
        """
        head = TrainableDFlashHead(arch, self.masks)
        self.weights.init_draft(head, generator=generator)
        return head
