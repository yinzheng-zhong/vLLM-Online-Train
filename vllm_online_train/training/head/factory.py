import torch

from vllm_online_train.logger import init_logger
from vllm_online_train.training.head.arch import DFlashHeadArch
from vllm_online_train.training.head.block_masks import BlockMaskBuilder
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead
from vllm_online_train.training.head.weights import HeadWeights

logger = init_logger(__name__)


class HeadFactory:
    def __init__(self, masks: BlockMaskBuilder, weights: HeadWeights) -> None:
        """Builds a `TrainableDFlashHead` that is initialised and correctly frozen.

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
        logger.debug(
            "Built head: num_layers=%d hidden_size=%d block_size=%d "
            "draft_vocab_size=%d",
            arch.num_layers,
            arch.hidden_size,
            arch.block_size,
            arch.draft_vocab_size,
        )
        return head
