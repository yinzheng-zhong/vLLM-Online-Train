import torch

from vllm_online_train.logger import init_logger
from vllm_online_train.training.head.arch import DFlashHeadArch
from vllm_online_train.training.head.block_masks import BlockMaskBuilder
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead
from vllm_online_train.training.head.weights import HeadWeights

logger = init_logger(__name__)


class HeadFactory:
    def __init__(
        self, block_mask_builder: BlockMaskBuilder, head_weights: HeadWeights
    ) -> None:
        """Builds a `TrainableDFlashHead` that is initialised and correctly frozen.

        Args:
            block_mask_builder: Injected into every head this factory builds.
            head_weights: Performs the initialisation and the freeze.
        """
        self.block_mask_builder = block_mask_builder
        self.head_weights = head_weights

    def create(
        self,
        head_arch: DFlashHeadArch,
        *,
        generator: torch.Generator | None = None,
    ) -> TrainableDFlashHead:
        """Build a head with small-init trained tensors and frozen borrowed ones.

        Args:
            head_arch: Shape of the head.
            generator: Draws the initial weights.

        Returns:
            The head, on the default device at the default dtype.
        """
        draft_head = TrainableDFlashHead(head_arch, self.block_mask_builder)
        self.head_weights.init_draft(draft_head, generator=generator)
        logger.debug(
            "Built head: num_layers=%d hidden_size=%d block_size=%d "
            "draft_vocab_size=%d",
            head_arch.num_layers,
            head_arch.hidden_size,
            head_arch.block_size,
            head_arch.draft_vocab_size,
        )
        return draft_head
