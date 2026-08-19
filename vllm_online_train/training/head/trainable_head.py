import torch
from torch import nn

from vllm_online_train.training.collate.replay_batch import ReplayBatch
from vllm_online_train.training.head.arch import DFlashHeadArch
from vllm_online_train.training.head.block_masks import BlockMaskBuilder
from vllm_online_train.training.head.model import DFlashModel


class TrainableDFlashHead(nn.Module):
    def __init__(
        self,
        head_arch: DFlashHeadArch,
        block_mask_builder: BlockMaskBuilder,
        *,
        borrowed_weight_device: torch.device | str | None = None,
    ) -> None:
        """Eager DFlash head with a training-shaped forward.

        The parameter tree mirrors the serving module name for name and shape for shape.
        The borrowed embedding table and LM head are held as frozen parameters; loading,
        casting and exporting them is `HeadWeights`' job.

        Args:
            head_arch: Shape of the head.
            block_mask_builder: Builds the anchored block mask and its position ids.
            borrowed_weight_device: Device holding `embed_tokens` and `lm_head`, the two
                tensors bound from the target. `None` uses the default device; `"meta"`
                gives them a shape but no storage, which suits a head that will only
                have its trained tensors read.
        """
        super().__init__()
        self.head_arch = head_arch
        self.block_mask_builder = block_mask_builder
        self.model = DFlashModel(
            head_arch, borrowed_weight_device=borrowed_weight_device
        )
        self.lm_head = nn.Linear(
            head_arch.hidden_size,
            head_arch.draft_vocab_size,
            bias=False,
            device=borrowed_weight_device,
        )

    @property
    def compute_dtype(self) -> torch.dtype:
        """The precision the trained layers run at."""
        return self.model.fc.weight.dtype

    def trainable_parameters(self) -> list[nn.Parameter]:
        """`fc`, `hidden_norm`, `layers.*` and `norm`."""
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    def project_context(self, target_features: torch.Tensor) -> torch.Tensor:
        """Project captured features into the head's width.

        The same projected tensor supplies keys and values at every depth: the context
        representation does not evolve with the draft's layers, only the block slots
        do.

        Args:
            target_features: `[B, T, num_features * H]`.

        Returns:
            `[B, T, H]`, cast to the compute dtype.

        Raises:
            ValueError: If the feature width is not what this head's `fc` expects.
        """
        expected_width = self.head_arch.feature_width
        if target_features.shape[-1] != expected_width:
            raise ValueError(
                f"target_features last dim is {target_features.shape[-1]}, "
                f"expected {expected_width} "
                f"(num_features={self.head_arch.num_features} x "
                f"hidden_size={self.head_arch.hidden_size}). The capture is reading a "
                "different number of target layers than this head expects."
            )
        cast_features = target_features.to(self.compute_dtype)
        return self.model.hidden_norm(self.model.fc(cast_features))

    def build_block_inputs(
        self, anchor_token_ids: torch.Tensor, block_keep_mask: torch.Tensor
    ) -> torch.Tensor:
        """Embed `[y_anchor, MASK x (K-1)]` for every block.

        Args:
            anchor_token_ids: `[B, A]` token at each anchor.
            block_keep_mask: `[B, A]` False for padding blocks.

        Returns:
            `[B, A*K, H]` embeddings, cast to the compute dtype.
        """
        batch_size, num_blocks = anchor_token_ids.shape
        block_size = self.head_arch.block_size
        slot_token_ids = torch.full(
            (batch_size, num_blocks, block_size),
            self.head_arch.mask_token_id,
            dtype=torch.long,
            device=anchor_token_ids.device,
        )
        # Padding blocks stay all-MASK, so they cannot leak a real token.
        slot_token_ids[:, :, 0] = torch.where(
            block_keep_mask,
            anchor_token_ids,
            torch.full_like(anchor_token_ids, self.head_arch.mask_token_id),
        )
        return self.model.embed_tokens(slot_token_ids.flatten(1)).to(
            self.compute_dtype
        )

    def forward_blocks(
        self,
        *,
        context_features: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        context_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score every anchor in one pass.

        Args:
            context_features: `[B, T, num_features * H]` captured features.
            anchor_positions: `[B, A]` absolute anchor index per block.
            block_keep_mask: `[B, A]` False for padding blocks.
            anchor_token_ids: `[B, A]` token at each anchor.
            context_valid_mask: `[B, T]` padding mask for the context.

        Returns:
            `[B, A, K, H]` pre-LM-head states.
        """
        context_hidden = self.project_context(context_features)
        batch_size, context_len, hidden_size = context_hidden.shape
        num_blocks = anchor_positions.shape[1]
        block_size = self.head_arch.block_size

        block_hidden = self.build_block_inputs(anchor_token_ids, block_keep_mask)
        query_positions = self.block_mask_builder.block_position_ids(
            anchor_positions, block_size
        )
        context_positions = torch.arange(
            context_len, device=context_hidden.device
        ).expand(batch_size, context_len)

        allow_mask = self.block_mask_builder.anchored_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            context_len=context_len,
            block_size=block_size,
            intra_block_causal=False,
            context_valid_mask=context_valid_mask,
        )
        attn_bias = self.block_mask_builder.to_additive_mask(
            allow_mask, context_hidden.dtype
        )

        for layer in self.model.layers:
            block_hidden = layer(
                block_hidden,
                context_hidden,
                query_positions,
                context_positions,
                attn_bias,
            )

        block_hidden = self.model.norm(block_hidden)
        return block_hidden.view(batch_size, num_blocks, block_size, hidden_size)

    def replay(self, replay_batch: ReplayBatch) -> torch.Tensor:
        """Hidden states at every scored position.

        Slot 0 is dropped: it re-feeds the anchor, whose token is already known.

        Args:
            replay_batch: The collated batch to replay.

        Returns:
            `[B, A, K-1, H]`.
        """
        anchor_positions = replay_batch.anchor_positions.clamp_min(0)
        anchor_token_ids = replay_batch.input_ids.gather(1, anchor_positions)
        block_hidden = self.forward_blocks(
            context_features=replay_batch.target_features,
            anchor_positions=anchor_positions,
            block_keep_mask=replay_batch.block_keep_mask,
            anchor_token_ids=anchor_token_ids,
            context_valid_mask=replay_batch.attention_mask,
        )
        return block_hidden[:, :, 1:]

    def project_to_vocab(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project through the borrowed LM head.

        Args:
            hidden_states: `[N, H]`.

        Returns:
            `[N, draft_vocab]`. Gradients flow back through the cast.
        """
        return self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
