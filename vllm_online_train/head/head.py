import torch
from torch import nn

from vllm_online_train.collate.batch import ReplayBatch
from vllm_online_train.head.arch import DFlashHeadArch
from vllm_online_train.head.masks import BlockMaskBuilder
from vllm_online_train.head.model import DFlashModel


class TrainableDFlashHead(nn.Module):
    """Eager DFlash head with a training-shaped forward.

    The parameter tree mirrors the serving module name for name and shape for shape.
    The borrowed embedding table and LM head are held as frozen parameters; loading,
    casting and exporting them is `HeadWeights`' job.
    """

    def __init__(self, arch: DFlashHeadArch, masks: BlockMaskBuilder) -> None:
        """
        Args:
            arch: Shape of the head.
            masks: Builds the anchored block mask and its position ids.
        """
        super().__init__()
        self.arch = arch
        self.masks = masks
        self.model = DFlashModel(arch)
        self.lm_head = nn.Linear(arch.hidden_size, arch.draft_vocab_size, bias=False)

    @property
    def compute_dtype(self) -> torch.dtype:
        """The precision the trained layers run at."""
        return self.model.fc.weight.dtype

    def trainable_parameters(self) -> list[nn.Parameter]:
        """`fc`, `hidden_norm`, `layers.*` and `norm`."""
        return [p for p in self.parameters() if p.requires_grad]

    def project_context(self, target_features: torch.Tensor) -> torch.Tensor:
        """Project captured features into the head's width.

        The same projected tensor supplies keys and values at every depth: the context
        representation does not evolve with the draft's layers, only the block slots
        do.

        Args:
            target_features: `[B, T, num_features * H]`.

        Returns:
            `[B, T, H]`.

        Raises:
            ValueError: If the feature width is not what this head's `fc` expects.
        """
        expected = self.arch.feature_width
        if target_features.shape[-1] != expected:
            raise ValueError(
                f"target_features last dim is {target_features.shape[-1]}, "
                f"expected {expected} (num_features={self.arch.num_features} x "
                f"hidden_size={self.arch.hidden_size}). The capture is reading a "
                "different number of target layers than this head expects."
            )
        return self.model.hidden_norm(self.model.fc(target_features))

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
        batch, num_blocks = anchor_token_ids.shape
        k = self.arch.block_size
        ids = torch.full(
            (batch, num_blocks, k),
            self.arch.mask_token_id,
            dtype=torch.long,
            device=anchor_token_ids.device,
        )
        # Padding blocks stay all-MASK, so they cannot leak a real token.
        ids[:, :, 0] = torch.where(
            block_keep_mask,
            anchor_token_ids,
            torch.full_like(anchor_token_ids, self.arch.mask_token_id),
        )
        return self.model.embed_tokens(ids.flatten(1)).to(self.compute_dtype)

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
        ctx_hidden = self.project_context(context_features)
        batch, context_len, hidden = ctx_hidden.shape
        num_blocks = anchor_positions.shape[1]
        k = self.arch.block_size

        block_hidden = self.build_block_inputs(anchor_token_ids, block_keep_mask)
        query_positions = self.masks.block_position_ids(anchor_positions, k)
        context_positions = torch.arange(
            context_len, device=ctx_hidden.device
        ).expand(batch, context_len)

        allow = self.masks.anchored_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            context_len=context_len,
            block_size=k,
            intra_block_causal=False,
            context_valid_mask=context_valid_mask,
        )
        attn_bias = self.masks.to_additive_mask(allow, ctx_hidden.dtype)

        for layer in self.model.layers:
            block_hidden = layer(
                block_hidden,
                ctx_hidden,
                query_positions,
                context_positions,
                attn_bias,
            )

        block_hidden = self.model.norm(block_hidden)
        return block_hidden.view(batch, num_blocks, k, hidden)

    def replay(self, batch: ReplayBatch) -> torch.Tensor:
        """Hidden states at every scored position.

        Slot 0 is dropped: it re-feeds the anchor, whose token is already known.

        Args:
            batch: The collated batch to replay.

        Returns:
            `[B, A, K-1, H]`.
        """
        anchors = batch.anchor_positions.clamp_min(0)
        anchor_tokens = batch.input_ids.gather(1, anchors)
        hidden = self.forward_blocks(
            context_features=batch.target_features,
            anchor_positions=anchors,
            block_keep_mask=batch.block_keep_mask,
            anchor_token_ids=anchor_tokens,
            context_valid_mask=batch.attention_mask,
        )
        return hidden[:, :, 1:]

    def project_to_vocab(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project through the borrowed LM head.

        Args:
            hidden: `[N, H]`.

        Returns:
            `[N, draft_vocab]`. Gradients flow back through the cast.
        """
        return self.lm_head(hidden.to(self.lm_head.weight.dtype))
