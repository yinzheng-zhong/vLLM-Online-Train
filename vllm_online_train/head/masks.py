import torch


class BlockMaskBuilder:
    """Builds the attention geometry of anchored block replay.

    A rollout is laid out as a shared context followed by every draft block:

    ```
    index:  0 .. T-1              T .. T+K-1    T+K .. T+2K-1   ...
    tokens: [ context (shared) ]  [ block 0 ]   [ block 1 ]     ...
    ```

    Block `m` is re-anchored: slot 0 re-feeds the anchor token `y[a_m]` and its
    position ids run `a_m .. a_m + K - 1`. The context rule is `kv < a_m`, strictly,
    because the anchor is re-fed as slot 0.
    """

    def anchored_block_mask(
        self,
        *,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        context_len: int,
        block_size: int,
        intra_block_causal: bool,
        context_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a boolean "may attend" mask.

        Args:
            anchor_positions: `[B, A]` absolute anchor index per block.
            block_keep_mask: `[B, A]` False for padding blocks.
            context_len: `T`, the shared context length.
            block_size: `K`, query slots per block including the anchor slot.
            intra_block_causal: True for autoregressive consumers, False for DFlash's
                bidirectional block.
            context_valid_mask: `[B, T]` padding mask for the context.

        Returns:
            `[B, 1, A*K, T + A*K]` bool; True means "may attend".

        Raises:
            ValueError: If `anchor_positions` is not 2-D or does not match
                `block_keep_mask` in shape.
        """
        self._check_shapes(anchor_positions, block_keep_mask)

        device = anchor_positions.device
        batch_size, num_blocks = anchor_positions.shape
        q_len = num_blocks * block_size
        kv_len = context_len + q_len

        q_idx = torch.arange(q_len, device=device).view(1, -1, 1)
        kv_idx = torch.arange(kv_len, device=device).view(1, 1, -1)

        q_block = q_idx // block_size
        q_slot = q_idx % block_size
        block_of_query = q_block.expand(batch_size, -1, 1).squeeze(-1)

        # Context region: everything strictly before this block's own anchor.
        anchor_per_query = anchor_positions.gather(1, block_of_query).unsqueeze(-1)
        allow_context = (kv_idx < context_len) & (kv_idx < anchor_per_query)

        if context_valid_mask is not None:
            padded = context_valid_mask.bool().unsqueeze(1)
            padded = torch.nn.functional.pad(padded, (0, q_len), value=True)
            allow_context = allow_context & padded

        # Draft region: only within the same block.
        is_draft = kv_idx >= context_len
        kv_block = (kv_idx - context_len) // block_size
        kv_slot = (kv_idx - context_len) % block_size
        same_block = is_draft & (kv_block == q_block)
        allow_draft = (
            same_block & (kv_slot <= q_slot) if intra_block_causal else same_block
        )

        allow = allow_context | allow_draft

        keep = block_keep_mask.gather(1, block_of_query).unsqueeze(-1)
        allow = allow & keep

        # Reopens each query's own slot after closing padding blocks, so no row is
        # fully masked: a softmax over an all -inf row yields NaN.
        diagonal = kv_idx == (context_len + q_idx)
        allow = allow | diagonal.expand(batch_size, -1, -1)

        return allow.unsqueeze(1)

    def to_additive_mask(
        self, allow: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert a boolean "may attend" mask into an additive attention bias.

        Args:
            allow: Bool tensor; True means "may attend".
            dtype: Bias dtype; disallowed entries become its minimum.

        Returns:
            A tensor of `allow`'s shape, 0 where allowed.
        """
        bias = torch.zeros(allow.shape, dtype=dtype, device=allow.device)
        return bias.masked_fill(~allow, torch.finfo(dtype).min)

    def block_position_ids(
        self, anchor_positions: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Position ids for every slot of every block.

        Args:
            anchor_positions: `[B, A]` absolute anchor index per block.
            block_size: Slots per block.

        Returns:
            `[B, A * block_size]` running `a_m, a_m+1, ..., a_m+block_size-1`.
        """
        offsets = torch.arange(block_size, device=anchor_positions.device).view(
            1, 1, -1
        )
        return (anchor_positions.unsqueeze(-1) + offsets).flatten(1)

    @staticmethod
    def _check_shapes(
        anchor_positions: torch.Tensor, block_keep_mask: torch.Tensor
    ) -> None:
        """Reject anchor and keep tensors that cannot describe the same blocks.

        Raises:
            ValueError: If `anchor_positions` is not `[B, A]` or the two disagree.
        """
        if anchor_positions.ndim != 2:
            raise ValueError(
                f"anchor_positions must be [B, A], got {tuple(anchor_positions.shape)}"
            )
        if anchor_positions.shape != block_keep_mask.shape:
            raise ValueError(
                "anchor_positions and block_keep_mask must have the same shape, got "
                f"{tuple(anchor_positions.shape)} vs {tuple(block_keep_mask.shape)}"
            )
