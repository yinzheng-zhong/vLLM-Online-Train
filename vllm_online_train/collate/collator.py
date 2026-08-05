import torch
from vllm.logger import init_logger

from vllm_online_train.capture.records import RolloutRecord
from vllm_online_train.collate.anchors import AnchorSampler
from vllm_online_train.collate.batch import BlockRecord, ReplayBatch
from vllm_online_train.collate.blocks import BlockBuilder
from vllm_online_train.collate.transfer import DeviceTransfer
from vllm_online_train.config.shapes import EngineShapes

logger = init_logger(__name__)


class RolloutCollator:
    def __init__(
        self,
        blocks: BlockBuilder,
        anchors: AnchorSampler,
        transfer: DeviceTransfer,
        shapes: EngineShapes,
        pad_token_id: int = 0,
    ) -> None:
        """Pads captured rollouts into one `ReplayBatch`.

        Args:
            blocks: Enumerates candidate anchors per rollout.
            anchors: Cuts them to the per-sequence cap.
            transfer: Moves the padded tensors to the training device.
            shapes: Supplies `D`, the feature width and the storage dtype.
            pad_token_id: Fills padded context and padded label slots.
        """
        self.blocks = blocks
        self.anchors = anchors
        self.transfer = transfer
        self.shapes = shapes
        self.pad_token_id = pad_token_id
        logger.debug(
            "rollout collator built: num_draft_tokens=%d pad_token_id=%d",
            shapes.num_draft_tokens,
            pad_token_id,
        )

    def collate(self, records: list[RolloutRecord]) -> ReplayBatch | None:
        """Build one padded batch.

        Args:
            records: Rollouts drawn from the pool.

        Returns:
            The batch, or `None` when no rollout yields a block.
        """
        usable = self._usable(records)
        if not usable:
            return None

        num_draft_tokens = self.shapes.num_draft_tokens
        # Features alone would need only `max_anchor`, since attention never reads at
        # or past a block's own anchor. `final_hidden` is read as far as
        # `max_anchor + D - 1`, so the longer reach wins.
        max_anchor = max(block.anchor for _, blocks in usable for block in blocks)
        context_len = max_anchor + num_draft_tokens
        max_blocks = max(len(blocks) for _, blocks in usable)

        context = self._pad_context(usable, context_len)
        blocks = self._pad_blocks(usable, max_blocks, num_draft_tokens)
        logger.debug(
            "collated batch: rollouts=%d context_len=%d max_blocks=%d",
            len(usable),
            context_len,
            max_blocks,
        )

        send = self.transfer.send
        return ReplayBatch(
            input_ids=send(context["input_ids"]),
            attention_mask=send(context["attention_mask"]),
            prompt_lengths=send(context["prompt_lengths"]),
            anchor_positions=send(blocks["anchor_positions"]),
            block_keep_mask=send(blocks["block_keep_mask"]),
            block_token_ids=send(blocks["block_token_ids"]),
            accepted_counts=send(blocks["accepted_counts"]),
            target_features=send(context["target_features"]),
            final_hidden=send(context["final_hidden"]),
            meta={"num_rollouts": len(usable), "context_len": context_len},
        )

    def _usable(
        self, records: list[RolloutRecord]
    ) -> list[tuple[RolloutRecord, list[BlockRecord]]]:
        """Pair each rollout with the blocks it contributes, dropping the empty ones."""
        usable: list[tuple[RolloutRecord, list[BlockRecord]]] = []
        for record in records:
            candidates = self.blocks.build(
                record,
                num_draft_tokens=self.shapes.num_draft_tokens,
                stride=self.anchors.stride,
            )
            if not candidates:
                continue
            usable.append((record, self.anchors.select(candidates)))
        return usable

    def _pad_context(
        self,
        usable: list[tuple[RolloutRecord, list[BlockRecord]]],
        context_len: int,
    ) -> dict[str, torch.Tensor]:
        """Right-pad the shared context of every row to `context_len`."""
        batch_size = len(usable)
        shapes = self.shapes

        input_ids = torch.full(
            (batch_size, context_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, context_len), dtype=torch.long)
        target_features = torch.zeros(
            (batch_size, context_len, shapes.feature_width), dtype=shapes.feature_dtype
        )
        final_hidden = torch.zeros(
            (batch_size, context_len, shapes.hidden_size), dtype=shapes.feature_dtype
        )
        prompt_lengths: list[int] = []

        for row, (record, _) in enumerate(usable):
            length = min(record.num_tokens, context_len)
            input_ids[row, :length] = record.token_ids[:length].to(torch.long)
            attention_mask[row, :length] = 1
            target_features[row, :length] = record.features[:length]
            final_hidden[row, :length] = record.final_hidden[:length]
            prompt_lengths.append(record.prompt_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_features": target_features,
            "final_hidden": final_hidden,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        }

    def _pad_blocks(
        self,
        usable: list[tuple[RolloutRecord, list[BlockRecord]]],
        max_blocks: int,
        num_draft_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Pad every row to `max_blocks` blocks, marking the filler as not kept."""
        anchor_rows: list[list[int]] = []
        keep_rows: list[list[bool]] = []
        token_rows: list[list[list[int]]] = []
        accepted_rows: list[list[int]] = []
        pad_block = [self.pad_token_id] * num_draft_tokens

        for _, blocks in usable:
            slack = max_blocks - len(blocks)
            anchor_rows.append([max(block.anchor, 0) for block in blocks] + [0] * slack)
            keep_rows.append([True] * len(blocks) + [False] * slack)
            token_rows.append(
                [list(block.draft_token_ids) for block in blocks]
                + [pad_block[:] for _ in range(slack)]
            )
            accepted_rows.append([block.accepted for block in blocks] + [0] * slack)

        return {
            "anchor_positions": torch.tensor(anchor_rows, dtype=torch.long),
            "block_keep_mask": torch.tensor(keep_rows, dtype=torch.bool),
            "block_token_ids": torch.tensor(token_rows, dtype=torch.long),
            "accepted_counts": torch.tensor(accepted_rows, dtype=torch.long),
        }
