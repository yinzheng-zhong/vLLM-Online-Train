import torch

from vllm_online_train.config.shapes import EngineShapes
from vllm_online_train.engine.rollouts.rollout_records import RolloutRecord
from vllm_online_train.logger import init_logger
from vllm_online_train.training.collate.anchor_sampler import AnchorSampler
from vllm_online_train.training.collate.block_builder import BlockBuilder
from vllm_online_train.training.collate.device_transfer import DeviceTransfer
from vllm_online_train.training.collate.replay_batch import BlockRecord, ReplayBatch

logger = init_logger(__name__)


class RolloutCollator:
    def __init__(
        self,
        block_builder: BlockBuilder,
        anchor_sampler: AnchorSampler,
        device_transfer: DeviceTransfer,
        engine_shapes: EngineShapes,
        pad_token_id: int = 0,
    ) -> None:
        """Pads captured rollouts into one `ReplayBatch`.

        Args:
            block_builder: Enumerates candidate anchors per rollout.
            anchor_sampler: Cuts them to the per-sequence cap.
            device_transfer: Moves the padded tensors to the training device.
            engine_shapes: Supplies `D`, the feature width and the storage dtype.
            pad_token_id: Fills padded context and padded label slots.
        """
        self.block_builder = block_builder
        self.anchor_sampler = anchor_sampler
        self.device_transfer = device_transfer
        self.engine_shapes = engine_shapes
        self.pad_token_id = pad_token_id
        logger.debug(
            "rollout collator built: num_draft_tokens=%d pad_token_id=%d",
            engine_shapes.num_draft_tokens,
            pad_token_id,
        )

    def collate(
        self, rollout_records: list[RolloutRecord]
    ) -> ReplayBatch | None:
        """Build one padded batch.

        Args:
            rollout_records: Rollouts drawn from the pool.

        Returns:
            The batch, or `None` when no rollout yields a block.
        """
        usable_rollouts = self._usable(rollout_records)
        if not usable_rollouts:
            return None

        num_draft_tokens = self.engine_shapes.num_draft_tokens
        # Features alone would need only `max_anchor`, since attention never reads at
        # or past a block's own anchor. `final_hidden` is read as far as
        # `max_anchor + D - 1`, so the longer reach wins.
        max_anchor = max(
            block_record.anchor
            for _, block_records in usable_rollouts
            for block_record in block_records
        )
        context_len = max_anchor + num_draft_tokens
        max_blocks = max(
            len(block_records) for _, block_records in usable_rollouts
        )

        context_tensors = self._pad_context(usable_rollouts, context_len)
        block_tensors = self._pad_blocks(
            usable_rollouts, max_blocks, num_draft_tokens
        )
        logger.debug(
            "collated batch: rollouts=%d context_len=%d max_blocks=%d",
            len(usable_rollouts),
            context_len,
            max_blocks,
        )

        send = self.device_transfer.send
        return ReplayBatch(
            input_ids=send(context_tensors["input_ids"]),
            attention_mask=send(context_tensors["attention_mask"]),
            prompt_lengths=send(context_tensors["prompt_lengths"]),
            anchor_positions=send(block_tensors["anchor_positions"]),
            block_keep_mask=send(block_tensors["block_keep_mask"]),
            block_token_ids=send(block_tensors["block_token_ids"]),
            accepted_counts=send(block_tensors["accepted_counts"]),
            target_features=send(context_tensors["target_features"]),
            final_hidden=send(context_tensors["final_hidden"]),
            meta={
                "num_rollouts": len(usable_rollouts),
                "context_len": context_len,
            },
        )

    def _usable(
        self, rollout_records: list[RolloutRecord]
    ) -> list[tuple[RolloutRecord, list[BlockRecord]]]:
        """Pair each rollout with the blocks it contributes, dropping the empty ones."""
        usable_rollouts: list[tuple[RolloutRecord, list[BlockRecord]]] = []
        for rollout_record in rollout_records:
            candidate_records = self.block_builder.build(
                rollout_record,
                num_draft_tokens=self.engine_shapes.num_draft_tokens,
                stride=self.anchor_sampler.stride,
            )
            if not candidate_records:
                continue
            usable_rollouts.append(
                (rollout_record, self.anchor_sampler.select(candidate_records))
            )
        return usable_rollouts

    def _pad_context(
        self,
        usable_rollouts: list[tuple[RolloutRecord, list[BlockRecord]]],
        context_len: int,
    ) -> dict[str, torch.Tensor]:
        """Right-pad the shared context of every row to `context_len`."""
        batch_size = len(usable_rollouts)
        engine_shapes = self.engine_shapes

        input_ids = torch.full(
            (batch_size, context_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, context_len), dtype=torch.long)
        target_features = torch.zeros(
            (batch_size, context_len, engine_shapes.feature_width),
            dtype=engine_shapes.feature_dtype,
        )
        final_hidden = torch.zeros(
            (batch_size, context_len, engine_shapes.hidden_size),
            dtype=engine_shapes.feature_dtype,
        )
        prompt_lengths: list[int] = []

        for row, (rollout_record, _) in enumerate(usable_rollouts):
            length = min(rollout_record.num_tokens, context_len)
            input_ids[row, :length] = rollout_record.token_ids[:length].to(torch.long)
            attention_mask[row, :length] = 1
            target_features[row, :length] = rollout_record.features[:length]
            final_hidden[row, :length] = rollout_record.final_hidden[:length]
            prompt_lengths.append(rollout_record.prompt_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_features": target_features,
            "final_hidden": final_hidden,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        }

    def _pad_blocks(
        self,
        usable_rollouts: list[tuple[RolloutRecord, list[BlockRecord]]],
        max_blocks: int,
        num_draft_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Pad every row to `max_blocks` blocks, marking the filler as not kept."""
        anchor_rows: list[list[int]] = []
        keep_rows: list[list[bool]] = []
        token_rows: list[list[list[int]]] = []
        accepted_rows: list[list[int]] = []
        pad_block = [self.pad_token_id] * num_draft_tokens

        for _, block_records in usable_rollouts:
            slack = max_blocks - len(block_records)
            anchor_rows.append(
                [max(block_record.anchor, 0) for block_record in block_records]
                + [0] * slack
            )
            keep_rows.append([True] * len(block_records) + [False] * slack)
            token_rows.append(
                [
                    list(block_record.draft_token_ids)
                    for block_record in block_records
                ]
                + [pad_block[:] for _ in range(slack)]
            )
            accepted_rows.append(
                [block_record.accepted for block_record in block_records]
                + [0] * slack
            )

        return {
            "anchor_positions": torch.tensor(anchor_rows, dtype=torch.long),
            "block_keep_mask": torch.tensor(keep_rows, dtype=torch.bool),
            "block_token_ids": torch.tensor(token_rows, dtype=torch.long),
            "accepted_counts": torch.tensor(accepted_rows, dtype=torch.long),
        }
