from vllm_online_train.engine.rollouts.rollout_records import RolloutRecord
from vllm_online_train.training.collate.replay_batch import BlockRecord


class BlockBuilder:
    """Cuts a captured rollout into fixed-stride, fully-accepted draft blocks."""

    def build(
        self,
        rollout_record: RolloutRecord,
        *,
        num_draft_tokens: int,
        stride: int,
    ) -> list[BlockRecord]:
        """Enumerate every anchor a rollout supports.

        Anchors start at `prompt_len - 1` and stop where a full block of labels no
        longer fits. Every block is labelled fully accepted, the labels being the
        target's own continuation.

        Args:
            rollout_record: The finished rollout.
            num_draft_tokens: `D`, labels per block.
            stride: Spacing between candidate anchors.

        Returns:
            The blocks, in anchor order.
        """
        token_ids = rollout_record.token_ids.tolist()
        num_tokens = len(token_ids)
        first_anchor = max(rollout_record.prompt_len - 1, 0)

        block_records: list[BlockRecord] = []
        for anchor in range(first_anchor, num_tokens - num_draft_tokens, stride):
            labels = token_ids[anchor + 1 : anchor + 1 + num_draft_tokens]
            if len(labels) < num_draft_tokens:
                break
            block_records.append(
                BlockRecord(
                    anchor=anchor,
                    draft_token_ids=labels,
                    accepted=num_draft_tokens,
                )
            )
        return block_records
