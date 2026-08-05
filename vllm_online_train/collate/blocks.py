from vllm_online_train.capture.records import RolloutRecord
from vllm_online_train.collate.batch import BlockRecord


class BlockBuilder:
    """Cuts a captured rollout into fixed-stride, fully-accepted draft blocks."""

    def build(
        self,
        record: RolloutRecord,
        *,
        num_draft_tokens: int,
        stride: int,
    ) -> list[BlockRecord]:
        """Enumerate every anchor a rollout supports.

        Anchors start at `prompt_len - 1` and stop where a full block of labels no
        longer fits. Every block is labelled fully accepted, the labels being the
        target's own continuation.

        Args:
            record: The finished rollout.
            num_draft_tokens: `D`, labels per block.
            stride: Spacing between candidate anchors.

        Returns:
            The blocks, in anchor order.
        """
        tokens = record.token_ids.tolist()
        total = len(tokens)
        first = max(record.prompt_len - 1, 0)

        blocks: list[BlockRecord] = []
        for anchor in range(first, total - num_draft_tokens, stride):
            labels = tokens[anchor + 1 : anchor + 1 + num_draft_tokens]
            if len(labels) < num_draft_tokens:
                break
            blocks.append(
                BlockRecord(
                    anchor=anchor,
                    draft_token_ids=labels,
                    accepted=num_draft_tokens,
                )
            )
        return blocks
