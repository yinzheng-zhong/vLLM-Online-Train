from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class ValidPositions:
    """Decides how many of a request's scheduled positions have usable activations."""

    @staticmethod
    def count(
        *,
        num_scheduled_tokens: int,
        num_draft_tokens: int,
        num_valid_tokens: int,
    ) -> int:
        """Usable positions for one request in one step.

        With `r` of `num_draft_tokens` draft tokens accepted, the activations at block
        position `i` are conditioned on `[..., x, d_1..d_i]`, so they sit on the true
        sequence only for `i <= r`. `num_valid_tokens` counts the non-`-1` entries of
        the sampled row, which is `r + 1`.

        Args:
            num_scheduled_tokens: Positions the step scheduled for this request.
            num_draft_tokens: Draft tokens the step verified for it.
            num_valid_tokens: Non-`-1` entries in its sampled row.

        Returns:
            Leading positions to keep. Every scheduled position when no speculation
            ran.
        """
        if num_draft_tokens == 0:
            return num_scheduled_tokens
        return min(num_valid_tokens, num_scheduled_tokens)

    @staticmethod
    def draft_counts(
        spec_decode_metadata: "SpecDecodeMetadata | None", num_reqs: int
    ) -> list[int]:
        """Per-request draft counts, padded to the batch.

        Args:
            spec_decode_metadata: This step's speculation metadata, or `None`.
            num_reqs: Requests in the batch.

        Returns:
            `num_reqs` counts, zero-filled where metadata is absent or short.
        """
        if spec_decode_metadata is None:
            return [0] * num_reqs
        draft_counts = list(spec_decode_metadata.num_draft_tokens)
        if len(draft_counts) < num_reqs:
            draft_counts.extend([0] * (num_reqs - len(draft_counts)))
        return draft_counts
