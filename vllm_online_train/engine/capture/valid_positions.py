from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class ValidPositions:
    """Decides how many of a request's scheduled positions have usable activations."""

    @staticmethod
    def count(*, scheduled: int, num_draft: int, num_valid: int) -> int:
        """Usable positions for one request in one step.

        With `r` of `num_draft` draft tokens accepted, the activations at block
        position `i` are conditioned on `[..., x, d_1..d_i]`, so they sit on the true
        sequence only for `i <= r`. `num_valid` counts the non-`-1` entries of the
        sampled row, which is `r + 1`.

        Args:
            scheduled: Positions the step scheduled for this request.
            num_draft: Draft tokens the step verified for it.
            num_valid: Non-`-1` entries in its sampled row.

        Returns:
            Leading positions to keep. Every scheduled position when no speculation
            ran.
        """
        if num_draft == 0:
            return scheduled
        return min(num_valid, scheduled)

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
        counts = list(spec_decode_metadata.num_draft_tokens)
        if len(counts) < num_reqs:
            counts.extend([0] * (num_reqs - len(counts)))
        return counts
