import inspect
from typing import Any


class SignatureGuard:
    """Refuses to wrap a `propose_draft_token_ids` this package does not recognise.

    Nothing about the wrapped call path is API, so a vLLM bump that reorders or
    renames these parameters must break at startup rather than silently capture
    activations from the wrong tensor.
    """

    EXPECTED_PARAMS: tuple[str, ...] = (
        "self",
        "scheduler_output",
        "sampled_token_ids",
        "sampling_metadata",
        "hidden_states",
        "sample_hidden_states",
        "aux_hidden_states",
        "spec_decode_metadata",
        "common_attn_metadata",
        "slot_mappings",
    )

    PINNED_COMMIT = "05a081486"
    """The vLLM commit `EXPECTED_PARAMS` was read from."""

    def check(self, method: Any) -> None:
        """Compare a method's parameters against the pin.

        Args:
            method: The unbound method about to be wrapped.

        Raises:
            RuntimeError: If the parameter names or their order differ.
        """
        found = tuple(inspect.signature(method).parameters)
        if found != self.EXPECTED_PARAMS:
            raise RuntimeError(
                "vllm-online-train wraps the private "
                "GPUModelRunner.propose_draft_token_ids and its signature has "
                "changed.\n"
                f"  expected ({self.PINNED_COMMIT}): {self.EXPECTED_PARAMS}\n"
                f"  found:                       {found}\n"
                "Re-verify the capture against the new call path -- particularly "
                "which tensor holds the final hidden state and whether "
                "sampled_token_ids still carries the -1 rejection mask -- then update "
                "EXPECTED_PARAMS."
            )
