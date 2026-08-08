import threading
from collections.abc import Callable
from typing import Any

from vllm_online_train.engine.hook.signature_guard import SignatureGuard
from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class MethodPatcher:
    def __init__(self, guard: SignatureGuard) -> None:
        """Owns the wrapping and unwrapping of `propose_draft_token_ids`.

        Idempotent and reversible: `register()` runs in several vLLM processes, and
        patching a class in one that never builds a runner is inert.

        Args:
            guard: Checks the method's signature against the pin before wrapping.
        """
        self.guard = guard
        self._lock = threading.Lock()
        self._installed = False
        self._original: Any = None

    @property
    def installed(self) -> bool:
        """Whether the runner method is currently wrapped."""
        return self._installed

    def install(self, observe: Callable[..., None]) -> bool:
        """Wrap the runner method so `observe` runs before it.

        Args:
            observe: Called with `(runner, scheduler_output, sampled_token_ids,
                hidden_states, aux_hidden_states, spec_decode_metadata)`.

        Returns:
            Whether the method is now wrapped.

        Raises:
            RuntimeError: If the method's signature does not match the pin.
        """
        with self._lock:
            if self._installed:
                return True

            # Imported here rather than at module scope, so the class exists
            # whichever order vLLM reached this from.
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

            # Read off the class, so `original` is a plain function whose first
            # positional parameter is `self`.
            original = GPUModelRunner.propose_draft_token_ids
            # Raises before the class attribute below is reassigned.
            self.guard.check(original)
            # Rebinds one entry in the class dict. Every instance, already built
            # or not, resolves the name to the replacement from here on.
            GPUModelRunner.propose_draft_token_ids = self._wrap(original, observe)
            # The only remaining reference to vLLM's function.
            self._original = original
            self._installed = True
            logger.debug(
                "Online training capture hook installed (pin %s)",
                self.guard.PINNED_COMMIT,
            )
            return True

    def uninstall(self) -> None:
        """Restore the original method, if one was wrapped."""
        with self._lock:
            if not self._installed:
                return
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

            # Rebinds the class dict entry back to vLLM's function.
            GPUModelRunner.propose_draft_token_ids = self._original
            self._original = None
            self._installed = False
            logger.debug("Online training capture hook uninstalled")

    @staticmethod
    def _wrap(original: Any, observe: Callable[..., None]) -> Any:
        """Build the replacement method.

        Args:
            original: The unbound method being wrapped.
            observe: Runs before the delegation, while `sampled_token_ids` still
                carries its `-1` rejection mask.

        Returns:
            A function with the same signature that observes, then delegates.
        """

        # vLLM's call site passes all nine arguments positionally, so this
        # parameter list must match `SignatureGuard.EXPECTED_PARAMS` in order.
        def propose_draft_token_ids(
            self,
            scheduler_output: Any,
            sampled_token_ids: Any,
            sampling_metadata: Any,
            hidden_states: Any,
            sample_hidden_states: Any,
            aux_hidden_states: Any,
            spec_decode_metadata: Any,
            common_attn_metadata: Any,
            slot_mappings: Any,
        ) -> Any:
            # Runs first: `sampled_token_ids` still holds -1 in rejected slots.
            observe(
                self,
                scheduler_output,
                sampled_token_ids,
                hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
            )
            # Delegates with every argument unchanged and returns the result
            # verbatim, so the serving path sees no difference.
            return original(
                self,
                scheduler_output,
                sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                common_attn_metadata,
                slot_mappings,
            )

        # Standard-library convention: makes `inspect.signature` and
        # `inspect.unwrap` resolve through the replacement to `original`.
        propose_draft_token_ids.__wrapped__ = original
        return propose_draft_token_ids
