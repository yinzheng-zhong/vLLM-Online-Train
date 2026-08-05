from typing import Any

import torch
from vllm.logger import init_logger

from vllm_online_train.checkpoint.naming import WeightNameRewriter
from vllm_online_train.head.arch import DFlashHeadArch
from vllm_online_train.head.head import TrainableDFlashHead
from vllm_online_train.head.weights import HeadWeights

logger = init_logger(__name__)


class WeightPublisher:
    DERIVED_BUFFERS = (
        "_hidden_norm_weight",
        "_fused_kv_weight",
        "_fused_kv_bias",
        "_k_norm_weights",
    )
    """Tensors `_build_context_kv_buffers` reassigns rather than writes in place."""

    def __init__(self, naming: WeightNameRewriter, weights: HeadWeights) -> None:
        """Loads a trained head into the live drafter, preserving buffer addresses.

        `DFlashQwen3Model._build_context_kv_buffers` keeps derived tensors beside the
        parameters. `load_weights` rebuilds them, but `torch.cat` allocates at a new
        address and anything holding the old pointer keeps reading the replaced tensor.
        This copies the new values back into the original storages and rebinds them.

        Args:
            naming: Un-fuses the head's parameters into on-disk naming.
            weights: Exports the head's parameter tree.
        """
        self.naming = naming
        self.weights = weights

    def publish(
        self,
        drafter_model: Any,
        head: TrainableDFlashHead,
        arch: DFlashHeadArch,
    ) -> None:
        """Push the trained weights into the running drafter.

        Must be called between engine steps. Weights are cast to the serving dtype
        here, since handing fp32 to a bf16 parameter would fail the load.

        Args:
            drafter_model: The live `DFlashQwen3ForCausalLM`.
            head: The trained head to publish from.
            arch: Supplies the split points for on-disk naming.

        Raises:
            RuntimeError: If a derived buffer changes shape or dtype across the
                rebuild, which would mean the two modules have diverged structurally.
        """
        inner = drafter_model.model
        serving_dtype = self.infer_dtype(inner)

        before = {name: getattr(inner, name, None) for name in self.DERIVED_BUFFERS}

        tensors = [
            (name, tensor.detach().to(serving_dtype))
            for name, tensor in self.naming.rewrite(self.weights.export(head), arch)
        ]
        with torch.inference_mode():
            drafter_model.load_weights(tensors)

        restored = self._restore_addresses(inner, before)
        logger.info(
            "Published online-trained DFlash weights (%d tensors, dtype %s); "
            "restored addresses for %s",
            len(tensors),
            serving_dtype,
            ", ".join(restored) or "nothing",
        )

    @staticmethod
    def _restore_addresses(
        inner: Any, before: dict[str, torch.Tensor | None]
    ) -> list[str]:
        """Copy rebuilt buffers back into their original storages.

        Args:
            inner: The drafter's inner model.
            before: Buffer name to the tensor held before the load.

        Returns:
            The names whose addresses were restored.

        Raises:
            RuntimeError: If a buffer changed shape or dtype across the rebuild.
        """
        restored: list[str] = []
        for name, original in before.items():
            rebuilt = getattr(inner, name, None)
            if original is None or rebuilt is None:
                continue
            if original.data_ptr() == rebuilt.data_ptr():
                continue
            if original.shape != rebuilt.shape or original.dtype != rebuilt.dtype:
                raise RuntimeError(
                    f"derived buffer {name} changed from {tuple(original.shape)}/"
                    f"{original.dtype} to {tuple(rebuilt.shape)}/{rebuilt.dtype} "
                    "across a publish. The trainable head and the serving module have "
                    "diverged; restart to adopt the checkpoint instead of publishing "
                    "hot."
                )
            original.copy_(rebuilt)
            setattr(inner, name, original)
            restored.append(name)
        return restored

    @staticmethod
    def infer_dtype(inner: Any) -> torch.dtype:
        """The dtype the drafter's parameters hold.

        Args:
            inner: The drafter's inner model.

        Returns:
            The first parameter's dtype, or bfloat16 when it has none.
        """
        for parameter in inner.parameters():
            return parameter.dtype
        return torch.bfloat16
