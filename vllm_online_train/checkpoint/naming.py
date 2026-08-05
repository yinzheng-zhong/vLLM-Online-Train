from collections.abc import Iterable, Iterator

import torch
from vllm.logger import init_logger

from vllm_online_train.head.arch import DFlashHeadArch

logger = init_logger(__name__)


class WeightNameRewriter:
    """Rewrites fused head weights into the split naming real checkpoints use.

    The trainable head keeps vLLM's fused parameter tree (`qkv_proj`, `gate_up_proj`).
    On-disk DFlash checkpoints are split (`q_proj` / `k_proj` / `v_proj`, `gate_proj` /
    `up_proj`) and `DFlashQwen3Model.hf_to_vllm_mapper` re-fuses them at load time.
    """

    SHARED_WITH_TARGET = ("embed_tokens", "lm_head")
    """Tensors the serving path binds from the target when the checkpoint omits
    them."""

    def rewrite(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        arch: DFlashHeadArch,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Un-fuse the projections and drop the target-bound tensors.

        Args:
            named_tensors: Fused-naming pairs, i.e. `HeadWeights.export()`.
            arch: Supplies the split points.

        Yields:
            One `(name, tensor)` pair per on-disk tensor.
        """
        logger.debug(
            "Rewriting fused head weights: q_size=%d, kv_size=%d, intermediate_size=%d",
            arch.q_size,
            arch.kv_size,
            arch.intermediate_size,
        )
        for name, tensor in named_tensors:
            if any(shared in name for shared in self.SHARED_WITH_TARGET):
                continue

            if name.endswith(("qkv_proj.weight", "qkv_proj.bias")):
                yield from self._split_qkv(name, tensor, arch)
            elif name.endswith("gate_up_proj.weight"):
                yield from self._split_gate_up(name, tensor, arch)
            else:
                yield name, tensor

    @staticmethod
    def _split_qkv(
        name: str, tensor: torch.Tensor, arch: DFlashHeadArch
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Split one fused QKV tensor into three.

        Args:
            name: The fused parameter name.
            tensor: Its value.
            arch: Supplies `q_size` and `kv_size`.

        Yields:
            The `q_proj`, `k_proj` and `v_proj` pairs.
        """
        suffix = name.rsplit(".", 1)[1]
        stem = name[: -len(f"qkv_proj.{suffix}")]
        q, k, v = tensor.split([arch.q_size, arch.kv_size, arch.kv_size], dim=0)
        yield f"{stem}q_proj.{suffix}", q
        yield f"{stem}k_proj.{suffix}", k
        yield f"{stem}v_proj.{suffix}", v

    @staticmethod
    def _split_gate_up(
        name: str, tensor: torch.Tensor, arch: DFlashHeadArch
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Split one fused gate/up tensor into two.

        Args:
            name: The fused parameter name.
            tensor: Its value.
            arch: Supplies `intermediate_size`.

        Yields:
            The `gate_proj` and `up_proj` pairs.
        """
        stem = name[: -len("gate_up_proj.weight")]
        gate, up = tensor.split(
            [arch.intermediate_size, arch.intermediate_size], dim=0
        )
        yield f"{stem}gate_proj.weight", gate
        yield f"{stem}up_proj.weight", up
