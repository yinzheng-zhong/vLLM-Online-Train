from collections.abc import Iterable, Iterator

import torch

from vllm_online_train.logger import init_logger
from vllm_online_train.training.head.arch import DFlashHeadArch

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

    SPLIT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
        ("gate_up_proj", ("gate_proj", "up_proj")),
    )
    """Fused projection name to the split names it concatenates, in that order."""

    def rewrite(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        head_arch: DFlashHeadArch,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Un-fuse the projections and drop the target-bound tensors.

        Args:
            named_tensors: Fused-naming pairs, i.e. `HeadWeights.export()`.
            head_arch: Supplies the split points.

        Yields:
            One `(name, tensor)` pair per on-disk tensor.
        """
        logger.debug(
            "Rewriting fused head weights: q_size=%d, kv_size=%d, intermediate_size=%d",
            head_arch.q_size,
            head_arch.kv_size,
            head_arch.intermediate_size,
        )
        for name, tensor in named_tensors:
            if any(shared_name in name for shared_name in self.SHARED_WITH_TARGET):
                continue

            if name.endswith(("qkv_proj.weight", "qkv_proj.bias")):
                yield from self._split_qkv(name, tensor, head_arch)
            elif name.endswith("gate_up_proj.weight"):
                yield from self._split_gate_up(name, tensor, head_arch)
            else:
                yield name, tensor

    def fuse(
        self, named_tensors: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Concatenate the split projections back into the head's fused naming.

        The inverse of `rewrite`, minus the tensors it drops: a checkpoint carries no
        `embed_tokens` or `lm_head`, so nothing here re-adds them.

        Args:
            named_tensors: On-disk pairs, i.e. what `rewrite` produced.

        Yields:
            One `(name, tensor)` pair per fused parameter.

        Raises:
            ValueError: If a split projection is missing one of its parts, which would
                otherwise fuse into a tensor of the wrong width.
        """
        on_disk = dict(named_tensors)
        split_names: set[str] = set()
        fused_tensors: dict[str, torch.Tensor] = {}

        for name in sorted(on_disk):
            split_group = self._split_group(name)
            if split_group is None:
                continue
            name_stem, fused_suffix, split_suffixes, param_suffix = split_group
            part_names = [
                f"{name_stem}.{split_suffix}.{param_suffix}"
                for split_suffix in split_suffixes
            ]
            absent_names = [part for part in part_names if part not in on_disk]
            if absent_names:
                raise ValueError(
                    f"checkpoint holds {name} but not {', '.join(absent_names)}. "
                    f"{fused_suffix} concatenates {', '.join(split_suffixes)} and "
                    "cannot be fused from part of itself."
                )
            split_names.update(part_names)
            fused_name = f"{name_stem}.{fused_suffix}.{param_suffix}"
            if fused_name not in fused_tensors:
                fused_tensors[fused_name] = torch.cat(
                    [on_disk[part] for part in part_names], dim=0
                )

        logger.debug(
            "Fused %d on-disk tensors into %d parameters",
            len(split_names),
            len(fused_tensors),
        )
        for name, tensor in on_disk.items():
            if name not in split_names:
                yield name, tensor
        yield from fused_tensors.items()

    @classmethod
    def _split_group(
        cls, name: str
    ) -> tuple[str, str, tuple[str, ...], str] | None:
        """The fused group an on-disk parameter name belongs to.

        Args:
            name: An on-disk parameter name.

        Returns:
            `(name_stem, fused_suffix, split_suffixes, param_suffix)`, or `None` when
            the name is not one part of a split projection.
        """
        if name.count(".") < 2:
            return None
        name_stem, projection_name, param_suffix = name.rsplit(".", 2)
        for fused_suffix, split_suffixes in cls.SPLIT_GROUPS:
            if projection_name in split_suffixes:
                return name_stem, fused_suffix, split_suffixes, param_suffix
        return None

    @staticmethod
    def _split_qkv(
        name: str, tensor: torch.Tensor, head_arch: DFlashHeadArch
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Split one fused QKV tensor into three.

        Args:
            name: The fused parameter name.
            tensor: Its value.
            head_arch: Supplies `q_size` and `kv_size`.

        Yields:
            The `q_proj`, `k_proj` and `v_proj` pairs.
        """
        param_suffix = name.rsplit(".", 1)[1]
        name_stem = name[: -len(f"qkv_proj.{param_suffix}")]
        query, key, value = tensor.split(
            [head_arch.q_size, head_arch.kv_size, head_arch.kv_size], dim=0
        )
        yield f"{name_stem}q_proj.{param_suffix}", query
        yield f"{name_stem}k_proj.{param_suffix}", key
        yield f"{name_stem}v_proj.{param_suffix}", value

    @staticmethod
    def _split_gate_up(
        name: str, tensor: torch.Tensor, head_arch: DFlashHeadArch
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Split one fused gate/up tensor into two.

        Args:
            name: The fused parameter name.
            tensor: Its value.
            head_arch: Supplies `intermediate_size`.

        Yields:
            The `gate_proj` and `up_proj` pairs.
        """
        name_stem = name[: -len("gate_up_proj.weight")]
        gate, up = tensor.split(
            [head_arch.intermediate_size, head_arch.intermediate_size], dim=0
        )
        yield f"{name_stem}gate_proj.weight", gate
        yield f"{name_stem}up_proj.weight", up
