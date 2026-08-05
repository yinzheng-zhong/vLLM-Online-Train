import json
from collections.abc import Iterable
from pathlib import Path

import torch
from vllm.logger import init_logger

from vllm_online_train.checkpoint.naming import WeightNameRewriter
from vllm_online_train.head.arch import DFlashHeadArch

logger = init_logger(__name__)


class CheckpointWriter:
    CONFIG_FILENAME = "config.json"
    WEIGHTS_FILENAME = "model.safetensors"

    def __init__(self, naming: WeightNameRewriter) -> None:
        """Writes a DFlash draft directory in the on-disk format.

        `embed_tokens` and `lm_head` are deliberately omitted: when a DFlash checkpoint
        lacks them the serving path binds the target's own modules onto the draft, which
        is the sharing the objective assumes.

        Args:
            naming: Un-fuses the head's parameters into on-disk naming.
        """
        self.naming = naming

    def write(
        self,
        directory: str | Path,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        arch: DFlashHeadArch,
        *,
        config: dict | None = None,
        dtype: torch.dtype | None = None,
    ) -> Path:
        """Write `model.safetensors`, and `config.json` when `config` is given.

        Args:
            directory: Destination; created if absent.
            named_tensors: Fused-naming pairs, i.e. `HeadWeights.export()`.
            arch: Supplies the split points for un-fusing.
            config: `config.json` contents. Omit to refresh only the weights.
            dtype: Cast before writing. `None` keeps the head's dtype.

        Returns:
            The path of the written safetensors file.
        """
        from safetensors.torch import save_file

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        tensors: dict[str, torch.Tensor] = {}
        for name, tensor in self.naming.rewrite(named_tensors, arch):
            tensor = tensor.detach().to("cpu")
            if dtype is not None:
                tensor = tensor.to(dtype)
            tensors[name] = tensor.contiguous()

        if config is not None:
            (path / self.CONFIG_FILENAME).write_text(
                json.dumps(config, indent=2) + "\n"
            )

        weights_path = path / self.WEIGHTS_FILENAME
        save_file(tensors, str(weights_path))
        logger.info("Wrote %d draft tensors to %s", len(tensors), weights_path)
        return weights_path
