from pathlib import Path

import torch

from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class CheckpointReader:
    WEIGHTS_GLOB = "*.safetensors"
    """Matches both the single file `CheckpointWriter` writes and a sharded draft."""

    def read(self, directory: str | Path) -> list[tuple[str, torch.Tensor]]:
        """Read every safetensors shard in a draft directory.

        Args:
            directory: A DFlash draft directory.

        Returns:
            On-disk `(name, tensor)` pairs on the CPU, empty when the directory holds
            no safetensors file.
        """
        from safetensors.torch import load_file

        checkpoint_dir = Path(directory)
        shard_paths = sorted(checkpoint_dir.glob(self.WEIGHTS_GLOB))
        if not shard_paths:
            logger.debug("No %s under %s", self.WEIGHTS_GLOB, checkpoint_dir)
            return []

        named_tensors: list[tuple[str, torch.Tensor]] = []
        for shard_path in shard_paths:
            named_tensors.extend(load_file(str(shard_path)).items())
        logger.debug(
            "Read %d draft tensors from %d shard(s) under %s",
            len(named_tensors),
            len(shard_paths),
            checkpoint_dir,
        )
        return named_tensors
