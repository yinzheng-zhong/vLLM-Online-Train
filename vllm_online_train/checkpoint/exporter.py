import shutil
from pathlib import Path

import torch
from vllm.logger import init_logger

from vllm_online_train.checkpoint.writer import CheckpointWriter
from vllm_online_train.head.head import TrainableDFlashHead
from vllm_online_train.head.weights import HeadWeights

logger = init_logger(__name__)


class CheckpointExporter:
    """Writes a head to a directory a restarted engine can serve directly.

    The weights go out in on-disk DFlash naming and the serving dtype, beside a
    verbatim copy of the `config.json` the engine is already running.
    """

    def __init__(
        self, writer: CheckpointWriter, weights: HeadWeights, source_dir: Path
    ) -> None:
        """
        Args:
            writer: Writes the safetensors file and the config.
            weights: Exports the head's parameter tree.
            source_dir: The served head's directory, whose `config.json` is copied.
        """
        self.writer = writer
        self.weights = weights
        self.source_dir = source_dir

    def export(
        self,
        head: TrainableDFlashHead,
        directory: str | Path,
        dtype: torch.dtype,
    ) -> str:
        """Write one checkpoint.

        Args:
            head: The head to export.
            directory: Destination; created if absent.
            dtype: Cast before writing.

        Returns:
            The path of the written safetensors file.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._copy_source_config(path)
        written = self.writer.write(
            path, self.weights.export(head), head.arch, dtype=dtype
        )
        return str(written)

    def _copy_source_config(self, path: Path) -> None:
        """Copy the served head's `config.json` beside the exported weights.

        Args:
            path: The checkpoint directory.
        """
        filename = self.writer.CONFIG_FILENAME
        source_config = self.source_dir / filename
        if source_config.is_file():
            shutil.copyfile(source_config, path / filename)
        else:
            logger.warning(
                "No %s beside the served head at %s; the exported checkpoint at %s "
                "will need one copied in before it can be served.",
                filename,
                self.source_dir,
                path,
            )
