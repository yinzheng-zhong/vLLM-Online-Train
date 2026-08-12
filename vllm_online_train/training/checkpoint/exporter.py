import shutil
from pathlib import Path

import torch

from vllm_online_train.logger import init_logger
from vllm_online_train.training.checkpoint.writer import CheckpointWriter
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead
from vllm_online_train.training.head.weights import HeadWeights

logger = init_logger(__name__)


class CheckpointExporter:
    def __init__(
        self,
        checkpoint_writer: CheckpointWriter,
        head_weights: HeadWeights,
        source_dir: Path,
    ) -> None:
        """Writes a head to a directory a restarted engine can serve directly.

        The weights go out in on-disk DFlash naming and the serving dtype, beside a
        verbatim copy of the `config.json` the engine is already running.

        Args:
            checkpoint_writer: Writes the safetensors file and the config.
            head_weights: Exports the head's parameter tree.
            source_dir: The served head's directory, whose `config.json` is copied.
        """
        self.checkpoint_writer = checkpoint_writer
        self.head_weights = head_weights
        self.source_dir = source_dir

    def export(
        self,
        draft_head: TrainableDFlashHead,
        directory: str | Path,
        dtype: torch.dtype,
    ) -> str:
        """Write one checkpoint.

        Args:
            draft_head: The head to export.
            directory: Destination; created if absent.
            dtype: Cast before writing.

        Returns:
            The path of the written safetensors file.
        """
        checkpoint_dir = Path(directory)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._copy_source_config(checkpoint_dir)
        written = self.checkpoint_writer.write(
            checkpoint_dir,
            self.head_weights.export(draft_head),
            draft_head.head_arch,
            dtype=dtype,
        )
        return str(written)

    def _copy_source_config(self, checkpoint_dir: Path) -> None:
        """Copy the served head's `config.json` beside the exported weights.

        Args:
            checkpoint_dir: The checkpoint directory.
        """
        config_filename = self.checkpoint_writer.CONFIG_FILENAME
        source_config = self.source_dir / config_filename
        if source_config.is_file():
            shutil.copyfile(source_config, checkpoint_dir / config_filename)
        else:
            logger.warning(
                "No %s beside the served head at %s; the exported checkpoint at %s "
                "will need one copied in before it can be served.",
                config_filename,
                self.source_dir,
                checkpoint_dir,
            )
