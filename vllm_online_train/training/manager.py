from pathlib import Path
from typing import Any

import torch

from vllm_online_train.config.shapes import ResolvedConfig
from vllm_online_train.contracts.provider import StateProvider
from vllm_online_train.engine.capture.rollout_buffer import RolloutBuffer
from vllm_online_train.engine.capture.rollout_capture import RolloutCapture
from vllm_online_train.logger import init_logger
from vllm_online_train.step import EngineStep
from vllm_online_train.training.checkpoint.exporter import CheckpointExporter
from vllm_online_train.training.checkpoint.weight_publisher import WeightPublisher
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead
from vllm_online_train.training.optim.trainer import OnlineTrainer

logger = init_logger(__name__)


class OnlineTrainManager:
    def __init__(
        self,
        config: ResolvedConfig,
        provider: StateProvider,
        buffer: RolloutBuffer,
        capture: RolloutCapture,
        head: TrainableDFlashHead,
        trainer: OnlineTrainer,
        exporter: CheckpointExporter,
        publisher: WeightPublisher,
    ) -> None:
        """The single handle over the capture, the pool, the head and the optimizer.

        Both threads hold it: the engine thread calls `observe` and `abort`, the trainer
        thread calls `train_step` and `metrics`. The split is by object rather than by
        lock -- the capture writes the pool and the trainer only reads it -- so nothing
        here blocks a serving step.

        Args:
            config: Settings paired with the resolved shapes.
            provider: Owns the target handle everything else borrows through.
            buffer: The rollout pool.
            capture: Tees engine steps into the pool.
            head: The head being trained.
            trainer: Runs the optimizer loop.
            exporter: Writes checkpoint directories.
            publisher: Loads trained weights into the live drafter.
        """
        self.config = config
        self.provider = provider
        self.buffer = buffer
        self.capture = capture
        self.head = head
        self.trainer = trainer
        self.exporter = exporter
        self.publisher = publisher
        self._drafter_model: Any | None = None

    def observe(self, step: EngineStep) -> None:
        """Tee this step's activations. Safe to call on every step.

        Args:
            step: The engine step's schedule and activations.
        """
        if step.aux_hidden_states is None:
            return
        if not isinstance(step.sampled_token_ids, torch.Tensor):
            # The `disable_padded_drafter_batch` path hands back per-request token
            # lists, which carry no rejection mask to derive the valid count from.
            return
        self.capture.observe(step)

    def abort(self, req_ids: set[str]) -> None:
        """Discard rollouts for requests that will not finish normally.

        Args:
            req_ids: Ids the engine aborted.
        """
        self.capture.abort(req_ids)

    def train_step(self) -> dict[str, float] | None:
        """Run one training micro-step, exporting or publishing when due.

        Draws the pool as the capture last left it. A rollout is admitted by the engine
        thread only after that same step drained the rollout's final chunk, so there is
        nothing here for the trainer thread to flush.

        Returns:
            The step's metrics, or `None` when the pool is too thin.
        """
        metrics = self.trainer.step_from_buffer(self.buffer)
        if metrics is None:
            return None
        if self._checkpoint_due(metrics):
            self._checkpoint(metrics)
        return metrics

    def _checkpoint_due(self, metrics: dict[str, float]) -> bool:
        """Whether this step lands on the checkpoint cadence.

        Args:
            metrics: The step's metrics.
        """
        cadence = self.config.settings.bookkeeping.checkpoint_every
        return bool(
            cadence
            and metrics.get("stepped")
            and self.trainer.step_count % cadence == 0
        )

    def _checkpoint(self, metrics: dict[str, float]) -> None:
        """Export and, when configured, publish. Records both in the metrics.

        Args:
            metrics: The step's metrics, annotated in place.
        """
        settings = self.config.settings
        if settings.bookkeeping.checkpoint_dir:
            metrics["checkpoint"] = 1.0
            self.export_checkpoint(settings.bookkeeping.checkpoint_dir)
        if settings.publish.hot:
            metrics["published"] = 1.0
            self.publish()

    def set_publish_target(self, drafter_model: Any | None) -> None:
        """Record the live drafter module that `publish` writes into.

        Args:
            drafter_model: The `DFlashQwen3ForCausalLM`, or `None`.
        """
        self._drafter_model = drafter_model
        logger.debug(
            "Publish target set to %s",
            "none" if drafter_model is None else type(drafter_model).__name__,
        )

    def publish(self) -> None:
        """Push the trained head into the live drafter, if one was resolved."""
        if self._drafter_model is None:
            logger.warning(
                "Hot publish requested but no live drafter was resolved; the "
                "checkpoint on disk is still current."
            )
            return
        self.publisher.publish(self._drafter_model, self.head, self.head.arch)
        logger.debug(
            "Published trained head into live drafter at step %d",
            self.trainer.step_count,
        )

    def export_checkpoint(self, directory: str | Path) -> str:
        """Write a directory a restarted engine can serve directly.

        Args:
            directory: Destination; created if absent.

        Returns:
            The path of the written safetensors file.
        """
        written = self.exporter.export(
            self.head, directory, self.provider.serving_dtype
        )
        logger.info(
            "Exported online-trained DFlash head at step %d to %s",
            self.trainer.step_count,
            directory,
        )
        return written

    def metrics(self) -> dict[str, float]:
        """Capture, buffer and training counters."""
        combined = self.capture.as_metrics()
        combined.update(
            {f"train/{k}": v for k, v in self.trainer.last_metrics.items()}
        )
        combined["train/step"] = float(self.trainer.step_count)
        return combined

    def num_buffered_rollouts(self) -> int:
        """Rollouts the pool holds."""
        return self.buffer.num_rollouts
