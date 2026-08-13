from pathlib import Path
from typing import Any

import torch

from vllm_online_train.config.shapes import ResolvedConfig
from vllm_online_train.contracts.provider import StateProvider
from vllm_online_train.engine.capture.rollout_capture import RolloutCapture
from vllm_online_train.engine.rollouts.rollout_buffer import RolloutBuffer
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
        resolved_config: ResolvedConfig,
        state_provider: StateProvider,
        rollout_buffer: RolloutBuffer,
        rollout_capture: RolloutCapture,
        draft_head: TrainableDFlashHead,
        online_trainer: OnlineTrainer,
        checkpoint_exporter: CheckpointExporter,
        weight_publisher: WeightPublisher,
    ) -> None:
        """The single handle over the capture, the pool, the head and the optimizer.

        Both threads hold it: the engine thread calls `observe` and `abort`, the trainer
        thread calls `train_step` and `metrics`. The split is by object rather than by
        lock -- the capture writes the pool and the trainer only reads it -- so nothing
        here blocks a serving step.

        Args:
            resolved_config: Settings paired with the resolved shapes.
            state_provider: Owns the target handle everything else borrows through.
            rollout_buffer: The rollout pool.
            rollout_capture: Tees engine steps into the pool.
            draft_head: The head being trained.
            online_trainer: Runs the optimizer loop.
            checkpoint_exporter: Writes checkpoint directories.
            weight_publisher: Loads trained weights into the live drafter.
        """
        self.resolved_config = resolved_config
        self.state_provider = state_provider
        self.rollout_buffer = rollout_buffer
        self.rollout_capture = rollout_capture
        self.draft_head = draft_head
        self.online_trainer = online_trainer
        self.checkpoint_exporter = checkpoint_exporter
        self.weight_publisher = weight_publisher
        self._drafter_model: Any | None = None

    def observe(self, engine_step: EngineStep) -> None:
        """Tee this step's activations. Safe to call on every step.

        Args:
            engine_step: The engine step's schedule and activations.
        """
        if engine_step.aux_hidden_states is None:
            return
        if not isinstance(engine_step.sampled_token_ids, torch.Tensor):
            # The `disable_padded_drafter_batch` path hands back per-request token
            # lists, which carry no rejection mask to derive the valid count from.
            return
        self.rollout_capture.observe(engine_step)

    def abort(self, req_ids: set[str]) -> None:
        """Discard rollouts for requests that will not finish normally.

        Args:
            req_ids: Ids the engine aborted.
        """
        self.rollout_capture.abort(req_ids)

    def train_step(self) -> dict[str, float] | None:
        """Run one training micro-step, exporting or publishing when due.

        Draws the pool as the capture last left it. A rollout is admitted by the engine
        thread only after that same step drained the rollout's final chunk, so there is
        nothing here for the trainer thread to flush.

        Returns:
            The step's metrics, or `None` when the pool is too thin.
        """
        metrics = self.online_trainer.step_from_buffer(self.rollout_buffer)
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
        bookkeeping_settings = self.resolved_config.online_train_settings.bookkeeping
        cadence = bookkeeping_settings.checkpoint_every
        return bool(
            cadence
            and metrics.get("stepped")
            and self.online_trainer.step_count % cadence == 0
        )

    def _checkpoint(self, metrics: dict[str, float]) -> None:
        """Export and, when configured, publish. Records both in the metrics.

        Args:
            metrics: The step's metrics, annotated in place.
        """
        online_train_settings = self.resolved_config.online_train_settings
        if online_train_settings.bookkeeping.checkpoint_dir:
            metrics["checkpoint"] = 1.0
            self.export_checkpoint(
                online_train_settings.bookkeeping.checkpoint_dir
            )
        if online_train_settings.publish.hot:
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
        self.weight_publisher.publish(
            self._drafter_model, self.draft_head, self.draft_head.head_arch
        )
        logger.debug(
            "Published trained head into live drafter at step %d",
            self.online_trainer.step_count,
        )

    def export_checkpoint(self, directory: str | Path) -> str:
        """Write a directory a restarted engine can serve directly.

        Args:
            directory: Destination; created if absent.

        Returns:
            The path of the written safetensors file.
        """
        written = self.checkpoint_exporter.export(
            self.draft_head, directory, self.state_provider.serving_dtype
        )
        logger.info(
            "Exported online-trained DFlash head at step %d to %s",
            self.online_trainer.step_count,
            directory,
        )
        return written

    def metrics(self) -> dict[str, float]:
        """Capture, buffer and training counters."""
        combined = self.rollout_capture.as_metrics()
        combined.update(
            {
                f"train/{name}": value
                for name, value in self.online_trainer.last_metrics.items()
            }
        )
        combined["train/step"] = float(self.online_trainer.step_count)
        return combined

    def num_buffered_rollouts(self) -> int:
        """Rollouts the pool holds."""
        return self.rollout_buffer.num_rollouts
