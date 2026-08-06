import time

import torch
from vllm.logger import init_logger

from vllm_online_train.config.settings import OptimSettings
from vllm_online_train.engine.capture.rollout_buffer import RolloutBuffer
from vllm_online_train.training.collate.replay_batch import ReplayBatch
from vllm_online_train.training.collate.rollout_collator import RolloutCollator
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead
from vllm_online_train.training.head.weights import HeadWeights
from vllm_online_train.training.loss.sft_objective import SFTObjective
from vllm_online_train.training.optim.schedule_factory import ScheduleFactory

logger = init_logger(__name__)


class OnlineTrainer:
    DEBUG_LOG_EVERY_STEPS: int = 50
    """Optimizer steps between debug-level grad_norm/lr log lines."""

    def __init__(
        self,
        settings: OptimSettings,
        head: TrainableDFlashHead,
        weights: HeadWeights,
        objective: SFTObjective,
        collator: RolloutCollator,
        schedules: ScheduleFactory,
        optimizer: torch.optim.Optimizer,
        sequences_per_step: int,
    ) -> None:
        """Trains a `TrainableDFlashHead` on batches drawn from a `RolloutBuffer`.

        Args:
            settings: Clipping, accumulation and the schedule shape.
            head: The head being trained.
            weights: Re-freezes the borrowed tensors after a state restore.
            objective: Scores one replayed batch.
            collator: Turns drawn rollouts into a batch.
            schedules: Rebuilds the learning-rate multiplier on restore.
            optimizer: Steps the trainable parameters.
            sequences_per_step: Rollouts drawn from the buffer per batch.
        """
        self.settings = settings
        self.head = head
        self.weights = weights
        self.objective = objective
        self.collator = collator
        self.schedules = schedules
        self.optimizer = optimizer
        self.sequences_per_step = sequences_per_step

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, schedules.create(settings)
        )
        self.step_count = 0
        self.micro_step_count = 0
        self.last_metrics: dict[str, float] = {}
        logger.debug(
            "OnlineTrainer: sequences_per_step=%d, grad_accum_steps=%d, grad_clip=%.3g",
            sequences_per_step,
            settings.grad_accum_steps,
            settings.grad_clip,
        )

    def step(self, batch: ReplayBatch) -> dict[str, float]:
        """Run one micro-batch, stepping the optimizer once per accumulation cycle.

        Args:
            batch: The batch to train on.

        Returns:
            Metrics for this micro-step, including whether the optimizer stepped.
        """
        settings = self.settings
        began = time.perf_counter()

        self.head.train()
        student_hidden = self.head.replay(batch)
        loss, metrics = self.objective(student_hidden=student_hidden, batch=batch)
        (loss / settings.grad_accum_steps).backward()

        self.micro_step_count += 1
        metrics["micro_step"] = float(self.micro_step_count)

        if self.micro_step_count % settings.grad_accum_steps == 0:
            parameters = self.head.trainable_parameters()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, settings.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.step_count += 1
            metrics["grad_norm"] = float(grad_norm)
            metrics["stepped"] = 1.0
            if self.step_count % self.DEBUG_LOG_EVERY_STEPS == 0:
                logger.debug(
                    "OnlineTrainer step %d: grad_norm=%.4f, lr=%.6g",
                    self.step_count,
                    float(grad_norm),
                    self.optimizer.param_groups[0]["lr"],
                )
        else:
            metrics["stepped"] = 0.0

        metrics["step"] = float(self.step_count)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["seconds"] = time.perf_counter() - began
        self.last_metrics = metrics
        return metrics

    def step_from_buffer(self, buffer: RolloutBuffer) -> dict[str, float] | None:
        """Draw, collate and train on one batch.

        Args:
            buffer: The pool to draw from.

        Returns:
            The step's metrics, or `None` when the pool is too thin or no drawn
            rollout yields a block.
        """
        if not buffer.can_sample(self.sequences_per_step):
            return None
        batch = self.collator.collate(buffer.sample(self.sequences_per_step))
        if batch is None:
            return None
        return self.step(batch)

    def state_dict(self) -> dict:
        """Everything needed to resume: weights, moments and schedule position."""
        return {
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step_count": self.step_count,
            "micro_step_count": self.micro_step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a run, keeping this run's schedule settings.

        `LambdaLR.state_dict()` saves the `__dict__` of any callable that is not a
        plain function, and `load_state_dict` writes it back. Restoring it would pin
        the multiplier to the checkpoint's `total_steps` rather than this run's, so
        the lambdas are rebuilt afterwards and only the step position is kept.

        Args:
            state: A mapping from `state_dict`.
        """
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
            self.scheduler.lr_lambdas = [
                self.schedules.create(self.settings)
                for _ in self.scheduler.lr_lambdas
            ]
        self.step_count = int(state.get("step_count", 0))
        self.micro_step_count = int(state.get("micro_step_count", 0))
        self.weights.freeze_shared(self.head)
        logger.debug(
            "OnlineTrainer resumed at step_count=%d, micro_step_count=%d",
            self.step_count,
            self.micro_step_count,
        )
