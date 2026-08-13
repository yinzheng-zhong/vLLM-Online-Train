import time

import torch

from vllm_online_train.config.settings import OptimSettings
from vllm_online_train.engine.rollouts.rollout_buffer import RolloutBuffer
from vllm_online_train.logger import init_logger
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
        optim_settings: OptimSettings,
        draft_head: TrainableDFlashHead,
        head_weights: HeadWeights,
        sft_objective: SFTObjective,
        rollout_collator: RolloutCollator,
        schedule_factory: ScheduleFactory,
        optimizer: torch.optim.Optimizer,
        sequences_per_step: int,
    ) -> None:
        """Trains a `TrainableDFlashHead` on batches drawn from a `RolloutBuffer`.

        Args:
            optim_settings: Clipping, accumulation and the schedule shape.
            draft_head: The head being trained.
            head_weights: Re-freezes the borrowed tensors after a state restore.
            sft_objective: Scores one replayed batch.
            rollout_collator: Turns drawn rollouts into a batch.
            schedule_factory: Rebuilds the learning-rate multiplier on restore.
            optimizer: Steps the trainable parameters.
            sequences_per_step: Rollouts drawn from the buffer per batch.
        """
        self.optim_settings = optim_settings
        self.draft_head = draft_head
        self.head_weights = head_weights
        self.sft_objective = sft_objective
        self.rollout_collator = rollout_collator
        self.schedule_factory = schedule_factory
        self.optimizer = optimizer
        self.sequences_per_step = sequences_per_step

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, schedule_factory.create(optim_settings)
        )
        self.step_count = 0
        self.micro_step_count = 0
        self.last_metrics: dict[str, float] = {}
        logger.debug(
            "OnlineTrainer: sequences_per_step=%d, grad_accum_steps=%d, grad_clip=%.3g",
            sequences_per_step,
            optim_settings.grad_accum_steps,
            optim_settings.grad_clip,
        )

    def step(self, replay_batch: ReplayBatch) -> dict[str, float]:
        """Run one micro-batch, stepping the optimizer once per accumulation cycle.

        Args:
            replay_batch: The batch to train on.

        Returns:
            Metrics for this micro-step, including whether the optimizer stepped.
        """
        optim_settings = self.optim_settings
        began = time.perf_counter()

        self.draft_head.train()
        student_hidden = self.draft_head.replay(replay_batch)
        loss, metrics = self.sft_objective(
            student_hidden=student_hidden, replay_batch=replay_batch
        )
        (loss / optim_settings.grad_accum_steps).backward()

        self.micro_step_count += 1
        metrics["micro_step"] = float(self.micro_step_count)

        if self.micro_step_count % optim_settings.grad_accum_steps == 0:
            parameters = self.draft_head.trainable_parameters()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, optim_settings.grad_clip
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
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

    def step_from_buffer(
        self, rollout_buffer: RolloutBuffer
    ) -> dict[str, float] | None:
        """Draw, collate and train on one batch.

        Args:
            rollout_buffer: The pool to draw from.

        Returns:
            The step's metrics, or `None` when the pool is too thin or no drawn
            rollout yields a block.
        """
        if not rollout_buffer.can_sample(self.sequences_per_step):
            return None
        replay_batch = self.rollout_collator.collate(
            rollout_buffer.sample(self.sequences_per_step)
        )
        if replay_batch is None:
            return None
        return self.step(replay_batch)

    def state_dict(self) -> dict:
        """Everything needed to resume: weights, moments and schedule position."""
        return {
            "head": self.draft_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "step_count": self.step_count,
            "micro_step_count": self.micro_step_count,
        }

    def load_state_dict(self, trainer_state: dict) -> None:
        """Restore a run, keeping this run's schedule settings.

        `LambdaLR.state_dict()` saves the `__dict__` of any callable that is not a
        plain function, and `load_state_dict` writes it back. Restoring it would pin
        the multiplier to the checkpoint's `total_steps` rather than this run's, so
        the lambdas are rebuilt afterwards and only the step position is kept.

        Args:
            trainer_state: A mapping from `state_dict`.
        """
        self.draft_head.load_state_dict(trainer_state["head"])
        self.optimizer.load_state_dict(trainer_state["optimizer"])
        if trainer_state.get("scheduler") is not None:
            self.lr_scheduler.load_state_dict(trainer_state["scheduler"])
            self.lr_scheduler.lr_lambdas = [
                self.schedule_factory.create(self.optim_settings)
                for _ in self.lr_scheduler.lr_lambdas
            ]
        self.step_count = int(trainer_state.get("step_count", 0))
        self.micro_step_count = int(trainer_state.get("micro_step_count", 0))
        self.head_weights.freeze_shared(self.draft_head)
        logger.debug(
            "OnlineTrainer resumed at step_count=%d, micro_step_count=%d",
            self.step_count,
            self.micro_step_count,
        )
