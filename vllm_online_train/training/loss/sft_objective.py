from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vllm_online_train.config.settings import ObjectiveSettings
from vllm_online_train.training.collate.replay_batch import ReplayBatch
from vllm_online_train.training.loss.kl_divergence import KLDivergence
from vllm_online_train.training.loss.position_decay import PositionDecay
from vllm_online_train.training.loss.teacher_scorer import TeacherScorer

ProjectFn = Callable[[torch.Tensor], torch.Tensor]


class SFTObjective:
    def __init__(
        self,
        objective_settings: ObjectiveSettings,
        kl_divergence: KLDivergence,
        teacher_scorer: TeacherScorer,
        position_decay: PositionDecay,
        student_project: ProjectFn,
        teacher_project: ProjectFn,
    ) -> None:
        """Full-vocabulary KL against the target, over every scored position.

        Takes hidden states rather than logits so the vocabulary projection lives inside
        the checkpointed chunk: a `[N, vocab]` fp32 log-softmax retained for backward is
        far larger than the `[N, H]` input it was computed from.

        Args:
            objective_settings: KL direction, auxiliary CE weight and chunk size.
            kl_divergence: Computes per-row KL.
            teacher_scorer: Gathers the activations that score each label.
            position_decay: In-block position weighting.
            student_project: `[N, H] -> [N, draft_vocab]`, the draft's LM head. Must
                stay differentiable.
            teacher_project: `[N, H] -> [N, vocab]`, the target's own projection.
        """
        self.objective_settings = objective_settings
        self.kl_divergence = kl_divergence
        self.teacher_scorer = teacher_scorer
        self.position_decay = position_decay
        self.student_project = student_project
        self.teacher_project = teacher_project

    def __call__(
        self, *, student_hidden: torch.Tensor, replay_batch: ReplayBatch
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Score a replayed batch.

        Args:
            student_hidden: `[B, A, D, H]` pre-LM-head states from `replay`.
            replay_batch: The batch those states were produced from.

        Returns:
            `(loss, metrics)`. The loss is a scalar carrying the student's graph.
        """
        objective_settings = self.objective_settings
        num_draft_tokens = replay_batch.num_draft_tokens
        hidden_size = student_hidden.shape[-1]

        accepted_mask, rejected_mask = replay_batch.acceptance_index_masks()
        valid_mask = (accepted_mask | rejected_mask).reshape(-1)
        if not bool(valid_mask.any()):
            return student_hidden.sum() * 0.0, {"loss/total": 0.0}
        valid_index = valid_mask.nonzero(as_tuple=True)[0]

        student_hidden_flat = student_hidden.reshape(-1, hidden_size)[valid_index]
        teacher_hidden_flat = self.teacher_scorer.hidden_for_labels(
            replay_batch, num_draft_tokens
        ).reshape(-1, hidden_size)[valid_index]
        labels = replay_batch.block_token_ids.reshape(-1)[valid_index]
        row_weights = self._row_weights(replay_batch, valid_index)

        num_rows = valid_index.numel()
        chunk_size = max(1, objective_settings.kl_chunk_size)
        weight_sum = row_weights.sum().clamp_min(1e-6)

        loss = student_hidden.sum() * 0.0
        ce_total = student_hidden.sum() * 0.0
        num_agree = 0.0

        for start in range(0, num_rows, chunk_size):
            rows = slice(start, min(start + chunk_size, num_rows))
            chunk_weights = row_weights[rows]

            # Recomputed in backward rather than retained, teacher logits included, so
            # peak is one `[chunk, vocab]` rather than the whole batch's.
            kl = checkpoint(
                self._chunk_kl,
                student_hidden_flat[rows],
                teacher_hidden_flat[rows],
                use_reentrant=False,
            )
            # Normalised by the global weight sum, so the result does not depend on
            # how the batch happened to be chunked.
            loss = loss + (kl * chunk_weights).sum() / weight_sum

            if objective_settings.label_smoothing_ce_weight > 0:
                ce = checkpoint(
                    self._chunk_ce,
                    student_hidden_flat[rows],
                    labels[rows],
                    use_reentrant=False,
                )
                ce_total = ce_total + (ce * chunk_weights).sum() / weight_sum

            with torch.no_grad():
                student_top1 = self.student_project(
                    student_hidden_flat[rows]
                ).argmax(-1)
                num_agree += float((student_top1 == labels[rows]).sum())

        metrics = {
            "loss/kl": float(loss.detach()),
            "count/tokens": float(num_rows),
            "train/top1_agree": num_agree / max(num_rows, 1),
        }
        if objective_settings.label_smoothing_ce_weight > 0:
            loss = loss + objective_settings.label_smoothing_ce_weight * ce_total
            metrics["loss/ce"] = float(ce_total.detach())
        metrics["loss/total"] = float(loss.detach())
        return loss, metrics

    def _row_weights(
        self, replay_batch: ReplayBatch, valid_index: torch.Tensor
    ) -> torch.Tensor:
        """Per-scored-position weights, flattened to the selected rows.

        Args:
            replay_batch: The batch being scored.
            valid_index: Flat indices of the valid positions.

        Returns:
            `[len(valid_index)]` fp32 weights; all ones under a uniform mean.
        """
        weights = self.position_decay.weights(replay_batch)
        if weights is None:
            return torch.ones_like(valid_index, dtype=torch.float32)
        return weights.reshape(-1)[valid_index].float()

    def _chunk_kl(
        self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Project one chunk to both vocabularies and take the KL.

        Args:
            student_hidden: `[n, H]` draft states.
            teacher_hidden: `[n, H]` target states.

        Returns:
            `[n]` divergences.

        Raises:
            ValueError: If the two vocabularies differ in size.
        """
        student_logits = self.student_project(student_hidden)
        with torch.no_grad():
            teacher_logits = self.teacher_project(teacher_hidden)
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            raise ValueError(
                f"student vocabulary is {student_logits.shape[-1]} but the target's "
                f"is {teacher_logits.shape[-1]}. A reduced draft vocabulary needs "
                "the teacher restricted to the same support before KL, or forward "
                "KL diverges on the tokens the student cannot represent."
            )
        return self.kl_divergence(
            student_logits, teacher_logits, self.objective_settings.kl_direction
        )

    def _chunk_ce(
        self, student_hidden: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy of one chunk against the emitted tokens.

        Args:
            student_hidden: `[n, H]` draft states.
            labels: `[n]` token ids.

        Returns:
            `[n]` per-row losses.
        """
        logits = self.student_project(student_hidden)
        return F.cross_entropy(logits.float(), labels, reduction="none")
