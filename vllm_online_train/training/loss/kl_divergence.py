import torch
import torch.nn.functional as F


class KLDivergence:
    """Full-vocabulary KL between a student and a teacher distribution."""

    def __call__(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        kl_direction: str,
    ) -> torch.Tensor:
        """Per-row KL, accumulated in fp32.

        Args:
            student_logits: `[N, vocab]` draft logits.
            teacher_logits: `[N, vocab]` target logits.
            kl_direction: `"forward"` for `KL(p || q)` with `p` the teacher, anything
                else for `KL(q || p)`.

        Returns:
            `[N]` divergences.
        """
        log_q = F.log_softmax(student_logits.float(), dim=-1)
        log_p = F.log_softmax(teacher_logits.float(), dim=-1)
        if kl_direction == "forward":
            return (log_p.exp() * (log_p - log_q)).sum(-1)
        return (log_q.exp() * (log_q - log_p)).sum(-1)
