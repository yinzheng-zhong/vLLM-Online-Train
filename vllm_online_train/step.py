from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_input_batch import InputBatch


@dataclass(frozen=True)
class EngineStep:
    """One engine step's schedule and activations, read off the model runner."""

    scheduler_output: "SchedulerOutput"
    """This step's schedule."""

    input_batch: "InputBatch"
    """Gives the request order the flat batch is laid out in."""

    input_ids: torch.Tensor
    """`[>=total_num_scheduled_tokens]` flat input token ids."""

    aux_hidden_states: list[torch.Tensor] | None
    """Per-feature-layer `[>=T, hidden_size]` activations in aux-layer order, or
    `None`."""

    hidden_states: torch.Tensor
    """`[>=T, hidden_size]` post-norm final activations."""

    sampled_token_ids: torch.Tensor
    """`[num_reqs, num_spec_tokens + 1]`, `-1` where rejected."""

    spec_decode_metadata: "SpecDecodeMetadata | None"
    """Per-request draft counts, or `None`."""
