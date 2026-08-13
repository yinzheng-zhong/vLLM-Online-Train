from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CapturedChunk:
    """One request's usable positions from a single engine step, on the host."""

    req_id: str

    token_ids: torch.Tensor
    """`[n]` token ids at the captured positions."""

    features: torch.Tensor
    """`[n, feature_width]` target features."""

    final_hidden: torch.Tensor
    """`[n, hidden_size]` post-norm final activations."""
