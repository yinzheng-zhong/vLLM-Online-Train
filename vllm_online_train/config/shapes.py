from dataclasses import dataclass

import torch

from vllm_online_train.config.settings import OnlineTrainSettings


@dataclass(frozen=True)
class EngineShapes:
    """Widths and layer ids read off the loaded target and draft configs."""

    aux_layer_ids: tuple[int, ...]
    """Target layers captured as features, in vLLM's aux-layer numbering."""

    hidden_size: int
    """Target hidden width, i.e. the width of one feature."""

    vocab_size: int

    block_size: int
    """`K`: draft query slots per block, including the anchor slot."""

    mask_token_id: int
    feature_dtype: torch.dtype

    @property
    def num_features(self) -> int:
        """Feature layers captured per token."""
        return len(self.aux_layer_ids)

    @property
    def num_draft_tokens(self) -> int:
        """Scored positions per block, `K - 1`."""
        return self.block_size - 1

    @property
    def feature_width(self) -> int:
        """Width of the concatenated feature vector, i.e. `fc.in_features`."""
        return self.num_features * self.hidden_size

    @property
    def bytes_per_token(self) -> int:
        """Buffer cost of one token: features, final hidden state and token id."""
        item = torch.empty((), dtype=self.feature_dtype).element_size()
        return (self.feature_width + self.hidden_size) * item + 4


@dataclass(frozen=True)
class ResolvedConfig:
    """Operator settings paired with the shapes resolved from a live engine."""

    online_train_settings: OnlineTrainSettings
    engine_shapes: EngineShapes

    position_decay_gamma: float
    """`ObjectiveSettings.position_decay_gamma`, with `None` replaced by the default
    for this block size."""

    @property
    def buffer_capacity_bytes(self) -> int:
        """Host memory the full pool would occupy."""
        return (
            self.online_train_settings.buffer.buffer_capacity_tokens
            * self.engine_shapes.bytes_per_token
        )
