from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class PendingRollout:
    """A request still generating, accumulating chunks in arrival order."""

    prompt_len: int
    token_ids: list[torch.Tensor] = field(default_factory=list)
    features: list[torch.Tensor] = field(default_factory=list)
    final_hidden: list[torch.Tensor] = field(default_factory=list)
    num_tokens: int = 0

    dropped: bool = False
    """Set when the rollout can no longer produce a usable record, so later chunks for
    the same request are ignored."""

    def append(
        self,
        token_ids: torch.Tensor,
        features: torch.Tensor,
        final_hidden: torch.Tensor,
    ) -> None:
        """Add one step's worth of positions.

        Args:
            token_ids: `[n]` token ids at the captured positions.
            features: `[n, feature_width]` target features.
            final_hidden: `[n, hidden_size]` post-norm final activations.
        """
        self.token_ids.append(token_ids)
        self.features.append(features)
        self.final_hidden.append(final_hidden)
        self.num_tokens += int(token_ids.shape[0])

    def discard(self) -> None:
        """Mark the rollout dropped and release the chunks accumulated so far."""
        self.dropped = True
        self.token_ids.clear()
        self.features.clear()
        self.final_hidden.clear()


@dataclass(slots=True)
class RolloutRecord:
    """One finished rollout, aligned position by position.

    Index `t` of every field refers to the same sequence position: `token_ids[t]` is
    the token at `t`, and `features[t]` / `final_hidden[t]` are the target activations
    computed at `t`. The arrays end one token short of everything the request emitted,
    because the final emitted token's activations would come from the step after
    generation stopped.
    """

    token_ids: torch.Tensor
    """`[T]` int32."""

    features: torch.Tensor
    """`[T, num_features * hidden_size]`, layers concatenated in aux-layer order."""

    final_hidden: torch.Tensor
    """`[T, hidden_size]` post-norm final hidden states."""

    prompt_len: int
    """Tokens belonging to the prompt. Anchors start at `prompt_len - 1`."""

    @property
    def num_tokens(self) -> int:
        """Positions the record covers."""
        return int(self.token_ids.shape[0])

    def __post_init__(self) -> None:
        """Check every field covers the same positions.

        Raises:
            ValueError: If the fields disagree on length.
        """
        t = self.num_tokens
        if self.features.shape[0] != t or self.final_hidden.shape[0] != t:
            raise ValueError(
                f"RolloutRecord fields disagree on length: token_ids={t}, "
                f"features={self.features.shape[0]}, "
                f"final_hidden={self.final_hidden.shape[0]}"
            )


@dataclass
class BufferStats:
    """Counters for why rollouts did and did not make it into the pool."""

    admitted: int = 0
    evicted: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_prefix_cache_hit: int = 0
    dropped_not_sampled: int = 0
    dropped_aborted: int = 0
    tokens_captured: int = 0

    def count_drop(self, reason: str) -> None:
        """Increment the `dropped_<reason>` counter, if one exists.

        Args:
            reason: The drop reason, e.g. `"too_long"`.
        """
        counter = f"dropped_{reason}"
        if hasattr(self, counter):
            setattr(self, counter, getattr(self, counter) + 1)

    def as_metrics(self) -> dict[str, float]:
        """Every counter under a `buffer/` prefix."""
        return {f"buffer/{k}": float(v) for k, v in vars(self).items()}
