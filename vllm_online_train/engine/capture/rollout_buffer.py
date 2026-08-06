import torch
from vllm.logger import init_logger

from vllm_online_train.config.settings import BufferSettings
from vllm_online_train.config.shapes import EngineShapes
from vllm_online_train.engine.capture.pool_sampler import PoolSampler
from vllm_online_train.engine.capture.records import (
    BufferStats,
    PendingRollout,
    RolloutRecord,
)

logger = init_logger(__name__)


class RolloutBuffer:
    def __init__(
        self,
        settings: BufferSettings,
        shapes: EngineShapes,
        sampler: PoolSampler,
        stats: BufferStats,
    ) -> None:
        """Host-side pool of finished rollouts, with capacity counted in tokens.

        Requests accumulate into a `PendingRollout` as their chunks arrive, then are
        concatenated into a `RolloutRecord` and admitted to the pool when they finish.

        This is the boundary between the two threads. The engine thread is the only
        writer; the trainer thread only reads, and an admitted record is immutable, so
        a draw needs no lock -- only a pool that cannot be resized underneath it.

        Args:
            settings: Capacity and the length bounds.
            shapes: Supplies `D`, which sets the minimum usable rollout length.
            sampler: Draws training batches from the pool.
            stats: Counters this buffer increments.
        """
        self.settings = settings
        self.shapes = shapes
        self.sampler = sampler
        self.stats = stats
        self._pending: dict[str, PendingRollout] = {}
        self._pool: list[RolloutRecord] = []
        self._pool_tokens = 0

    @property
    def num_rollouts(self) -> int:
        """Rollouts admitted to the pool."""
        return len(self._pool)

    @property
    def num_tokens(self) -> int:
        """Tokens the pool holds."""
        return self._pool_tokens

    @property
    def num_pending(self) -> int:
        """Requests still accumulating."""
        return len(self._pending)

    def can_sample(self, count: int) -> bool:
        """Whether the pool holds at least `count` rollouts.

        Args:
            count: Rollouts a training batch needs.
        """
        return len(self._pool) >= count

    def begin(self, req_id: str, prompt_len: int) -> None:
        """Register a request. Idempotent across a chunked prefill.

        Args:
            req_id: The engine's request id.
            prompt_len: Tokens belonging to the prompt.
        """
        if req_id not in self._pending:
            self._pending[req_id] = PendingRollout(prompt_len=prompt_len)

    def add_chunk(
        self,
        req_id: str,
        token_ids: torch.Tensor,
        features: torch.Tensor,
        final_hidden: torch.Tensor,
    ) -> None:
        """Append one step's worth of valid positions.

        Chunks must arrive in sequence order and hold only positions conditioned on
        accepted tokens.

        Args:
            req_id: The engine's request id.
            token_ids: `[n]` token ids at the captured positions.
            features: `[n, feature_width]` target features.
            final_hidden: `[n, hidden_size]` post-norm final activations.

        Raises:
            ValueError: If the three tensors disagree on length.
        """
        pending = self._pending.get(req_id)
        if pending is None or pending.dropped:
            return

        num = int(token_ids.shape[0])
        if num == 0:
            return
        if features.shape[0] != num or final_hidden.shape[0] != num:
            raise ValueError(
                f"chunk for {req_id} disagrees on length: token_ids={num}, "
                f"features={features.shape[0]}, "
                f"final_hidden={final_hidden.shape[0]}"
            )

        if pending.num_tokens + num > self.settings.max_rollout_tokens:
            self.drop(req_id, reason="too_long")
            return

        pending.append(token_ids, features, final_hidden)
        self.stats.tokens_captured += num

    def drop(self, req_id: str, *, reason: str) -> None:
        """Discard a rollout in progress and stop accumulating for it.

        The entry is kept, marked dropped, so later chunks are ignored rather than
        starting a fresh rollout whose position 0 is not the start of the sequence.

        Args:
            req_id: The engine's request id.
            reason: Names the `dropped_<reason>` counter to increment.
        """
        pending = self._pending.get(req_id)
        if pending is None:
            self._pending[req_id] = PendingRollout(prompt_len=0, dropped=True)
        elif not pending.dropped:
            pending.discard()
        self.stats.count_drop(reason)
        logger.debug("dropped rollout %s: reason=%s", req_id, reason)

    def finish(self, req_id: str) -> RolloutRecord | None:
        """Seal a request and admit it to the pool.

        Args:
            req_id: The engine's request id.

        Returns:
            The admitted record, or `None` when the rollout was dropped or is too
            short to yield an anchor.
        """
        pending = self._pending.pop(req_id, None)
        if pending is None or pending.dropped:
            return None
        if pending.num_tokens < self.settings.min_rollout_tokens:
            self.stats.dropped_too_short += 1
            return None

        record = RolloutRecord(
            token_ids=torch.cat(pending.token_ids),
            features=torch.cat(pending.features),
            final_hidden=torch.cat(pending.final_hidden),
            prompt_len=pending.prompt_len,
        )
        # Mirrors `BlockBuilder.build`'s range exactly. The bound is `num_draft_tokens`
        # rather than `block_size`, because slot 0 re-feeds the anchor.
        first_anchor = max(pending.prompt_len - 1, 0)
        if record.num_tokens <= first_anchor + self.shapes.num_draft_tokens:
            self.stats.dropped_too_short += 1
            return None

        self._admit(record)
        logger.debug(
            "admitted rollout %s: %d tokens, pool now %d rollouts/%d tokens",
            req_id,
            record.num_tokens,
            self.num_rollouts,
            self.num_tokens,
        )
        return record

    def _admit(self, record: RolloutRecord) -> None:
        """Add a record to the pool, evicting the oldest until capacity holds."""
        self._pool.append(record)
        self._pool_tokens += record.num_tokens
        self.stats.admitted += 1
        while self._pool_tokens > self.settings.buffer_capacity_tokens:
            evicted = self._pool.pop(0)
            self._pool_tokens -= evicted.num_tokens
            self.stats.evicted += 1

    def sample(self, count: int) -> list[RolloutRecord]:
        """Draw a training batch without removing it from the pool.

        The sampler is handed a snapshot rather than the live list, taken in one
        bytecode: an eviction landing inside a draw would otherwise index past the end
        of the pool the count was read from.

        Args:
            count: Rollouts to draw.

        Returns:
            The drawn rollouts.
        """
        return self.sampler.draw(list(self._pool), count)

    def clear(self) -> None:
        """Drop every pending and pooled rollout."""
        self._pending.clear()
        self._pool.clear()
        self._pool_tokens = 0

    def as_metrics(self) -> dict[str, float]:
        """Counters plus the current fill, under a `buffer/` prefix."""
        metrics = self.stats.as_metrics()
        metrics["buffer/rollouts"] = float(len(self._pool))
        metrics["buffer/tokens"] = float(self._pool_tokens)
        metrics["buffer/pending"] = float(len(self._pending))
        metrics["buffer/fill"] = float(
            self._pool_tokens / self.settings.buffer_capacity_tokens
        )
        return metrics
