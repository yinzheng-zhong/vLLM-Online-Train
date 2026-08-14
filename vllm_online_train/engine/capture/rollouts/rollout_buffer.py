import torch

from vllm_online_train.config.settings import BufferSettings
from vllm_online_train.config.shapes import EngineShapes
from vllm_online_train.engine.capture.rollouts.pooled_rollout_sampler import (
    PooledRolloutSampler,
)
from vllm_online_train.engine.capture.rollouts.rollout_records import (
    BufferStats,
    PendingRollout,
    RolloutRecord,
)
from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class RolloutBuffer:
    def __init__(
        self,
        buffer_settings: BufferSettings,
        engine_shapes: EngineShapes,
        pooled_rollout_sampler: PooledRolloutSampler,
        buffer_stats: BufferStats,
    ) -> None:
        """Host-side pool of finished rollouts, with capacity counted in tokens.

        Requests accumulate into a `PendingRollout` as their chunks arrive, then are
        concatenated into a `RolloutRecord` and admitted to the pool when they finish.

        This is the boundary between the two threads. The engine thread is the only
        writer of the pool and of every admitted record's tensors, and the trainer
        thread writes only each record's `draws` counter, so a draw needs no lock --
        only a pool that cannot be resized underneath it. A record that reaches
        `max_draws_per_rollout` stops being drawn immediately and is dropped from the
        pool by the engine thread on the next admission.

        Args:
            buffer_settings: Capacity and the length bounds.
            engine_shapes: Supplies `D`, which sets the minimum usable rollout length.
            pooled_rollout_sampler: Draws training batches from the pool.
            buffer_stats: Counters this buffer increments.
        """
        self.buffer_settings = buffer_settings
        self.engine_shapes = engine_shapes
        self.pooled_rollout_sampler = pooled_rollout_sampler
        self.buffer_stats = buffer_stats
        self._pending_rollouts: dict[str, PendingRollout] = {}
        self._pooled_rollouts: list[RolloutRecord] = []
        self._pooled_tokens = 0

    @property
    def num_rollouts(self) -> int:
        """Rollouts admitted to the pool."""
        return len(self._pooled_rollouts)

    @property
    def num_tokens(self) -> int:
        """Tokens the pool holds."""
        return self._pooled_tokens

    @property
    def num_pending(self) -> int:
        """Requests still accumulating."""
        return len(self._pending_rollouts)

    @property
    def num_eligible(self) -> int:
        """Pooled rollouts still under the draw cap."""
        return len(self.pooled_rollout_sampler.eligible(self._pooled_rollouts))

    def can_sample(self, count: int) -> bool:
        """Whether the pool holds at least `count` rollouts still under the draw cap.

        Args:
            count: Rollouts a training batch needs.
        """
        return self.num_eligible >= count

    def begin(self, req_id: str, prompt_len: int) -> None:
        """Register a request. Idempotent across a chunked prefill.

        Args:
            req_id: The engine's request id.
            prompt_len: Tokens belonging to the prompt.
        """
        if req_id not in self._pending_rollouts:
            self._pending_rollouts[req_id] = PendingRollout(prompt_len=prompt_len)

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
        pending_rollout = self._pending_rollouts.get(req_id)
        if pending_rollout is None or pending_rollout.dropped:
            return

        num_chunk_tokens = int(token_ids.shape[0])
        if num_chunk_tokens == 0:
            return
        if (
            features.shape[0] != num_chunk_tokens
            or final_hidden.shape[0] != num_chunk_tokens
        ):
            raise ValueError(
                f"chunk for {req_id} disagrees on length: "
                f"token_ids={num_chunk_tokens}, features={features.shape[0]}, "
                f"final_hidden={final_hidden.shape[0]}"
            )

        max_rollout_tokens = self.buffer_settings.max_rollout_tokens
        if pending_rollout.num_tokens + num_chunk_tokens > max_rollout_tokens:
            self.drop(req_id, reason="too_long")
            return

        pending_rollout.append(token_ids, features, final_hidden)
        self.buffer_stats.tokens_captured += num_chunk_tokens

    def drop(self, req_id: str, *, reason: str) -> None:
        """Discard a rollout in progress and stop accumulating for it.

        The entry is kept, marked dropped, so later chunks are ignored rather than
        starting a fresh rollout whose position 0 is not the start of the sequence.

        Args:
            req_id: The engine's request id.
            reason: Names the `dropped_<reason>` counter to increment.
        """
        pending_rollout = self._pending_rollouts.get(req_id)
        if pending_rollout is None:
            self._pending_rollouts[req_id] = PendingRollout(
                prompt_len=0, dropped=True
            )
        elif not pending_rollout.dropped:
            pending_rollout.discard()
        self.buffer_stats.count_drop(reason)
        logger.debug("dropped rollout %s: reason=%s", req_id, reason)

    def seal(self, req_id: str) -> RolloutRecord | None:
        """Seal a request and admit it to the pool.

        Args:
            req_id: The engine's request id.

        Returns:
            The admitted record, or `None` when the rollout was dropped or is too
            short to yield an anchor.
        """
        pending_rollout = self._pending_rollouts.pop(req_id, None)
        if pending_rollout is None or pending_rollout.dropped:
            return None
        if pending_rollout.num_tokens < self.buffer_settings.min_rollout_tokens:
            self.buffer_stats.dropped_too_short += 1
            return None

        rollout_record = RolloutRecord(
            token_ids=torch.cat(pending_rollout.token_ids),
            features=torch.cat(pending_rollout.features),
            final_hidden=torch.cat(pending_rollout.final_hidden),
            prompt_len=pending_rollout.prompt_len,
        )
        # Mirrors `BlockBuilder.build`'s range exactly. The bound is `num_draft_tokens`
        # rather than `block_size`, because slot 0 re-feeds the anchor.
        first_anchor = max(pending_rollout.prompt_len - 1, 0)
        num_draft_tokens = self.engine_shapes.num_draft_tokens
        if rollout_record.num_tokens <= first_anchor + num_draft_tokens:
            self.buffer_stats.dropped_too_short += 1
            return None

        self._admit(rollout_record)
        logger.debug(
            "admitted rollout %s: %d tokens, pool now %d rollouts/%d tokens",
            req_id,
            rollout_record.num_tokens,
            self.num_rollouts,
            self.num_tokens,
        )
        return rollout_record

    def _admit(self, rollout_record: RolloutRecord) -> None:
        """Add a record to the pool, retiring spent records and evicting the oldest
        until capacity holds."""
        self._pooled_rollouts.append(rollout_record)
        self._pooled_tokens += rollout_record.num_tokens
        self.buffer_stats.admitted += 1
        self._retire_spent()
        while self._pooled_tokens > self.buffer_settings.buffer_capacity_tokens:
            evicted_record = self._pooled_rollouts.pop(0)
            self._pooled_tokens -= evicted_record.num_tokens
            self.buffer_stats.evicted += 1

    def _retire_spent(self) -> None:
        """Drop the records that have reached the draw cap, freeing their tensors.

        The pool is rebound rather than mutated in place, so a draw that read the old
        list still sees a complete pool. `list.remove` is unusable here: the record's
        generated `__eq__` compares tensors and raises on the first non-identical
        element it reaches.
        """
        pooled_rollouts = self._pooled_rollouts
        kept_records = self.pooled_rollout_sampler.eligible(pooled_rollouts)
        if len(kept_records) == len(pooled_rollouts):
            return
        self._pooled_rollouts = kept_records
        self._pooled_tokens = sum(record.num_tokens for record in kept_records)
        self.buffer_stats.retired += len(pooled_rollouts) - len(kept_records)

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
        return self.pooled_rollout_sampler.draw(list(self._pooled_rollouts), count)

    def clear(self) -> None:
        """Drop every pending and pooled rollout."""
        self._pending_rollouts.clear()
        self._pooled_rollouts.clear()
        self._pooled_tokens = 0

    def as_metrics(self) -> dict[str, float]:
        """Counters plus the current fill and reuse, under a `buffer/` prefix."""
        pooled_rollouts = self._pooled_rollouts
        metrics = self.buffer_stats.as_metrics()
        metrics["buffer/rollouts"] = float(len(pooled_rollouts))
        metrics["buffer/tokens"] = float(self._pooled_tokens)
        metrics["buffer/pending"] = float(len(self._pending_rollouts))
        metrics["buffer/fill"] = float(
            self._pooled_tokens / self.buffer_settings.buffer_capacity_tokens
        )
        metrics["buffer/eligible"] = float(
            len(self.pooled_rollout_sampler.eligible(pooled_rollouts))
        )
        metrics["buffer/mean_draws"] = float(
            sum(record.draws for record in pooled_rollouts) / len(pooled_rollouts)
            if pooled_rollouts
            else 0.0
        )
        return metrics
