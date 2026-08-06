import time

from vllm.logger import init_logger

from vllm_online_train.config.settings import GateSettings

logger = init_logger(__name__)


class IdleGate:
    def __init__(self, settings: GateSettings) -> None:
        """Shared step state, written by the engine thread and read by the trainer.

        Lock-free: both fields are single machine words written by one thread and read
        by another, and the reader only needs a recent value.

        Args:
            settings: Quiet time, busy-token bound and poll interval.
        """
        self.settings = settings
        self.last_step_monotonic = time.monotonic()
        self.last_step_tokens = 0
        self.step_count = 0

        self._open_count = 0
        self._open_seconds = 0.0
        self._opened_at: float | None = None
        self._created = time.monotonic()

    def note_step(self, num_scheduled_tokens: int) -> None:
        """Record that an engine step just ran.

        Args:
            num_scheduled_tokens: Tokens that step scheduled.
        """
        self.last_step_monotonic = time.monotonic()
        self.last_step_tokens = num_scheduled_tokens
        self.step_count += 1

    def is_open(self) -> bool:
        """Whether training may hold the GPU right now."""
        if time.monotonic() - self.last_step_monotonic < self.settings.idle_seconds:
            return False
        if (
            self.settings.busy_tokens
            and self.last_step_tokens > self.settings.busy_tokens
        ):
            return False
        return True

    def mark_opened(self) -> None:
        """Start timing an opening, if one is not already being timed."""
        if self._opened_at is None:
            self._opened_at = time.monotonic()
            self._open_count += 1
            logger.debug("Gate opened (opening #%d)", self._open_count)

    def mark_closed(self) -> None:
        """Stop timing the current opening, if any."""
        if self._opened_at is not None:
            duration = time.monotonic() - self._opened_at
            self._open_seconds += duration
            self._opened_at = None
            logger.debug("Gate closed after %.3fs", duration)

    def as_metrics(self) -> dict[str, float]:
        """Open share, opening count and the engine's step counters."""
        # One clock reading for both terms. Sampling `now` twice lets the open window
        # advance past the elapsed window and reports a fraction above 1.
        now = time.monotonic()
        elapsed = max(now - self._created, 1e-9)
        open_seconds = self._open_seconds
        if self._opened_at is not None:
            open_seconds += now - self._opened_at
        return {
            "gate/open_fraction": open_seconds / elapsed,
            "gate/openings": float(self._open_count),
            "gate/open_seconds": open_seconds,
            "gate/elapsed_seconds": elapsed,
            "engine/steps": float(self.step_count),
            "engine/last_step_tokens": float(self.last_step_tokens),
        }
