from vllm.logger import init_logger

from vllm_online_train.contracts.session import TrainingSession

logger = init_logger(__name__)


class CaptureState:
    """Per-process capture stat."""

    def __init__(self) -> None:
        self.session: TrainingSession | None = None
        self.disabled = False
        self.reason: str | None = None

    @property
    def active(self) -> bool:
        """Whether a session is built and capture is still allowed."""
        return self.session is not None and not self.disabled

    def activate(self, session: TrainingSession) -> None:
        """Adopt a built session.

        Args:
            session: The assembled training subsystem.
        """
        self.session = session
        logger.debug("Online training capture state activated")

    def disable(self, reason: str, *, exc_info: bool = False) -> None:
        """Stop capturing for the rest of the process. Idempotent.

        Args:
            reason: Recorded and logged.
            exc_info: Log at exception level, including the traceback.
        """
        if self.disabled:
            return
        self.disabled = True
        self.reason = reason
        self.shutdown()
        if exc_info:
            logger.exception("Online training capture disabled: %s", reason)
        else:
            logger.info("Online training inactive: %s", reason)

    def shutdown(self) -> None:
        """Stop the session, if one is running."""
        if self.session is not None:
            self.session.stop()
            self.session = None

    def reset(self) -> None:
        """Return to the state a fresh process starts in."""
        self.shutdown()
        self.disabled = False
        self.reason = None
        logger.debug("Online training capture state reset")
