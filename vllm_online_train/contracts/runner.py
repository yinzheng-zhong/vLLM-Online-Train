from typing import Protocol, runtime_checkable


@runtime_checkable
class StepRunner(Protocol):
    """Runs one training micro-step and reports the subsystem's counters."""

    def train_step(self) -> dict[str, float] | None:
        """Run one micro-step.

        Returns:
            The step's metrics, or `None` when there was nothing to train on.
        """
        ...

    def metrics(self) -> dict[str, float]:
        """Capture, buffer and training counters."""
        ...
