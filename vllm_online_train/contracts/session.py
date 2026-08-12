from typing import Protocol, runtime_checkable

from vllm_online_train.step import EngineStep


@runtime_checkable
class TrainingSession(Protocol):
    """Everything the capture hook drives, behind one handle."""

    def note_step(self, num_scheduled_tokens: int) -> None:
        """Record that an engine step just ran.

        Args:
            num_scheduled_tokens: Tokens that step scheduled.
        """
        ...

    def observe(self, engine_step: EngineStep) -> None:
        """Tee one engine step's activations.

        Args:
            engine_step: The engine step's schedule and activations.
        """
        ...

    def stop(self) -> None:
        """Stop the trainer thread and release the metrics sink."""
        ...
