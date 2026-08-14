from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricsSink(Protocol):
    """Receives named metric records from the trainer and status threads."""

    def write(self, event: str, field_values: dict[str, Any]) -> None:
        """Append one record.

        Args:
            event: Record kind, e.g. `"train"` or `"start"`.
            field_values: Field name to value.
        """
        ...

    def close(self) -> None:
        """Release the underlying handle."""
        ...
