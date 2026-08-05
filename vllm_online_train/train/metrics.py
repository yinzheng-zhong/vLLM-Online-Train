import json
import os
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from vllm.logger import init_logger

logger = init_logger(__name__)


class JsonlMetricsSink:
    def __init__(self, path: str | Path | None) -> None:
        """Append-only JSONL writer, safe to call from the trainer thread.

        A no-op when constructed without a path, so callers never branch on whether
        metrics are configured. Every write is flushed, because the engine may be killed
        at any time.

        Args:
            path: Destination. The pid is appended to the stem, so each worker under
                TP>1 writes its own file. `None` disables the sink.
        """
        self._lock = threading.Lock()
        self._handle: TextIO | None = None
        self._path: Path | None = None
        if path is None:
            return

        base = Path(path).expanduser()
        try:
            base.parent.mkdir(parents=True, exist_ok=True)
            stem = base.with_name(f"{base.stem}.{os.getpid()}{base.suffix}")
            self._path = stem
            self._handle = stem.open("a", buffering=1)
            logger.info("Online training metrics -> %s", stem)
        except OSError as exc:
            logger.warning(
                "Could not open online-train metrics path %s (%s); metrics disabled.",
                path,
                exc,
            )
            self._handle = None

    @property
    def path(self) -> Path | None:
        """The file being written, pid suffix included."""
        return self._path

    @property
    def enabled(self) -> bool:
        """Whether writes reach a file."""
        return self._handle is not None

    def write(self, event: str, values: dict[str, Any]) -> None:
        """Append one record. Never raises into the caller.

        Args:
            event: Record kind, e.g. `"train"` or `"start"`.
            values: Field name to value.
        """
        if self._handle is None:
            return
        record = {"t": time.time(), "event": event}
        record.update(values)
        line = json.dumps(record, default=float)
        try:
            with self._lock:
                self._handle.write(line + "\n")
        except (OSError, ValueError) as exc:
            logger.warning("Online training metrics write failed (%s); disabling.", exc)
            self.close()

    def close(self) -> None:
        """Close the handle and disable further writes."""
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                finally:
                    self._handle = None
