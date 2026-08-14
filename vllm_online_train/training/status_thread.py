import threading

from vllm_online_train.config.settings import BookkeepingSettings
from vllm_online_train.contracts.runner import StepRunner
from vllm_online_train.contracts.sink import MetricsSink
from vllm_online_train.logger import init_logger
from vllm_online_train.training.idle_gate import IdleGate

logger = init_logger(__name__)


class StatusThread:
    def __init__(
        self,
        idle_gate: IdleGate,
        step_runner: StepRunner,
        bookkeeping_settings: BookkeepingSettings,
        metrics_sink: MetricsSink,
    ) -> None:
        """Writes a `status` record on a wall-clock cadence.

        A `train` record only appears once a micro-step has succeeded, so a run whose
        gate never opens or whose pool never fills writes nothing at all. This reports
        the same capture, buffer and gate counters on a timer instead, on its own thread
        so a slow or wedged micro-step cannot delay it.

        Every counter it reads is a plain integer or a snapshot of one, so it takes no
        lock and blocks neither the engine thread nor the trainer.

        Args:
            idle_gate: Supplies the gate and engine-step counters.
            step_runner: Supplies the capture, buffer and training counters.
            bookkeeping_settings: Supplies the cadence.
            metrics_sink: Receives the records.
        """
        self.idle_gate = idle_gate
        self.step_runner = step_runner
        self.bookkeeping_settings = bookkeeping_settings
        self.metrics_sink = metrics_sink
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reports = 0
        self.errors = 0

    @property
    def enabled(self) -> bool:
        """Whether a cadence is configured."""
        return self.bookkeeping_settings.status_every_ms > 0

    def start(self) -> None:
        """Start the reporting thread. Idempotent, and a no-op when disabled."""
        if self._thread is not None or not self.enabled:
            return
        # Daemon: process shutdown does not wait for it.
        self._thread = threading.Thread(
            target=self._run, name="online-train-status", daemon=True
        )
        self._thread.start()
        logger.info(
            "Online training status thread started (status_every_ms=%.0f)",
            self.bookkeeping_settings.status_every_ms,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to exit and join it.

        Args:
            timeout: Seconds to wait for the join.
        """
        logger.debug("Stopping online train status thread")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        """Report on the cadence until stopped."""
        status_seconds = self.bookkeeping_settings.status_seconds
        # Waits before the first report, so it does not duplicate the `start` record.
        while not self._stop.wait(status_seconds):
            self._report()

    def _report(self) -> None:
        """Write one snapshot to the sink and the logger.

        Swallows every failure: a counter this cannot read is not a reason to stop
        reporting, let alone to take serving down.
        """
        try:
            status_record = self.step_runner.metrics()
            status_record.update(self.idle_gate.as_metrics())
        except Exception:
            self.errors += 1
            logger.exception(
                "Online training status snapshot failed (%d so far)", self.errors
            )
            return

        self.reports += 1
        self.metrics_sink.write("status", status_record)
        logger.info(
            "Online training status: pool %d rollouts / %d tokens (%.0f%% full, "
            "%d eligible), %d capturing, %d admitted, %d dropped, gate open %.1f%%, "
            "step %d",
            int(status_record.get("buffer/rollouts", 0)),
            int(status_record.get("buffer/tokens", 0)),
            100.0 * status_record.get("buffer/fill", 0.0),
            int(status_record.get("buffer/eligible", 0)),
            int(status_record.get("buffer/pending", 0)),
            int(status_record.get("buffer/admitted", 0)),
            int(self._num_dropped(status_record)),
            100.0 * status_record.get("gate/open_fraction", 0.0),
            int(status_record.get("train/step", 0)),
        )

    @staticmethod
    def _num_dropped(status_record: dict[str, float]) -> float:
        """Rollouts dropped for any reason.

        Args:
            status_record: The snapshot, holding one `buffer/dropped_*` per reason.

        Returns:
            The sum across every reason.
        """
        return sum(
            value
            for name, value in status_record.items()
            if name.startswith("buffer/dropped_")
        )
