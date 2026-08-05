import threading

from vllm.logger import init_logger

from vllm_online_train.config.settings import BookkeepingSettings, GateSettings
from vllm_online_train.contracts.runner import StepRunner
from vllm_online_train.contracts.sink import MetricsSink
from vllm_online_train.train.gate import IdleGate

logger = init_logger(__name__)


class TrainerThread:
    """Polls the gate and runs training micro-batches while it is open.

    The gate is re-checked between micro-batches, so a step that lands mid-opening
    ends it: micro-batch duration is the latency quantum, not the poll interval.
    """

    def __init__(
        self,
        gate: IdleGate,
        runner: StepRunner,
        settings: GateSettings,
        bookkeeping: BookkeepingSettings,
        sink: MetricsSink,
    ) -> None:
        """
        Args:
            gate: Decides when training may hold the GPU.
            runner: Runs one micro-step and reports counters.
            settings: Poll interval and the per-opening cap.
            bookkeeping: Supplies the logging cadence.
            sink: Receives the metric records.
        """
        self.gate = gate
        self.runner = runner
        self.settings = settings
        self.bookkeeping = bookkeeping
        self.sink = sink
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.micro_steps = 0
        self.errors = 0

    def start(self) -> None:
        """Start the polling thread. Idempotent."""
        if self._thread is not None:
            return
        # Daemon: process shutdown does not wait for it.
        self._thread = threading.Thread(
            target=self._run, name="online-trainer", daemon=True
        )
        self._thread.start()
        logger.info(
            "Online trainer thread started (idle_ms=%.0f, poll_ms=%.0f, "
            "busy_tokens=%d)",
            self.settings.idle_ms,
            self.settings.poll_ms,
            self.settings.busy_tokens,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to exit and join it.

        Args:
            timeout: Seconds to wait for the join.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        """Poll the gate, training in bursts while it stays open."""
        poll = self.settings.poll_seconds
        cap = self.settings.max_micro_steps_per_gate
        while not self._stop.is_set():
            if not self.gate.is_open():
                self.gate.mark_closed()
                self._stop.wait(poll)
                continue

            self.gate.mark_opened()
            in_this_opening = 0
            starved = False
            while (
                not self._stop.is_set()
                and self.gate.is_open()
                and (cap == 0 or in_this_opening < cap)
            ):
                if not self._train_once():
                    starved = True
                    break
                in_this_opening += 1
            self.gate.mark_closed()
            if starved or in_this_opening == 0:
                # Backs off rather than reopening immediately, which would busy-loop
                # on an idle engine with an empty buffer.
                self._stop.wait(poll)

    def _train_once(self) -> bool:
        """Run one micro-batch.

        Returns:
            False when there was nothing to train on or the step raised.
        """
        try:
            metrics = self.runner.train_step()
        except Exception:
            self.errors += 1
            logger.exception(
                "Online training step failed (%d so far); pausing this gate opening.",
                self.errors,
            )
            self.sink.write("train_error", {"errors": float(self.errors)})
            return False

        if metrics is None:
            return False

        self.micro_steps += 1
        log_every = self.bookkeeping.log_every
        if log_every and self.micro_steps % log_every == 0:
            self._log(metrics)
        return True

    def _log(self, metrics: dict[str, float]) -> None:
        """Write one combined record to the sink and the logger.

        Args:
            metrics: This micro-step's metrics.
        """
        record = dict(metrics)
        record.update(self.gate.as_metrics())
        record.update(self.runner.metrics())
        self.sink.write("train", record)
        logger.info(
            "Online training step %d: loss %.4f, top1 %.3f, gate open %.1f%%",
            int(metrics.get("step", 0)),
            metrics.get("loss/total", float("nan")),
            metrics.get("train/top1_agree", float("nan")),
            100.0 * record.get("gate/open_fraction", 0.0),
        )
