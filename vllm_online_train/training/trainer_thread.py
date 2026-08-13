import threading

from vllm_online_train.config.settings import BookkeepingSettings, GateSettings
from vllm_online_train.contracts.runner import StepRunner
from vllm_online_train.contracts.sink import MetricsSink
from vllm_online_train.logger import init_logger
from vllm_online_train.training.idle_gate import IdleGate

logger = init_logger(__name__)


class TrainerThread:
    def __init__(
        self,
        idle_gate: IdleGate,
        step_runner: StepRunner,
        gate_settings: GateSettings,
        bookkeeping_settings: BookkeepingSettings,
        metrics_sink: MetricsSink,
    ) -> None:
        """Polls the gate and runs training micro-batches while it is open.

        The gate is re-checked between micro-batches, so a step that lands mid-opening
        ends it: micro-batch duration is the latency quantum, not the poll interval.

        Args:
            idle_gate: Decides when training may hold the GPU.
            step_runner: Runs one micro-step and reports counters.
            gate_settings: Poll interval and the per-opening cap.
            bookkeeping_settings: Supplies the logging cadence.
            metrics_sink: Receives the metric records.
        """
        self.idle_gate = idle_gate
        self.step_runner = step_runner
        self.gate_settings = gate_settings
        self.bookkeeping_settings = bookkeeping_settings
        self.metrics_sink = metrics_sink
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
            self.gate_settings.idle_ms,
            self.gate_settings.poll_ms,
            self.gate_settings.busy_tokens,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to exit and join it.

        Args:
            timeout: Seconds to wait for the join.
        """
        logger.debug("Stopping online trainer thread")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        """Poll the gate, training in bursts while it stays open. Main training entry point."""
        poll_seconds = self.gate_settings.poll_seconds
        micro_step_cap = self.gate_settings.max_micro_steps_per_gate
        while not self._stop.is_set():
            if not self.idle_gate.is_open():
                self.idle_gate.mark_closed()
                self._stop.wait(poll_seconds)
                continue

            self.idle_gate.mark_opened()
            micro_steps_this_opening = 0
            starved = False
            while (
                not self._stop.is_set()
                and self.idle_gate.is_open()
                and (
                    micro_step_cap == 0
                    or micro_steps_this_opening < micro_step_cap
                )
            ):
                if not self._train_once():
                    starved = True
                    break
                micro_steps_this_opening += 1
            self.idle_gate.mark_closed()
            if starved or micro_steps_this_opening == 0:
                # Backs off rather than reopening immediately, which would busy-loop
                # on an idle engine with an empty buffer.
                self._stop.wait(poll_seconds)

    def _train_once(self) -> bool:
        """Run one micro-batch.

        Returns:
            False when there was nothing to train on or the step raised.
        """
        try:
            metrics = self.step_runner.train_step()
        except Exception:
            self.errors += 1
            logger.exception(
                "Online training step failed (%d so far); pausing this gate opening.",
                self.errors,
            )
            self.metrics_sink.write("train_error", {"errors": float(self.errors)})
            return False

        if metrics is None:
            return False

        self.micro_steps += 1
        log_every = self.bookkeeping_settings.log_every
        if log_every and self.micro_steps % log_every == 0:
            self._log(metrics)
        return True

    def _log(self, metrics: dict[str, float]) -> None:
        """Write one combined record to the sink and the logger.

        Args:
            metrics: This micro-step's metrics.
        """
        combined_record = dict(metrics)
        combined_record.update(self.idle_gate.as_metrics())
        combined_record.update(self.step_runner.metrics())
        self.metrics_sink.write("train", combined_record)
        logger.info(
            "Online training step %d: loss %.4f, top1 %.3f, gate open %.1f%%",
            int(metrics.get("step", 0)),
            metrics.get("loss/total", float("nan")),
            metrics.get("train/top1_agree", float("nan")),
            100.0 * combined_record.get("gate/open_fraction", 0.0),
        )
