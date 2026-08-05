from vllm.logger import init_logger

from vllm_online_train.contracts.sink import MetricsSink
from vllm_online_train.step import EngineStep
from vllm_online_train.train.gate import IdleGate
from vllm_online_train.train.manager import OnlineTrainManager
from vllm_online_train.train.thread import TrainerThread

logger = init_logger(__name__)


class OnlineTrainSession:
    def __init__(
        self,
        manager: OnlineTrainManager,
        gate: IdleGate,
        thread: TrainerThread,
        sink: MetricsSink,
    ) -> None:
        """One handle over everything the capture hook drives.

        Args:
            manager: Owns the capture, the pool, the head and the optimizer.
            gate: Records engine steps and decides when training may run.
            thread: Polls the gate and runs micro-batches.
            sink: Receives the metric records.
        """
        self.manager = manager
        self.gate = gate
        self.thread = thread
        self.sink = sink

    def start(self) -> None:
        """Start the trainer thread and record the run's shapes."""
        self.thread.start()
        shapes = self.manager.config.shapes
        settings = self.manager.config.settings
        logger.debug(
            "Starting online train session (block_size=%d, buffer_capacity_tokens=%d, "
            "publish_mode=%s)",
            shapes.block_size,
            settings.buffer.buffer_capacity_tokens,
            settings.publish.publish_mode,
        )
        self.sink.write(
            "start",
            {
                "features": shapes.num_features,
                "aux_layer_ids": list(shapes.aux_layer_ids),
                "block_size": shapes.block_size,
                "bytes_per_token": shapes.bytes_per_token,
                "buffer_capacity_tokens": settings.buffer.buffer_capacity_tokens,
                "publish_mode": settings.publish.publish_mode,
            },
        )

    def note_step(self, num_scheduled_tokens: int) -> None:
        """Record that an engine step just ran.

        Args:
            num_scheduled_tokens: Tokens that step scheduled.
        """
        self.gate.note_step(num_scheduled_tokens)

    def observe(self, step: EngineStep) -> None:
        """Tee one engine step's activations.

        Args:
            step: The engine step's schedule and activations.
        """
        self.manager.observe(step)

    def stop(self) -> None:
        """Stop the trainer thread and release the metrics sink."""
        logger.debug("Stopping online train session")
        self.thread.stop()
        self.sink.close()
