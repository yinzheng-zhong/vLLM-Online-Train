from vllm_online_train.contracts.sink import MetricsSink
from vllm_online_train.logger import init_logger
from vllm_online_train.step import EngineStep
from vllm_online_train.training.idle_gate import IdleGate
from vllm_online_train.training.manager import OnlineTrainManager
from vllm_online_train.training.status_thread import StatusThread
from vllm_online_train.training.trainer_thread import TrainerThread

logger = init_logger(__name__)


class OnlineTrainSession:
    def __init__(
        self,
        online_train_manager: OnlineTrainManager,
        idle_gate: IdleGate,
        trainer_thread: TrainerThread,
        status_thread: StatusThread,
        metrics_sink: MetricsSink,
    ) -> None:
        """One handle over everything the capture hook drives.

        Args:
            online_train_manager: Owns the capture, the pool, the head and the
                optimizer.
            idle_gate: Records engine steps and decides when training may run.
            trainer_thread: Polls the gate and runs micro-batches.
            status_thread: Reports the capture and pool counters on a timer.
            metrics_sink: Receives the metric records.
        """
        self.online_train_manager = online_train_manager
        self.idle_gate = idle_gate
        self.trainer_thread = trainer_thread
        self.status_thread = status_thread
        self.metrics_sink = metrics_sink

    def start(self) -> None:
        """Start the trainer and status threads, and record the run's shapes."""
        self.trainer_thread.start()
        self.status_thread.start()
        resolved_config = self.online_train_manager.resolved_config
        engine_shapes = resolved_config.engine_shapes
        online_train_settings = resolved_config.online_train_settings
        buffer_capacity_tokens = (
            online_train_settings.buffer.buffer_capacity_tokens
        )
        publish_mode = online_train_settings.publish.publish_mode
        logger.debug(
            "Starting online train session (block_size=%d, buffer_capacity_tokens=%d, "
            "publish_mode=%s)",
            engine_shapes.block_size,
            buffer_capacity_tokens,
            publish_mode,
        )
        self.metrics_sink.write(
            "start",
            {
                "features": engine_shapes.num_features,
                "aux_layer_ids": list(engine_shapes.aux_layer_ids),
                "block_size": engine_shapes.block_size,
                "bytes_per_token": engine_shapes.bytes_per_token,
                "buffer_capacity_tokens": buffer_capacity_tokens,
                "publish_mode": publish_mode,
            },
        )

    def note_step(self, num_scheduled_tokens: int) -> None:
        """Record that an engine step just ran.

        Args:
            num_scheduled_tokens: Tokens that step scheduled.
        """
        self.idle_gate.note_step(num_scheduled_tokens)

    def observe(self, engine_step: EngineStep) -> None:
        """Tee one engine step's activations.

        Args:
            engine_step: The engine step's schedule and activations.
        """
        self.online_train_manager.observe(engine_step)

    def stop(self) -> None:
        """Stop both threads and release the metrics sink."""
        logger.debug("Stopping online train session")
        self.trainer_thread.stop()
        self.status_thread.stop()
        self.metrics_sink.close()
