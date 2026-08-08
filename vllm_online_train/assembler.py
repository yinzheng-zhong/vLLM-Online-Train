import random
from pathlib import Path
from typing import Any

import torch

from vllm_online_train.config.device_placement import DevicePlacement
from vllm_online_train.config.resolver import ConfigResolver
from vllm_online_train.config.settings import OnlineTrainSettings
from vllm_online_train.config.shapes import ResolvedConfig
from vllm_online_train.contracts.provider import StateProvider
from vllm_online_train.engine.capture.feature_stager import FeatureStager
from vllm_online_train.engine.capture.pool_sampler import PoolSampler
from vllm_online_train.engine.capture.records import BufferStats
from vllm_online_train.engine.capture.request_sampler import RequestSampler
from vllm_online_train.engine.capture.rollout_buffer import RolloutBuffer
from vllm_online_train.engine.capture.rollout_capture import RolloutCapture
from vllm_online_train.engine.capture.valid_positions import ValidPositions
from vllm_online_train.engine.state.engine import EngineStateProvider
from vllm_online_train.engine.state.mirrored import MirroredStateProvider
from vllm_online_train.engine.state.target_locator import TargetLocator
from vllm_online_train.logger import init_logger
from vllm_online_train.training.checkpoint.drafter_locator import DrafterLocator
from vllm_online_train.training.checkpoint.exporter import CheckpointExporter
from vllm_online_train.training.checkpoint.weight_names import WeightNameRewriter
from vllm_online_train.training.checkpoint.weight_publisher import WeightPublisher
from vllm_online_train.training.checkpoint.writer import CheckpointWriter
from vllm_online_train.training.collate.anchor_sampler import AnchorSampler
from vllm_online_train.training.collate.block_builder import BlockBuilder
from vllm_online_train.training.collate.device_transfer import DeviceTransfer
from vllm_online_train.training.collate.rollout_collator import RolloutCollator
from vllm_online_train.training.head.arch import ArchFactory
from vllm_online_train.training.head.block_masks import BlockMaskBuilder
from vllm_online_train.training.head.factory import HeadFactory
from vllm_online_train.training.head.trainable_head import TrainableDFlashHead
from vllm_online_train.training.head.weights import HeadWeights
from vllm_online_train.training.idle_gate import IdleGate
from vllm_online_train.training.jsonl_metrics import JsonlMetricsSink
from vllm_online_train.training.loss.kl_divergence import KLDivergence
from vllm_online_train.training.loss.position_decay import PositionDecay
from vllm_online_train.training.loss.sft_objective import SFTObjective
from vllm_online_train.training.loss.teacher_scorer import TeacherScorer
from vllm_online_train.training.manager import OnlineTrainManager
from vllm_online_train.training.optim.schedule_factory import ScheduleFactory
from vllm_online_train.training.optim.trainer import OnlineTrainer
from vllm_online_train.training.session import OnlineTrainSession
from vllm_online_train.training.trainer_thread import TrainerThread

logger = init_logger(__name__)


class SessionAssembler:
    def __init__(self) -> None:
        """Assembles the whole training subsystem from a model runner and settings.

        This is the package's only wiring site for a training session: every
        collaborator is constructed here and injected downwards, and nothing below this
        point looks anything up.

        Construct the collaborators that carry no settings and are shared.
        """
        self.masks = BlockMaskBuilder()
        self.naming = WeightNameRewriter()
        self.weights = HeadWeights()
        self.resolver = ConfigResolver()
        self.placement = DevicePlacement()
        self.arch_factory = ArchFactory()
        self.head_factory = HeadFactory(self.masks, self.weights)
        self.writer = CheckpointWriter(self.naming)
        self.publisher = WeightPublisher(self.naming, self.weights)
        self.locator = DrafterLocator()
        self.target_locator = TargetLocator()
        self.positions = ValidPositions()
        self.blocks = BlockBuilder()
        self.divergence = KLDivergence()
        self.teacher = TeacherScorer(self.masks)
        self.schedules = ScheduleFactory()

    def build(
        self, runner: Any, settings: OnlineTrainSettings
    ) -> OnlineTrainSession | None:
        """Assemble and start a training session.

        Args:
            runner: The `GPUModelRunner` the capture hook is wrapping.
            settings: The operator's settings.

        Returns:
            The started session, or `None` when this engine cannot supply features.

        Raises:
            ValueError: If the engine is configured for DFlash but the draft and
                target are an incompatible pair, or `train_device` names a device this
                process cannot train on.
        """
        vllm_config = getattr(runner, "vllm_config", None)
        spec_config = self.applicable(vllm_config, settings)
        if spec_config is None:
            return None

        config = self.resolver.resolve(vllm_config, settings)
        engine = EngineStateProvider(
            self.target_locator,
            runner.get_model(),
            runner.device,
            vllm_config.model_config.dtype,
        )
        provider = self.build_provider(config, engine)
        manager = self.build_manager(
            runner, vllm_config, config, provider, Path(spec_config.model)
        )
        self.log_summary(config, provider.device)
        return self.start_session(manager, config)

    @staticmethod
    def applicable(vllm_config: Any, settings: OnlineTrainSettings) -> Any | None:
        """Decide whether this engine can supply the features the head trains on.

        Args:
            vllm_config: The engine config, or `None`.
            settings: The operator's settings.

        Returns:
            The speculative config, or `None` after logging the reason.
        """
        if vllm_config is None:
            logger.info("Online training inactive: runner exposes no vllm_config")
            return None
        if not settings.enabled:
            logger.info("Online training inactive: settings set enabled=false")
            return None

        spec_config = getattr(vllm_config, "speculative_config", None)
        if spec_config is None or not spec_config.use_dflash():
            method = None if spec_config is None else spec_config.method
            logger.info(
                "Online training inactive: speculative method is %r, not 'dflash'; "
                "the features the head trains on are a by-product of the DFlash "
                "serving path",
                method,
            )
            return None
        return spec_config

    def build_provider(
        self, config: ResolvedConfig, engine: StateProvider
    ) -> StateProvider:
        """Choose where the borrowed target state is served from.

        Args:
            config: Supplies the requested training device and the shared precision.
            engine: The provider holding the target handle, on the engine's device.

        Returns:
            `engine` itself when the head trains on the serving device, otherwise a
            mirror of it on the training device, verified against it.

        Raises:
            ValueError: If `train_device` names a device this process cannot train on,
                or the mirrored projection does not match the engine's own.
        """
        device = self.placement.resolve(
            config.settings.placement.train_device, engine.device
        )
        if device == engine.device:
            logger.debug("training on the engine's own device %s; no mirror", device)
            return engine

        mirror = MirroredStateProvider(
            engine, device, self.shared_dtype(config, engine)
        )
        deviation = mirror.verify_parity()
        logger.info(
            "Online training on %s, off the engine's %s: the teacher projection is "
            "mirrored there at %s and matches the engine's own to %.2g of its largest "
            "logit",
            device,
            engine.device,
            mirror.projection_dtype,
            deviation,
        )
        return mirror

    def build_manager(
        self,
        runner: Any,
        vllm_config: Any,
        config: ResolvedConfig,
        provider: StateProvider,
        source_dir: Path,
    ) -> OnlineTrainManager:
        """Build the handle that owns the capture, the pool, the head and the loop.

        Args:
            runner: The `GPUModelRunner`, for the live drafter and the engine's device.
            vllm_config: The engine config, for the draft model's shape.
            config: Settings paired with the resolved shapes.
            provider: Owns the target handle everything else borrows through.
            source_dir: The served head's directory.

        Returns:
            The manager, with its publish target already resolved.
        """
        head = self.build_head(vllm_config, config, provider)
        buffer = self.build_buffer(config)
        manager = OnlineTrainManager(
            config=config,
            provider=provider,
            buffer=buffer,
            # The engine's device, not the provider's: the gather runs where the
            # activations are.
            capture=self.build_capture(config, buffer, runner.device),
            head=head,
            trainer=self.build_trainer(config, head, provider),
            exporter=CheckpointExporter(self.writer, self.weights, source_dir),
            publisher=self.publisher,
        )
        manager.set_publish_target(self.locator.find(runner))
        return manager

    def build_head(
        self,
        vllm_config: Any,
        config: ResolvedConfig,
        provider: StateProvider,
    ) -> TrainableDFlashHead:
        """Build the head this engine's drafter is served with.

        Args:
            vllm_config: The engine config, for the draft model's shape.
            config: Settings paired with the resolved shapes.
            provider: Supplies the embedding table and the LM head.

        Returns:
            The head, on the provider's device, trained tensors at fp32.
        """
        arch = self.arch_factory.create(vllm_config, config.shapes)
        head = self.head_factory.create(arch).to(
            device=provider.device, dtype=torch.float32
        )
        self.weights.load_shared(
            head, provider.embedding_weight(), provider.lm_head_weight()
        )
        self.weights.cast_shared(head, self.shared_dtype(config, provider))
        return head

    @staticmethod
    def shared_dtype(config: ResolvedConfig, provider: StateProvider) -> torch.dtype:
        """The precision the borrowed embedding table and LM head are held at.

        Args:
            config: Supplies `shared_dtype`.
            provider: Supplies the serving dtype.

        Returns:
            fp32 when the settings ask for it, otherwise the serving dtype.
        """
        if config.settings.capture.shared_dtype == "float32":
            return torch.float32
        return provider.serving_dtype

    @staticmethod
    def build_buffer(config: ResolvedConfig) -> RolloutBuffer:
        """Build the rollout pool and its draw strategy.

        Args:
            config: Supplies the buffer settings, the shapes and the seed.

        Returns:
            The pool.
        """
        settings = config.settings
        sampler = PoolSampler(
            settings.buffer, random.Random(settings.bookkeeping.seed)
        )
        return RolloutBuffer(settings.buffer, config.shapes, sampler, BufferStats())

    def build_capture(
        self, config: ResolvedConfig, buffer: RolloutBuffer, device: torch.device
    ) -> RolloutCapture:
        """Build the capture and the staging it writes through.

        Args:
            config: Supplies the capture settings, the shapes and the seed.
            buffer: Receives the drained chunks.
            device: Where the gather runs.

        Returns:
            The capture.
        """
        settings = config.settings
        # A chunk larger than `max_rollout_tokens` can only belong to a rollout that
        # will be dropped for length, so staging never needs to be bigger.
        stager = FeatureStager(
            config.shapes, settings.buffer.max_rollout_tokens, device
        )
        sampler = RequestSampler(settings.capture, settings.bookkeeping.seed)
        return RolloutCapture(
            settings.capture, buffer, stager, sampler, self.positions
        )

    def build_trainer(
        self,
        config: ResolvedConfig,
        head: TrainableDFlashHead,
        provider: StateProvider,
    ) -> OnlineTrainer:
        """Build the optimizer loop for one run.

        Args:
            config: Settings paired with the resolved shapes.
            head: The head being trained.
            provider: Supplies the teacher distribution.

        Returns:
            The trainer.
        """
        settings = config.settings
        optimizer = torch.optim.AdamW(
            head.trainable_parameters(),
            lr=settings.optim.learning_rate,
            weight_decay=settings.optim.weight_decay,
        )
        return OnlineTrainer(
            settings.optim,
            head,
            self.weights,
            self.build_objective(config, head, provider),
            self.build_collator(config, provider),
            self.schedules,
            optimizer,
            settings.anchors.sequences_per_step,
        )

    def build_objective(
        self,
        config: ResolvedConfig,
        head: TrainableDFlashHead,
        provider: StateProvider,
    ) -> SFTObjective:
        """Build the loss for one run.

        Args:
            config: Supplies the objective settings and the decay constant.
            head: Supplies the student projection.
            provider: Supplies the teacher projection.

        Returns:
            The objective.
        """
        return SFTObjective(
            config.settings.objective,
            self.divergence,
            self.teacher,
            PositionDecay(config.position_decay_gamma),
            head.project_to_vocab,
            provider.teacher_logits,
        )

    def build_collator(
        self, config: ResolvedConfig, provider: StateProvider
    ) -> RolloutCollator:
        """Build the collator for one run.

        Args:
            config: Supplies the anchor settings, the shapes and the seed.
            provider: Supplies the training device.

        Returns:
            The collator.
        """
        settings = config.settings
        return RolloutCollator(
            self.blocks,
            AnchorSampler(
                settings.anchors, random.Random(settings.bookkeeping.seed)
            ),
            DeviceTransfer(provider.device),
            config.shapes,
        )

    @staticmethod
    def start_session(
        manager: OnlineTrainManager, config: ResolvedConfig
    ) -> OnlineTrainSession:
        """Wire the gate, the thread and the sink around a manager, and start them.

        Args:
            manager: The assembled training subsystem.
            config: Supplies the gate and bookkeeping settings.

        Returns:
            The started session.
        """
        settings = config.settings
        sink = JsonlMetricsSink(settings.bookkeeping.metrics_path)
        gate = IdleGate(settings.gate)
        thread = TrainerThread(
            gate, manager, settings.gate, settings.bookkeeping, sink
        )
        session = OnlineTrainSession(manager, gate, thread, sink)
        session.start()
        return session

    @staticmethod
    def log_summary(config: ResolvedConfig, train_device: torch.device) -> None:
        """Log the shapes and the memory the run will occupy.

        Args:
            config: Settings paired with the resolved shapes.
            train_device: Where the head trains.
        """
        shapes = config.shapes
        logger.info(
            "Online training enabled on %s: %d feature layers %s, block size %d, "
            "%.1f KiB/token, buffer %d tokens (~%.1f GiB host)",
            train_device,
            shapes.num_features,
            list(shapes.aux_layer_ids),
            shapes.block_size,
            shapes.bytes_per_token / 1024,
            config.settings.buffer.buffer_capacity_tokens,
            config.buffer_capacity_bytes / 2**30,
        )
