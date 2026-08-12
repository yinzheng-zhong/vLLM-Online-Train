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
        self.block_mask_builder = BlockMaskBuilder()
        self.weight_name_rewriter = WeightNameRewriter()
        self.head_weights = HeadWeights()
        self.config_resolver = ConfigResolver()
        self.device_placement = DevicePlacement()
        self.arch_factory = ArchFactory()
        self.head_factory = HeadFactory(self.block_mask_builder, self.head_weights)
        self.checkpoint_writer = CheckpointWriter(self.weight_name_rewriter)
        self.weight_publisher = WeightPublisher(
            self.weight_name_rewriter, self.head_weights
        )
        self.drafter_locator = DrafterLocator()
        self.target_locator = TargetLocator()
        self.valid_positions = ValidPositions()
        self.block_builder = BlockBuilder()
        self.kl_divergence = KLDivergence()
        self.teacher_scorer = TeacherScorer(self.block_mask_builder)
        self.schedule_factory = ScheduleFactory()

    def build(
        self, model_runner: Any, online_train_settings: OnlineTrainSettings
    ) -> OnlineTrainSession | None:
        """Assemble and start a training session.

        Args:
            model_runner: The `GPUModelRunner` the capture hook is wrapping.
            online_train_settings: The operator's settings.

        Returns:
            The started session, or `None` when this engine cannot supply features.

        Raises:
            ValueError: If the engine is configured for DFlash but the draft and
                target are an incompatible pair, or `train_device` names a device this
                process cannot train on.
        """
        vllm_config = getattr(model_runner, "vllm_config", None)
        speculative_config = self.applicable(vllm_config, online_train_settings)
        if speculative_config is None:
            return None

        # Allocates the head, the borrowed state and the staging buffers outside any
        # `inference_mode` region the caller is already inside.
        with torch.inference_mode(False):
            resolved_config = self.config_resolver.resolve(
                vllm_config, online_train_settings
            )
            engine_provider = EngineStateProvider(
                self.target_locator,
                model_runner.get_model(),
                model_runner.device,
                vllm_config.model_config.dtype,
            )
            state_provider = self.build_provider(resolved_config, engine_provider)
            online_train_manager = self.build_manager(
                model_runner,
                vllm_config,
                resolved_config,
                state_provider,
                Path(speculative_config.model),
            )
            self.log_summary(resolved_config, state_provider.device)
            return self.start_session(online_train_manager, resolved_config)

    @staticmethod
    def applicable(
        vllm_config: Any, online_train_settings: OnlineTrainSettings
    ) -> Any | None:
        """Decide whether this engine can supply the features the head trains on.

        Args:
            vllm_config: The engine config, or `None`.
            online_train_settings: The operator's settings.

        Returns:
            The speculative config, or `None` after logging the reason.
        """
        if vllm_config is None:
            logger.info("Online training inactive: runner exposes no vllm_config")
            return None
        if not online_train_settings.enabled:
            logger.info("Online training inactive: settings set enabled=false")
            return None

        speculative_config = getattr(vllm_config, "speculative_config", None)
        if speculative_config is None or not speculative_config.use_dflash():
            method = (
                None if speculative_config is None else speculative_config.method
            )
            logger.info(
                "Online training inactive: speculative method is %r, not 'dflash'; "
                "the features the head trains on are a by-product of the DFlash "
                "serving path",
                method,
            )
            return None
        return speculative_config

    def build_provider(
        self, resolved_config: ResolvedConfig, engine_provider: StateProvider
    ) -> StateProvider:
        """Choose where the borrowed target state is served from.

        Args:
            resolved_config: Supplies the requested training device and the shared
                precision.
            engine_provider: The provider holding the target handle, on the engine's
                device.

        Returns:
            `engine_provider` itself when the head trains on the serving device,
            otherwise a mirror of it on the training device, verified against it.

        Raises:
            ValueError: If `train_device` names a device this process cannot train on,
                or the mirrored projection does not match the engine's own.
        """
        train_device = self.device_placement.resolve(
            resolved_config.online_train_settings.placement.train_device,
            engine_provider.device,
        )
        if train_device == engine_provider.device:
            logger.debug(
                "training on the engine's own device %s; no mirror", train_device
            )
            return engine_provider

        mirrored_provider = MirroredStateProvider(
            engine_provider,
            train_device,
            self.shared_dtype(resolved_config, engine_provider),
        )
        deviation = mirrored_provider.verify_parity()
        logger.info(
            "Online training on %s, off the engine's %s: the teacher projection is "
            "mirrored there at %s and matches the engine's own to %.2g of its largest "
            "logit",
            train_device,
            engine_provider.device,
            mirrored_provider.projection_dtype,
            deviation,
        )
        return mirrored_provider

    def build_manager(
        self,
        model_runner: Any,
        vllm_config: Any,
        resolved_config: ResolvedConfig,
        state_provider: StateProvider,
        source_dir: Path,
    ) -> OnlineTrainManager:
        """Build the handle that owns the capture, the pool, the head and the loop.

        Args:
            model_runner: The `GPUModelRunner`, for the live drafter and the engine's
                device.
            vllm_config: The engine config, for the draft model's shape.
            resolved_config: Settings paired with the resolved shapes.
            state_provider: Owns the target handle everything else borrows through.
            source_dir: The served head's directory.

        Returns:
            The manager, with its publish target already resolved.
        """
        draft_head = self.build_head(vllm_config, resolved_config, state_provider)
        rollout_buffer = self.build_buffer(resolved_config)
        online_train_manager = OnlineTrainManager(
            resolved_config=resolved_config,
            state_provider=state_provider,
            rollout_buffer=rollout_buffer,
            # The engine's device, not the provider's: the gather runs where the
            # activations are.
            rollout_capture=self.build_capture(
                resolved_config, rollout_buffer, model_runner.device
            ),
            draft_head=draft_head,
            online_trainer=self.build_trainer(
                resolved_config, draft_head, state_provider
            ),
            checkpoint_exporter=CheckpointExporter(
                self.checkpoint_writer, self.head_weights, source_dir
            ),
            weight_publisher=self.weight_publisher,
        )
        online_train_manager.set_publish_target(
            self.drafter_locator.find(model_runner)
        )
        return online_train_manager

    def build_head(
        self,
        vllm_config: Any,
        resolved_config: ResolvedConfig,
        state_provider: StateProvider,
    ) -> TrainableDFlashHead:
        """Build the head this engine's drafter is served with.

        Args:
            vllm_config: The engine config, for the draft model's shape.
            resolved_config: Settings paired with the resolved shapes.
            state_provider: Supplies the embedding table and the LM head.

        Returns:
            The head, on the provider's device, trained tensors at fp32 and borrowed
            tensors at the shared dtype.
        """
        head_arch = self.arch_factory.create(
            vllm_config, resolved_config.engine_shapes
        )
        shared_dtype = self.shared_dtype(resolved_config, state_provider)
        draft_head = self.head_factory.create(head_arch).to(dtype=torch.float32)
        self.head_weights.cast_shared(draft_head, shared_dtype)
        draft_head = draft_head.to(device=state_provider.device)
        self.head_weights.load_embedding(
            draft_head, state_provider.embedding_weight(dtype=shared_dtype)
        )
        self.head_weights.load_lm_head(
            draft_head, state_provider.lm_head_weight(dtype=shared_dtype)
        )
        return draft_head

    @staticmethod
    def shared_dtype(
        resolved_config: ResolvedConfig, state_provider: StateProvider
    ) -> torch.dtype:
        """The precision the borrowed embedding table and LM head are held at.

        Args:
            resolved_config: Supplies `shared_dtype`.
            state_provider: Supplies the serving dtype.

        Returns:
            fp32 when the settings ask for it, otherwise the serving dtype.
        """
        capture_settings = resolved_config.online_train_settings.capture
        if capture_settings.shared_dtype == "float32":
            return torch.float32
        return state_provider.serving_dtype

    @staticmethod
    def build_buffer(resolved_config: ResolvedConfig) -> RolloutBuffer:
        """Build the rollout pool and its draw strategy.

        Args:
            resolved_config: Supplies the buffer settings, the shapes and the seed.

        Returns:
            The pool.
        """
        online_train_settings = resolved_config.online_train_settings
        pool_sampler = PoolSampler(
            online_train_settings.buffer,
            random.Random(online_train_settings.bookkeeping.seed),
        )
        return RolloutBuffer(
            online_train_settings.buffer,
            resolved_config.engine_shapes,
            pool_sampler,
            BufferStats(),
        )

    def build_capture(
        self,
        resolved_config: ResolvedConfig,
        rollout_buffer: RolloutBuffer,
        device: torch.device,
    ) -> RolloutCapture:
        """Build the capture and the staging it writes through.

        Args:
            resolved_config: Supplies the capture settings, the shapes and the seed.
            rollout_buffer: Receives the drained chunks.
            device: Where the gather runs.

        Returns:
            The capture.
        """
        online_train_settings = resolved_config.online_train_settings
        # A chunk larger than `max_rollout_tokens` can only belong to a rollout that
        # will be dropped for length, so staging never needs to be bigger.
        feature_stager = FeatureStager(
            resolved_config.engine_shapes,
            online_train_settings.buffer.max_rollout_tokens,
            device,
        )
        request_sampler = RequestSampler(
            online_train_settings.capture,
            online_train_settings.bookkeeping.seed,
        )
        return RolloutCapture(
            online_train_settings.capture,
            rollout_buffer,
            feature_stager,
            request_sampler,
            self.valid_positions,
        )

    def build_trainer(
        self,
        resolved_config: ResolvedConfig,
        draft_head: TrainableDFlashHead,
        state_provider: StateProvider,
    ) -> OnlineTrainer:
        """Build the optimizer loop for one run.

        Args:
            resolved_config: Settings paired with the resolved shapes.
            draft_head: The head being trained.
            state_provider: Supplies the teacher distribution.

        Returns:
            The trainer.
        """
        online_train_settings = resolved_config.online_train_settings
        optimizer = torch.optim.AdamW(
            draft_head.trainable_parameters(),
            lr=online_train_settings.optim.learning_rate,
            weight_decay=online_train_settings.optim.weight_decay,
        )
        return OnlineTrainer(
            online_train_settings.optim,
            draft_head,
            self.head_weights,
            self.build_objective(resolved_config, draft_head, state_provider),
            self.build_collator(resolved_config, state_provider),
            self.schedule_factory,
            optimizer,
            online_train_settings.anchors.sequences_per_step,
        )

    def build_objective(
        self,
        resolved_config: ResolvedConfig,
        draft_head: TrainableDFlashHead,
        state_provider: StateProvider,
    ) -> SFTObjective:
        """Build the loss for one run.

        Args:
            resolved_config: Supplies the objective settings and the decay constant.
            draft_head: Supplies the student projection.
            state_provider: Supplies the teacher projection.

        Returns:
            The objective.
        """
        return SFTObjective(
            resolved_config.online_train_settings.objective,
            self.kl_divergence,
            self.teacher_scorer,
            PositionDecay(resolved_config.position_decay_gamma),
            draft_head.project_to_vocab,
            state_provider.teacher_logits,
        )

    def build_collator(
        self, resolved_config: ResolvedConfig, state_provider: StateProvider
    ) -> RolloutCollator:
        """Build the collator for one run.

        Args:
            resolved_config: Supplies the anchor settings, the shapes and the seed.
            state_provider: Supplies the training device.

        Returns:
            The collator.
        """
        online_train_settings = resolved_config.online_train_settings
        return RolloutCollator(
            self.block_builder,
            AnchorSampler(
                online_train_settings.anchors,
                random.Random(online_train_settings.bookkeeping.seed),
            ),
            DeviceTransfer(state_provider.device),
            resolved_config.engine_shapes,
        )

    @staticmethod
    def start_session(
        online_train_manager: OnlineTrainManager, resolved_config: ResolvedConfig
    ) -> OnlineTrainSession:
        """Wire the gate, the thread and the sink around a manager, and start them.

        Args:
            online_train_manager: The assembled training subsystem.
            resolved_config: Supplies the gate and bookkeeping settings.

        Returns:
            The started session.
        """
        online_train_settings = resolved_config.online_train_settings
        metrics_sink = JsonlMetricsSink(
            online_train_settings.bookkeeping.metrics_path
        )
        idle_gate = IdleGate(online_train_settings.gate)
        trainer_thread = TrainerThread(
            idle_gate,
            online_train_manager,
            online_train_settings.gate,
            online_train_settings.bookkeeping,
            metrics_sink,
        )
        online_train_session = OnlineTrainSession(
            online_train_manager, idle_gate, trainer_thread, metrics_sink
        )
        online_train_session.start()
        return online_train_session

    @staticmethod
    def log_summary(
        resolved_config: ResolvedConfig, train_device: torch.device
    ) -> None:
        """Log the shapes and the memory the run will occupy.

        Args:
            resolved_config: Settings paired with the resolved shapes.
            train_device: Where the head trains.
        """
        engine_shapes = resolved_config.engine_shapes
        logger.info(
            "Online training enabled on %s: %d feature layers %s, block size %d, "
            "%.1f KiB/token, buffer %d tokens (~%.1f GiB host)",
            train_device,
            engine_shapes.num_features,
            list(engine_shapes.aux_layer_ids),
            engine_shapes.block_size,
            engine_shapes.bytes_per_token / 1024,
            resolved_config.online_train_settings.buffer.buffer_capacity_tokens,
            resolved_config.buffer_capacity_bytes / 2**30,
        )
