"""The test-side wiring site.

Built the way `assembler.py` builds it: one mask builder shared between the head and
the teacher gather, one naming rewriter shared between the writer and the publisher,
one weight helper shared between the factory and the publisher.

This lives beside `conftest.py` rather than inside it because pytest imports a
conftest without an `__init__.py` under two module names, which would give the tests
two copies of every instance and make an identity assertion meaningless.
"""

from pathlib import Path

from vllm_online_train.config.device_placement import DevicePlacement
from vllm_online_train.config.resolver import ConfigResolver
from vllm_online_train.config.settings_factory import SettingsFactory
from vllm_online_train.engine.capture.valid_positions import ValidPositions
from vllm_online_train.training.checkpoint.draft_config import DraftConfigBuilder
from vllm_online_train.training.checkpoint.drafter_locator import DrafterLocator
from vllm_online_train.training.checkpoint.weight_names import WeightNameRewriter
from vllm_online_train.training.checkpoint.weight_publisher import WeightPublisher
from vllm_online_train.training.checkpoint.writer import CheckpointWriter
from vllm_online_train.training.collate.block_builder import BlockBuilder
from vllm_online_train.training.head.arch import ArchFactory
from vllm_online_train.training.head.block_masks import BlockMaskBuilder
from vllm_online_train.training.head.factory import HeadFactory
from vllm_online_train.training.head.feature_layers import TargetLayerPlanner
from vllm_online_train.training.head.weights import HeadWeights
from vllm_online_train.training.loss.kl_divergence import KLDivergence
from vllm_online_train.training.loss.teacher_scorer import TeacherScorer
from vllm_online_train.training.optim.schedule_factory import ScheduleFactory

REPO_ROOT = Path(__file__).resolve().parents[1]
"""The repository root, for tests that read a file shipped beside the package."""

settings_factory = SettingsFactory()
config_resolver = ConfigResolver()
device_placement = DevicePlacement()
block_mask_builder = BlockMaskBuilder()
head_weights = HeadWeights()
head_factory = HeadFactory(block_mask_builder, head_weights)
arch_factory = ArchFactory()
weight_name_rewriter = WeightNameRewriter()
checkpoint_writer = CheckpointWriter(weight_name_rewriter)
weight_publisher = WeightPublisher(weight_name_rewriter, head_weights)
drafter_locator = DrafterLocator()
target_layer_planner = TargetLayerPlanner()
draft_config_builder = DraftConfigBuilder()
block_builder = BlockBuilder()
valid_positions = ValidPositions()
kl_divergence = KLDivergence()
teacher_scorer = TeacherScorer(block_mask_builder)
schedule_factory = ScheduleFactory()
