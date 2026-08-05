"""The test-side wiring site.

Built the way `assembler.py` builds it: one mask builder shared between the head and
the teacher gather, one naming rewriter shared between the writer and the publisher,
one weight helper shared between the factory and the publisher.

This lives beside `conftest.py` rather than inside it because pytest imports a
conftest without an `__init__.py` under two module names, which would give the tests
two copies of every instance and make an identity assertion meaningless.
"""

from vllm_online_train.capture.positions import ValidPositions
from vllm_online_train.checkpoint.config import DraftConfigBuilder
from vllm_online_train.checkpoint.layers import TargetLayerPlanner
from vllm_online_train.checkpoint.locator import DrafterLocator
from vllm_online_train.checkpoint.naming import WeightNameRewriter
from vllm_online_train.checkpoint.publisher import WeightPublisher
from vllm_online_train.checkpoint.writer import CheckpointWriter
from vllm_online_train.collate.blocks import BlockBuilder
from vllm_online_train.config.factory import SettingsFactory
from vllm_online_train.config.placement import DevicePlacement
from vllm_online_train.config.resolver import ConfigResolver
from vllm_online_train.head.arch import ArchFactory
from vllm_online_train.head.factory import HeadFactory
from vllm_online_train.head.masks import BlockMaskBuilder
from vllm_online_train.head.weights import HeadWeights
from vllm_online_train.train.loss.divergence import KLDivergence
from vllm_online_train.train.loss.teacher import TeacherScorer
from vllm_online_train.train.optim.schedule import ScheduleFactory

settings_factory = SettingsFactory()
config_resolver = ConfigResolver()
device_placement = DevicePlacement()
mask_builder = BlockMaskBuilder()
head_weights = HeadWeights()
head_factory = HeadFactory(mask_builder, head_weights)
arch_factory = ArchFactory()
weight_naming = WeightNameRewriter()
checkpoint_writer = CheckpointWriter(weight_naming)
weight_publisher = WeightPublisher(weight_naming, head_weights)
drafter_locator = DrafterLocator()
layer_planner = TargetLayerPlanner()
draft_config_builder = DraftConfigBuilder()
block_builder = BlockBuilder()
valid_positions = ValidPositions()
kl_divergence = KLDivergence()
teacher_scorer = TeacherScorer(mask_builder)
schedule_factory = ScheduleFactory()
