from dataclasses import dataclass
from typing import Any

from vllm_online_train.logger import init_logger
from vllm_online_train.training.head.arch import ArchFactory, DFlashHeadArch
from vllm_online_train.training.head.feature_layers import TargetLayerPlanner

logger = init_logger(__name__)


@dataclass(frozen=True)
class InitHeadRequest:
    """What the operator asked for when materialising a fresh draft checkpoint.

    Every width is `None` to mean "read it off the target". This is what the CLI
    parses argv into, so no `argparse.Namespace` reaches `head/`.
    """

    target: str
    layers: int
    features: int
    block_size: int

    target_layer_ids: list[int] | None = None
    mask_token_id: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    rope_theta: float | None = None


@dataclass(frozen=True)
class InitHeadPlan:
    """A fresh head's resolved shape, the layers it reads, and where they came
    from."""

    head_arch: DFlashHeadArch
    target_layer_ids: list[int]
    """DFlash-numbered feature layers. vLLM adds 1 to reach its aux numbering."""

    num_target_layers: int

    target_config: Any
    """The target's transformers config, for the metadata the draft config copies."""


class InitHeadPlanner:
    def __init__(
        self, target_layer_planner: TargetLayerPlanner, arch_factory: ArchFactory
    ) -> None:
        """Resolves a fresh head's shape from a target config plus overrides.

        Args:
            target_layer_planner: Places the feature layers when none are given.
            arch_factory: Supplies the RoPE base fallback.
        """
        self.target_layer_planner = target_layer_planner
        self.arch_factory = arch_factory

    def plan(self, init_head_request: InitHeadRequest) -> InitHeadPlan:
        """Read the target config and resolve every width.

        Args:
            init_head_request: The operator's choices.

        Returns:
            The plan, ready for `HeadFactory.create` and `DraftConfigBuilder.build`.

        Raises:
            ValueError: If the feature layers reach past the target's depth, or the
                target has no spare embedding row for a mask token.
        """
        target_config = self.load_target_config(init_head_request.target)

        hidden_size = init_head_request.hidden_size or int(target_config.hidden_size)
        num_heads = init_head_request.num_heads or int(
            target_config.num_attention_heads
        )
        head_dim = init_head_request.head_dim or int(
            getattr(target_config, "head_dim", None) or hidden_size // num_heads
        )
        vocab_size = int(target_config.vocab_size)
        num_target_layers = int(target_config.num_hidden_layers)

        target_layer_ids = (
            init_head_request.target_layer_ids
            or self.target_layer_planner.plan(
                num_target_layers, init_head_request.features
            )
        )
        self._check_reach(target_layer_ids, num_target_layers)

        mask_token_id = init_head_request.mask_token_id
        if mask_token_id is None:
            mask_token_id = self.resolve_mask_token_id(
                init_head_request.target, vocab_size
            )

        head_arch = DFlashHeadArch(
            num_layers=init_head_request.layers,
            hidden_size=hidden_size,
            intermediate_size=init_head_request.intermediate_size
            or int(target_config.intermediate_size),
            num_heads=num_heads,
            num_kv_heads=init_head_request.num_kv_heads
            or int(getattr(target_config, "num_key_value_heads", num_heads)),
            head_dim=head_dim,
            num_features=len(target_layer_ids),
            block_size=init_head_request.block_size,
            vocab_size=vocab_size,
            draft_vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            rms_norm_eps=float(getattr(target_config, "rms_norm_eps", 1e-6)),
            rope_theta=float(
                init_head_request.rope_theta
                or self.arch_factory.rope_theta(target_config)
            ),
            attention_bias=bool(getattr(target_config, "attention_bias", False)),
        )
        logger.debug(
            "Resolved head plan: hidden_size=%d, num_heads=%d, "
            "target_layer_ids=%s, mask_token_id=%d",
            hidden_size,
            num_heads,
            target_layer_ids,
            mask_token_id,
        )
        return InitHeadPlan(
            head_arch=head_arch,
            target_layer_ids=list(target_layer_ids),
            num_target_layers=num_target_layers,
            target_config=target_config,
        )

    @staticmethod
    def load_target_config(target: str) -> Any:
        """Read a target model's text config.

        Args:
            target: Model id or path.

        Returns:
            The config, unwrapped from `text_config` for multimodal targets.
        """
        from transformers import AutoConfig

        target_config = AutoConfig.from_pretrained(target)
        if hasattr(target_config, "text_config"):
            return target_config.text_config
        return target_config

    @staticmethod
    def resolve_mask_token_id(target: str, vocab_size: int) -> int:
        """The first embedding row past the tokenizer's real tokens.

        Args:
            target: Model id or path, for the tokenizer.
            vocab_size: The target's padded vocabulary size.

        Returns:
            `len(tokenizer)`, a genuinely unused embedding row.

        Raises:
            ValueError: If the vocabulary is not padded past the tokenizer.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(target)
        mask_token_id = len(tokenizer)
        if mask_token_id >= vocab_size:
            raise ValueError(
                f"{target} has no spare embedding row for the mask token: tokenizer "
                f"holds {mask_token_id} tokens and vocab_size is {vocab_size}. Pass "
                "--mask-token-id to reuse a token you know is never generated."
            )
        return mask_token_id

    @staticmethod
    def _check_reach(target_layer_ids: list[int], num_target_layers: int) -> None:
        """Reject feature layers that reach past the target's depth.

        Raises:
            ValueError: If any id plus vLLM's `+1` falls outside the model.
        """
        if max(target_layer_ids) + 1 > num_target_layers:
            raise ValueError(
                f"target_layer_ids {target_layer_ids} reach aux layer "
                f"{max(target_layer_ids) + 1}, past the target's {num_target_layers} "
                "layers. vLLM adds 1 to DFlash ids to get its own aux numbering."
            )
