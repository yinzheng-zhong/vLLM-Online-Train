from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_online_train.config.settings import OnlineTrainSettings
from vllm_online_train.config.shapes import EngineShapes, ResolvedConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig

logger = init_logger(__name__)


class ConfigResolver:
    """Derives `ResolvedConfig` from a live `VllmConfig`, rejecting bad pairings."""

    FEATURE_DTYPES: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    def resolve(
        self,
        vllm_config: "VllmConfig",
        settings: OnlineTrainSettings | None = None,
    ) -> ResolvedConfig:
        """Resolve shapes and validate the draft/target pair.

        Args:
            vllm_config: The engine config to resolve shapes from.
            settings: Operator settings. Defaults to
                `vllm_config.online_train_config`, then to section defaults.

        Returns:
            The settings paired with the resolved shapes.

        Raises:
            ValueError: If the engine cannot supply the features the head trains on,
                or the draft and target are an incompatible pair.
        """
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
        )

        if settings is None:
            settings = getattr(vllm_config, "online_train_config", None)
            settings = settings or OnlineTrainSettings()

        spec_config = self._require_dflash(vllm_config)
        aux_layer_ids = get_eagle3_aux_layers_from_config(spec_config)
        if not aux_layer_ids:
            raise ValueError(
                "Online training could not resolve the target feature layers. The "
                "draft config must set one of dflash_config.target_layer_ids, "
                "dflash_config.layer_ids or eagle_aux_hidden_state_layer_ids."
            )

        dflash_config = self._dflash_config(spec_config)
        if not dflash_config.get("use_aux_hidden_state", True):
            raise ValueError(
                "Online training requires dflash_config.use_aux_hidden_state; without "
                "it the drafter consumes only the final hidden state and no per-layer "
                "features are produced."
            )

        hidden_size = vllm_config.model_config.get_hidden_size()
        self._check_feature_width(vllm_config, len(aux_layer_ids), hidden_size)
        block_size = self._resolve_block_size(spec_config, settings)

        shapes = EngineShapes(
            aux_layer_ids=tuple(aux_layer_ids),
            hidden_size=hidden_size,
            vocab_size=vllm_config.model_config.get_vocab_size(),
            block_size=block_size,
            mask_token_id=dflash_config.get("mask_token_id") or 0,
            feature_dtype=self._feature_dtype(vllm_config, settings),
        )
        gamma = settings.objective.position_decay_gamma
        if gamma is None:
            gamma = settings.objective.default_gamma(block_size)

        logger.debug(
            "Resolved online-train shapes: block_size=%d hidden_size=%d "
            "num_features=%d feature_dtype=%s position_decay_gamma=%.3f",
            shapes.block_size,
            shapes.hidden_size,
            shapes.num_features,
            shapes.feature_dtype,
            gamma,
        )
        return ResolvedConfig(
            settings=settings, shapes=shapes, position_decay_gamma=gamma
        )

    @staticmethod
    def _require_dflash(vllm_config: "VllmConfig") -> "SpeculativeConfig":
        """Return the speculative config, or reject a non-DFlash engine.

        Args:
            vllm_config: The engine config.

        Returns:
            The engine's speculative config.

        Raises:
            ValueError: If the engine is not serving `method: "dflash"`.
        """
        spec_config = vllm_config.speculative_config
        if spec_config is None or not spec_config.use_dflash():
            raise ValueError(
                "Online training requires speculative_config.method == 'dflash'; got "
                f"{None if spec_config is None else spec_config.method!r}. The "
                "features the head trains on are a by-product of the DFlash serving "
                "path, so no other method can supply them."
            )
        return spec_config

    @staticmethod
    def _dflash_config(spec_config: "SpeculativeConfig") -> dict:
        """The draft checkpoint's `dflash_config` block, or an empty mapping."""
        hf_config = spec_config.draft_model_config.hf_config
        return getattr(hf_config, "dflash_config", None) or {}

    @staticmethod
    def _check_feature_width(
        vllm_config: "VllmConfig", num_features: int, hidden_size: int
    ) -> None:
        """Compare the resolved feature width against the drafter's `fc`.

        Raises:
            ValueError: If they disagree.
        """
        from vllm.model_executor.models.qwen3_dflash import _get_dflash_fc_input_size

        feature_width = num_features * hidden_size
        expected_width = _get_dflash_fc_input_size(vllm_config)
        if feature_width != expected_width:
            raise ValueError(
                f"Resolved {num_features} feature layers x {hidden_size} hidden "
                f"= {feature_width}, but the drafter's fc expects {expected_width}. "
                "The draft and target are an incompatible pair, or target_hidden_size "
                "disagrees with the target model."
            )

    @staticmethod
    def _resolve_block_size(
        spec_config: "SpeculativeConfig", settings: OnlineTrainSettings
    ) -> int:
        """`num_speculative_tokens + 1`, checked against the rollout length bound.

        Raises:
            ValueError: If a block would have no predicted slot, or no rollout could
                hold one.
        """
        block_size = spec_config.num_speculative_tokens + 1
        if block_size < 2:
            raise ValueError(
                "Online training needs num_speculative_tokens >= 1 so a block has at "
                "least one predicted slot beside the anchor; got "
                f"{spec_config.num_speculative_tokens}."
            )
        max_rollout_tokens = settings.buffer.max_rollout_tokens
        if max_rollout_tokens <= block_size:
            raise ValueError(
                f"max_rollout_tokens ({max_rollout_tokens}) must exceed the "
                f"block size ({block_size}) or no rollout can yield an anchor."
            )
        return block_size

    @classmethod
    def _feature_dtype(
        cls, vllm_config: "VllmConfig", settings: OnlineTrainSettings
    ) -> torch.dtype:
        """The configured feature dtype, falling back to the model's."""
        name = settings.capture.feature_dtype
        if name is None:
            return vllm_config.model_config.dtype
        return cls.FEATURE_DTYPES[name]
