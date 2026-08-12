from collections.abc import Iterable

from vllm_online_train.logger import init_logger
from vllm_online_train.training.head.arch import DFlashHeadArch

logger = init_logger(__name__)


class DraftConfigBuilder:
    """Builds the `config.json` a DFlash draft directory needs."""

    ARCHITECTURE = "DFlashDraftModel"
    """What vLLM's `registry.py` maps to `qwen3_dflash:DFlashQwen3ForCausalLM`."""

    def build(
        self,
        head_arch: DFlashHeadArch,
        *,
        target_layer_ids: Iterable[int],
        num_target_layers: int,
        dtype: str = "bfloat16",
        max_position_embeddings: int = 40960,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> dict:
        """Assemble the config a restarted engine loads the head from.

        `model_type: "qwen3"` makes this load as a `Qwen3Config`; `architectures`
        selects the DFlash loader. `block_size` is recorded for the operator, since
        the served block comes from `num_speculative_tokens + 1`.

        Args:
            head_arch: Shape of the head.
            target_layer_ids: DFlash-numbered feature layers.
            num_target_layers: Layers in the target model.
            dtype: Recorded storage dtype.
            max_position_embeddings: Recorded position limit.
            bos_token_id: Written only when given.
            eos_token_id: Written only when given.

        Returns:
            The config mapping.

        Raises:
            ValueError: If the number of layer ids disagrees with
                `head_arch.num_features`.
        """
        target_layer_ids = list(target_layer_ids)
        if len(target_layer_ids) != head_arch.num_features:
            raise ValueError(
                f"{len(target_layer_ids)} target_layer_ids for a head expecting "
                f"{head_arch.num_features} features; fc would be built at the wrong "
                "width."
            )
        draft_config = {
            "architectures": [self.ARCHITECTURE],
            "model_type": "qwen3",
            "attention_bias": head_arch.attention_bias,
            "attention_dropout": 0.0,
            "block_size": head_arch.block_size,
            "dflash_config": {
                "mask_token_id": head_arch.mask_token_id,
                "target_layer_ids": target_layer_ids,
            },
            "dtype": dtype,
            "head_dim": head_arch.head_dim,
            "hidden_act": "silu",
            "hidden_size": head_arch.hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": head_arch.intermediate_size,
            "layer_types": ["full_attention"] * head_arch.num_layers,
            "max_position_embeddings": max_position_embeddings,
            "max_window_layers": head_arch.num_layers,
            "num_attention_heads": head_arch.num_heads,
            "num_hidden_layers": head_arch.num_layers,
            "num_key_value_heads": head_arch.num_kv_heads,
            "num_target_layers": num_target_layers,
            "rms_norm_eps": head_arch.rms_norm_eps,
            "rope_scaling": None,
            "rope_theta": head_arch.rope_theta,
            "sliding_window": None,
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": head_arch.vocab_size,
        }
        if bos_token_id is not None:
            draft_config["bos_token_id"] = bos_token_id
        if eos_token_id is not None:
            draft_config["eos_token_id"] = eos_token_id
        logger.debug(
            "Built draft config: block_size=%d, num_target_layers=%d, "
            "target_layer_ids=%s",
            head_arch.block_size,
            num_target_layers,
            target_layer_ids,
        )
        return draft_config
