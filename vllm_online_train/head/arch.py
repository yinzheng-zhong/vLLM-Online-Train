from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_online_train.config.shapes import EngineShapes

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class DFlashHeadArch:
    """Shape of the draft head, read off the draft checkpoint's config."""

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_features: int
    block_size: int
    vocab_size: int
    draft_vocab_size: int
    mask_token_id: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool

    @property
    def q_size(self) -> int:
        """Width of the fused query projection."""
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        """Width of one of the two fused key/value projections."""
        return self.num_kv_heads * self.head_dim

    @property
    def feature_width(self) -> int:
        """`fc.in_features`: features concatenated across layers."""
        return self.num_features * self.hidden_size

    @property
    def n_rep(self) -> int:
        """Query heads per key/value head."""
        return self.num_heads // self.num_kv_heads


class ArchFactory:
    """Reads a `DFlashHeadArch` off the draft checkpoint's config."""

    DEFAULT_ROPE_THETA = 1_000_000.0
    """Used when the draft config omits one. The serving side defaults to the same
    value via `set_default_rope_theta`, and the two must agree."""

    def create(
        self, vllm_config: "VllmConfig", shapes: EngineShapes
    ) -> DFlashHeadArch:
        """Build the arch of the head the engine is serving.

        Args:
            vllm_config: The engine config, for the draft model's `hf_config`.
            shapes: Supplies the feature count, block size and mask token.

        Returns:
            The head's shape.
        """
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        head_dim = int(getattr(hf_config, "head_dim", None) or hidden_size // num_heads)

        return DFlashHeadArch(
            num_layers=int(hf_config.num_hidden_layers),
            hidden_size=hidden_size,
            intermediate_size=int(hf_config.intermediate_size),
            num_heads=num_heads,
            num_kv_heads=int(getattr(hf_config, "num_key_value_heads", num_heads)),
            head_dim=head_dim,
            num_features=shapes.num_features,
            block_size=shapes.block_size,
            vocab_size=int(hf_config.vocab_size),
            draft_vocab_size=int(
                getattr(hf_config, "draft_vocab_size", None) or hf_config.vocab_size
            ),
            mask_token_id=shapes.mask_token_id,
            rms_norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-6)),
            rope_theta=self.rope_theta(hf_config),
            attention_bias=bool(getattr(hf_config, "attention_bias", False)),
        )

    @classmethod
    def rope_theta(cls, hf_config: Any) -> float:
        """The RoPE base a config declares, or the DFlash default.

        Args:
            hf_config: A transformers config with `rope_parameters` or `rope_theta`.

        Returns:
            The declared base, else `DEFAULT_ROPE_THETA`.
        """
        rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
        theta = rope_parameters.get("rope_theta") or getattr(
            hf_config, "rope_theta", None
        )
        return float(theta or cls.DEFAULT_ROPE_THETA)
