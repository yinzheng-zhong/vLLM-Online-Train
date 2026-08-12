from torch import nn

from vllm_online_train.training.head.arch import DFlashHeadArch
from vllm_online_train.training.head.layers.decoder import DFlashLayer
from vllm_online_train.training.head.layers.norm import RMSNorm


class DFlashModel(nn.Module):
    def __init__(self, head_arch: DFlashHeadArch) -> None:
        """Mirrors `DFlashQwen3Model`, so `state_dict()` keys line up under `model.`.

        Args:
            head_arch: Supplies the layer count and every width.
        """
        super().__init__()
        self.head_arch = head_arch
        self.embed_tokens = nn.Embedding(
            head_arch.vocab_size, head_arch.hidden_size
        )
        self.layers = nn.ModuleList(
            [DFlashLayer(head_arch) for _ in range(head_arch.num_layers)]
        )
        self.fc = nn.Linear(
            head_arch.feature_width, head_arch.hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(head_arch.hidden_size, head_arch.rms_norm_eps)
        self.norm = RMSNorm(head_arch.hidden_size, head_arch.rms_norm_eps)
