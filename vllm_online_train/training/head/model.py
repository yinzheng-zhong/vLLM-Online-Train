from torch import nn

from vllm_online_train.training.head.arch import DFlashHeadArch
from vllm_online_train.training.head.layers.decoder import DFlashLayer
from vllm_online_train.training.head.layers.norm import RMSNorm


class DFlashModel(nn.Module):
    def __init__(self, arch: DFlashHeadArch) -> None:
        """Mirrors `DFlashQwen3Model`, so `state_dict()` keys line up under `model.`.

        Args:
            arch: Supplies the layer count and every width.
        """
        super().__init__()
        self.arch = arch
        self.embed_tokens = nn.Embedding(arch.vocab_size, arch.hidden_size)
        self.layers = nn.ModuleList(
            [DFlashLayer(arch) for _ in range(arch.num_layers)]
        )
        self.fc = nn.Linear(arch.feature_width, arch.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(arch.hidden_size, arch.rms_norm_eps)
        self.norm = RMSNorm(arch.hidden_size, arch.rms_norm_eps)
