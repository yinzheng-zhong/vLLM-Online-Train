import torch

from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class DeviceTransfer:
    def __init__(self, device: torch.device) -> None:
        """Moves collated host tensors onto the training device.

        Args:
            device: Destination for every tensor passed to `send`.
        """
        self.device = device
        logger.debug(
            "device transfer targets %s (pin_memory=%s)",
            device,
            device.type == "cuda",
        )

    def send(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy one tensor to the device.

        On CUDA the tensor is pinned first, so `non_blocking` overlaps the transfer
        with host work; from pageable memory the call would block regardless.

        Args:
            tensor: A host tensor.

        Returns:
            The tensor on `self.device`.
        """
        if self.device.type != "cuda":
            return tensor.to(self.device)
        return tensor.pin_memory().to(self.device, non_blocking=True)
