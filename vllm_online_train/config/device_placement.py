import torch

from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class DevicePlacement:
    """Turns `placement.train_device` into the device the head trains on."""

    def resolve(
        self, requested_device: str | None, engine_device: torch.device
    ) -> torch.device:
        """Name the device the head, its optimizer and its batches occupy.

        Args:
            requested_device: The `train_device` setting, e.g. `"cuda:1"`. `None`
                selects the engine's own device.
            engine_device: The device the engine serves on.

        Returns:
            The training device. A requested type carrying no index resolves to
            `engine_device` when the types match, so only an explicit index moves the
            training off the serving device.

        Raises:
            ValueError: If the string names no device, or names a CUDA index this
                process cannot see.
        """
        if requested_device is None:
            logger.debug(
                "No train_device set; training on engine device %s", engine_device
            )
            return engine_device

        train_device = self._parse(requested_device)
        if train_device.index is None:
            resolved_device = (
                engine_device
                if train_device.type == engine_device.type
                else train_device
            )
            logger.debug(
                "train_device=%r carries no index; resolved to %s",
                requested_device,
                resolved_device,
            )
            return resolved_device
        self._check_visible(train_device)
        logger.debug(
            "train_device=%r resolved to explicit device %s",
            requested_device,
            train_device,
        )
        return train_device

    @staticmethod
    def _parse(requested_device: str) -> torch.device:
        """Read one device string.

        Args:
            requested_device: The `train_device` setting.

        Returns:
            The device it names.

        Raises:
            ValueError: If it names no device.
        """
        try:
            return torch.device(requested_device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                f"train_device={requested_device!r} does not name a device: {exc}"
            ) from exc

    @staticmethod
    def _check_visible(train_device: torch.device) -> None:
        """Check that a CUDA index is within this process's device count.

        Args:
            train_device: An indexed device.

        Raises:
            ValueError: If the index is out of range.
        """
        if train_device.type != "cuda":
            return
        visible_device_count = torch.cuda.device_count()
        if train_device.index >= visible_device_count:
            raise ValueError(
                f"train_device={str(train_device)!r} but this process can see "
                f"{visible_device_count} CUDA device(s), numbered from 0. "
                "CUDA_VISIBLE_DEVICES narrows and renumbers what the plugin sees, and "
                "vLLM sets it per process for data-parallel deployments."
            )
