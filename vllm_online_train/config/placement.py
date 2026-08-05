import torch


class DevicePlacement:
    """Turns `placement.train_device` into the device the head trains on."""

    def resolve(
        self, requested: str | None, engine_device: torch.device
    ) -> torch.device:
        """Name the device the head, its optimizer and its batches occupy.

        Args:
            requested: The `train_device` setting, e.g. `"cuda:1"`. `None` selects the
                engine's own device.
            engine_device: The device the engine serves on.

        Returns:
            The training device. A requested type carrying no index resolves to
            `engine_device` when the types match, so only an explicit index moves the
            training off the serving device.

        Raises:
            ValueError: If the string names no device, or names a CUDA index this
                process cannot see.
        """
        if requested is None:
            return engine_device

        device = self._parse(requested)
        if device.index is None:
            return engine_device if device.type == engine_device.type else device
        self._check_visible(device)
        return device

    @staticmethod
    def _parse(requested: str) -> torch.device:
        """Read one device string.

        Args:
            requested: The `train_device` setting.

        Returns:
            The device it names.

        Raises:
            ValueError: If it names no device.
        """
        try:
            return torch.device(requested)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                f"train_device={requested!r} does not name a device: {exc}"
            ) from exc

    @staticmethod
    def _check_visible(device: torch.device) -> None:
        """Check that a CUDA index is within this process's device count.

        Args:
            device: An indexed device.

        Raises:
            ValueError: If the index is out of range.
        """
        if device.type != "cuda":
            return
        count = torch.cuda.device_count()
        if device.index >= count:
            raise ValueError(
                f"train_device={str(device)!r} but this process can see {count} CUDA "
                "device(s), numbered from 0. CUDA_VISIBLE_DEVICES narrows and "
                "renumbers what the plugin sees, and vLLM sets it per process for "
                "data-parallel deployments."
            )
