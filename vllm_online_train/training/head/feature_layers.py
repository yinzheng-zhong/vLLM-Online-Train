from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class TargetLayerPlanner:
    """Chooses which target layers a head reads features from."""

    def plan(self, num_target_layers: int, num_features: int) -> list[int]:
        """Spread feature layers evenly over the target, in DFlash numbering.

        36 target layers and 5 features give `[1, 9, 17, 25, 33]`. The top of the
        range stops three layers short of the end, relaxing that cutoff only when a
        shallow target leaves no room for it.

        These are DFlash ids; vLLM adds 1 to reach its own aux-layer numbering.

        Args:
            num_target_layers: Layers in the target model.
            num_features: Feature layers to place.

        Returns:
            `num_features` ascending DFlash layer ids.

        Raises:
            ValueError: If the target is too shallow to supply that many distinct
                feature layers.
        """
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")
        if num_target_layers < 2:
            raise ValueError(
                f"a {num_target_layers}-layer target cannot supply feature layers"
            )
        if num_features == 1:
            return [1]

        highest_id = self._highest_id(num_target_layers, num_features)
        stride = max((highest_id - 1) // (num_features - 1), 1)
        layer_ids = [1 + index * stride for index in range(num_features)]
        logger.debug(
            "Placed %d feature layers with stride %d: %s",
            num_features,
            stride,
            layer_ids,
        )
        return layer_ids

    @staticmethod
    def _highest_id(num_target_layers: int, num_features: int) -> int:
        """The largest usable DFlash id for a target of this depth.

        Raises:
            ValueError: If even the relaxed cutoff cannot fit `num_features` ids.
        """
        highest_id = min(max(num_target_layers - 3, 1), num_target_layers - 1)
        if num_features > highest_id:
            highest_id = num_target_layers - 1
        if num_features > highest_id:
            raise ValueError(
                f"cannot place {num_features} distinct feature layers in a "
                f"{num_target_layers}-layer target: ids must lie in [1, {highest_id}] "
                "so that vLLM's +1 into aux numbering stays inside the model. Reduce "
                "--features."
            )
        return highest_id
