from collections.abc import Callable

from vllm_online_train.config.settings import OptimSettings
from vllm_online_train.logger import init_logger
from vllm_online_train.training.optim.cosine_warmup import CosineWithWarmup
from vllm_online_train.training.optim.warmup_only import WarmupOnly

logger = init_logger(__name__)


class ScheduleFactory:
    """Selects the learning-rate schedule the settings ask for."""

    OPEN_ENDED_WARMUP_STEPS = 100
    """Warmup length a `warmup_ratio` of 0.05 maps to when no horizon is set."""

    def create(self, optim_settings: OptimSettings) -> Callable[[int], float]:
        """Build the multiplier for one run.

        Args:
            optim_settings: Supplies `total_steps` and `warmup_ratio`.

        Returns:
            `CosineWithWarmup` when a horizon is set, else `WarmupOnly`.
        """
        if optim_settings.total_steps:
            logger.debug(
                "ScheduleFactory: selected CosineWithWarmup for total_steps=%d",
                optim_settings.total_steps,
            )
            return CosineWithWarmup(
                optim_settings.total_steps, optim_settings.warmup_ratio
            )
        logger.debug("ScheduleFactory: no total_steps set, selected WarmupOnly")
        return WarmupOnly(self.warmup_steps(optim_settings))

    @classmethod
    def warmup_steps(cls, optim_settings: OptimSettings) -> int:
        """Warmup length for an open-ended run.

        Args:
            optim_settings: Supplies `warmup_ratio`.

        Returns:
            `warmup_ratio` scaled so 0.05 gives `OPEN_ENDED_WARMUP_STEPS`.
        """
        return max(
            1,
            round(cls.OPEN_ENDED_WARMUP_STEPS * optim_settings.warmup_ratio / 0.05),
        )
