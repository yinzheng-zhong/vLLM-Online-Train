import hashlib

from vllm.logger import init_logger

from vllm_online_train.config.settings import CaptureSettings

logger = init_logger(__name__)


class RequestSampler:
    def __init__(self, settings: CaptureSettings, seed: int) -> None:
        """Decides once per request whether its activations are captured.

        Args:
            settings: Supplies the capture sample rate.
            seed: Keys the hash, so the decision is reproducible across runs.
        """
        self.settings = settings
        self.seed = seed
        logger.debug(
            "RequestSampler configured: capture_sample_rate=%s, seed=%d",
            settings.capture_sample_rate,
            seed,
        )

    def accepts(self, req_id: str) -> bool:
        """Whether this request is captured.

        Args:
            req_id: The engine's request id.

        Returns:
            True at a sample rate of 1.0, else whether the request's keyed hash falls
            below the rate. The same id always gives the same answer, so a chunked
            prefill stays consistent with itself.
        """
        rate = self.settings.capture_sample_rate
        if rate >= 1.0:
            return True
        return self.stable_unit_hash(req_id, self.seed) < rate

    @staticmethod
    def stable_unit_hash(text: str, seed: int) -> float:
        """A deterministic value in [0, 1) for a string.

        Args:
            text: The string to hash.
            seed: Keys the hash.

        Returns:
            The keyed blake2b digest scaled into [0, 1).
        """
        digest = hashlib.blake2b(
            text.encode(), key=str(seed).encode()[:64], digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") / float(1 << 64)
