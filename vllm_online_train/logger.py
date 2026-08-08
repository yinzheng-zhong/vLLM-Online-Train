import logging

from vllm.logger import init_logger as init_vllm_logger

LOGGER_ROOT = "vllm.online_train"
TAG = "[online-train]"


class PackageTag(logging.Filter):
    """Marks every record this package emits with `TAG`."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Prefix the record's message with the package tag.

        Args:
            record: The record about to be emitted.

        Returns:
            `True`, always: this filter drops nothing.
        """
        message = str(record.msg)
        if not message.startswith(TAG):
            record.msg = f"{TAG} {message}"
        return True


def init_logger(name: str) -> logging.Logger:
    """A tagged logger inside vLLM's own `vllm` logger tree.

    Args:
        name: The calling module's `__name__`.

    Returns:
        A logger named `vllm.online_train.<module path>`, carrying vLLM's handler,
        format and level, whose records are tagged with `TAG`.
    """
    suffix = name.partition(".")[2]
    logger = init_vllm_logger(f"{LOGGER_ROOT}.{suffix}" if suffix else LOGGER_ROOT)
    if not any(isinstance(existing, PackageTag) for existing in logger.filters):
        logger.addFilter(PackageTag())
    return logger
