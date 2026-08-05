import json
import os

from vllm.logger import init_logger

from vllm_online_train.config.factory import SettingsFactory
from vllm_online_train.config.settings import OnlineTrainSettings

logger = init_logger(__name__)


class SettingsLoader:
    CONFIG_ENV = "ONLINE_TRAIN_CONFIG"
    """Path to a JSON file of flat settings fields, or inline JSON."""

    def __init__(self, factory: SettingsFactory) -> None:
        """Reads `OnlineTrainSettings` from the environment.

        Args:
            factory: Routes the flat mapping into sections.
        """
        self.factory = factory

    def load(self, source: str | None = None) -> OnlineTrainSettings | None:
        """Read and parse the configured settings.

        Args:
            source: Overrides the environment variable. A path or inline JSON.

        Returns:
            The settings, or `None` when nothing is configured, which is what keeps
            an installed-but-unconfigured plugin inert.

        Raises:
            ValueError: If the value names a file that does not exist, or a field is
                not accepted.
            TypeError: If the JSON does not hold an object.
            json.JSONDecodeError: If the JSON does not parse.
        """
        raw = source if source is not None else os.environ.get(self.CONFIG_ENV)
        if not raw:
            return None

        fields = self._parse(raw)
        # JSON has no comments and unknown fields are rejected, so underscore-prefixed
        # keys are reserved for annotation.
        fields = {k: v for k, v in fields.items() if not k.startswith("_")}
        # Pointing the variable at a config is itself the opt-in.
        fields.setdefault("enabled", True)
        logger.debug(
            "Parsed %d settings field(s), enabled=%s", len(fields), fields["enabled"]
        )
        return self.factory.create(fields)

    def _parse(self, raw: str) -> dict:
        """Read the raw value as inline JSON or as a path to a JSON file.

        Args:
            raw: The environment value.

        Returns:
            The decoded mapping.

        Raises:
            ValueError: If the value is neither JSON nor an existing file.
            TypeError: If the JSON is not an object.
        """
        text = raw
        if not raw.lstrip().startswith("{"):
            path = os.path.expanduser(raw)
            if not os.path.isfile(path):
                raise ValueError(
                    f"{self.CONFIG_ENV}={raw!r} is not a file and is not JSON"
                )
            with open(path) as handle:
                text = handle.read()
            logger.debug("Loading settings from file %s", path)
        else:
            logger.debug("Loading settings from inline JSON")

        fields = json.loads(text)
        if not isinstance(fields, dict):
            raise TypeError(
                f"{self.CONFIG_ENV} must hold a JSON object, got {type(fields)}"
            )
        return fields
