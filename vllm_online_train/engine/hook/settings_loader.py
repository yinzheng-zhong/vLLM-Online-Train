import json
import os

from vllm_online_train.config.settings import OnlineTrainSettings
from vllm_online_train.config.settings_factory import SettingsFactory
from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class SettingsLoader:
    CONFIG_ENV = "ONLINE_TRAIN_CONFIG"
    """Path to a JSON file of flat settings fields, or inline JSON."""

    def __init__(self, settings_factory: SettingsFactory) -> None:
        """Reads `OnlineTrainSettings` from the environment.

        Args:
            settings_factory: Routes the flat mapping into sections.
        """
        self.settings_factory = settings_factory

    def load(
        self, config_source: str | None = None
    ) -> OnlineTrainSettings | None:
        """Read and parse the configured settings.

        Args:
            config_source: Overrides the environment variable. A path or inline
                JSON.

        Returns:
            The settings, or `None` when nothing is configured, which is what keeps
            an installed-but-unconfigured plugin inert.

        Raises:
            ValueError: If the value names a file that does not exist, or a field is
                not accepted.
            TypeError: If the JSON does not hold an object.
            json.JSONDecodeError: If the JSON does not parse.
        """
        raw_source = (
            config_source
            if config_source is not None
            else os.environ.get(self.CONFIG_ENV)
        )
        if not raw_source:
            return None

        field_values = self._parse(raw_source)
        # JSON has no comments and unknown fields are rejected, so underscore-prefixed
        # keys are reserved for annotation.
        field_values = {
            field_name: value
            for field_name, value in field_values.items()
            if not field_name.startswith("_")
        }
        # Pointing the variable at a config is itself the opt-in.
        field_values.setdefault("enabled", True)
        logger.debug(
            "Parsed %d settings field(s), enabled=%s",
            len(field_values),
            field_values["enabled"],
        )
        return self.settings_factory.create(field_values)

    def _parse(self, raw_source: str) -> dict:
        """Read the raw value as inline JSON or as a path to a JSON file.

        Args:
            raw_source: The environment value.

        Returns:
            The decoded mapping.

        Raises:
            ValueError: If the value is neither JSON nor an existing file.
            TypeError: If the JSON is not an object.
        """
        text = raw_source
        if not raw_source.lstrip().startswith("{"):
            config_path = os.path.expanduser(raw_source)
            if not os.path.isfile(config_path):
                raise ValueError(
                    f"{self.CONFIG_ENV}={raw_source!r} is not a file and is not JSON"
                )
            with open(config_path) as handle:
                text = handle.read()
            logger.debug("Loading settings from file %s", config_path)
        else:
            logger.debug("Loading settings from inline JSON")

        field_values = json.loads(text)
        if not isinstance(field_values, dict):
            raise TypeError(
                f"{self.CONFIG_ENV} must hold a JSON object, got {type(field_values)}"
            )
        return field_values
