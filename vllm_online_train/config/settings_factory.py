from dataclasses import fields
from typing import Any

from vllm_online_train.config.settings import (
    AnchorSettings,
    BookkeepingSettings,
    BufferSettings,
    CaptureSettings,
    GateSettings,
    HeadInitSettings,
    ObjectiveSettings,
    OnlineTrainSettings,
    OptimSettings,
    PlacementSettings,
    PublishSettings,
)
from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class SettingsFactory:
    """Builds `OnlineTrainSettings` from the flat mapping an operator writes."""

    SECTIONS: dict[str, type] = {
        "head_init": HeadInitSettings,
        "objective": ObjectiveSettings,
        "optim": OptimSettings,
        "anchors": AnchorSettings,
        "buffer": BufferSettings,
        "capture": CaptureSettings,
        "gate": GateSettings,
        "placement": PlacementSettings,
        "publish": PublishSettings,
        "bookkeeping": BookkeepingSettings,
    }
    """Attribute name on `OnlineTrainSettings` to the section class behind it."""

    TOP_LEVEL: frozenset[str] = frozenset({"enabled"})
    """Flat keys that belong to the aggregate rather than to any section."""

    def field_owner(self, field_name: str) -> str | None:
        """Name the section a flat key belongs to.

        Args:
            field_name: A flat key, e.g. `"idle_ms"`.

        Returns:
            The section attribute name, `"enabled"` for the master switch, or `None`
            when no section declares the key.
        """
        if field_name in self.TOP_LEVEL:
            return field_name
        for section_name, section_cls in self.SECTIONS.items():
            if any(field.name == field_name for field in fields(section_cls)):
                return section_name
        return None

    def known_fields(self) -> frozenset[str]:
        """Every flat key any section accepts, plus the top-level keys."""
        field_names = set(self.TOP_LEVEL)
        for section_cls in self.SECTIONS.values():
            field_names.update(field.name for field in fields(section_cls))
        return frozenset(field_names)

    def create(self, field_values: dict[str, Any]) -> OnlineTrainSettings:
        """Route flat keys into their sections and build the aggregate.

        Args:
            field_values: Flat field name to value, as read from
                `ONLINE_TRAIN_CONFIG`.

        Returns:
            The frozen sectioned settings.

        Raises:
            ValueError: If a key belongs to no section. The message names the closest
                accepted keys.
        """
        routed_values: dict[str, dict[str, Any]] = {
            section_name: {} for section_name in self.SECTIONS
        }
        top_level_values: dict[str, Any] = {}
        unknown_names: list[str] = []

        for field_name, value in field_values.items():
            owner_name = self.field_owner(field_name)
            if owner_name is None:
                unknown_names.append(field_name)
            elif owner_name in self.TOP_LEVEL:
                top_level_values[owner_name] = value
            else:
                routed_values[owner_name][field_name] = value

        if unknown_names:
            accepted_names = ", ".join(sorted(self.known_fields()))
            raise ValueError(
                f"unknown online-train settings: {', '.join(sorted(unknown_names))}. "
                f"Accepted fields are: {accepted_names}"
            )

        sections = {
            section_name: section_cls(**routed_values[section_name])
            for section_name, section_cls in self.SECTIONS.items()
        }
        logger.debug(
            "Routed %d online-train keys into %d sections (%d at top level)",
            sum(len(values) for values in routed_values.values()),
            len(sections),
            len(top_level_values),
        )
        return OnlineTrainSettings(**top_level_values, **sections)
