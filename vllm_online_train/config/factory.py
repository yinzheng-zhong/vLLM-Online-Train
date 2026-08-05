from dataclasses import fields
from typing import Any

from vllm_online_train.config.settings import (
    AnchorSettings,
    BookkeepingSettings,
    BufferSettings,
    CaptureSettings,
    GateSettings,
    ObjectiveSettings,
    OnlineTrainSettings,
    OptimSettings,
    PlacementSettings,
    PublishSettings,
)


class SettingsFactory:
    """Builds `OnlineTrainSettings` from the flat mapping an operator writes."""

    SECTIONS: dict[str, type] = {
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

    def field_owner(self, name: str) -> str | None:
        """Name the section a flat key belongs to.

        Args:
            name: A flat key, e.g. `"idle_ms"`.

        Returns:
            The section attribute name, `"enabled"` for the master switch, or `None`
            when no section declares the key.
        """
        if name in self.TOP_LEVEL:
            return name
        for section, cls in self.SECTIONS.items():
            if any(f.name == name for f in fields(cls)):
                return section
        return None

    def known_fields(self) -> frozenset[str]:
        """Every flat key any section accepts, plus the top-level keys."""
        names = set(self.TOP_LEVEL)
        for cls in self.SECTIONS.values():
            names.update(f.name for f in fields(cls))
        return frozenset(names)

    def create(self, values: dict[str, Any]) -> OnlineTrainSettings:
        """Route flat keys into their sections and build the aggregate.

        Args:
            values: Flat field name to value, as read from `ONLINE_TRAIN_CONFIG`.

        Returns:
            The frozen sectioned settings.

        Raises:
            ValueError: If a key belongs to no section. The message names the closest
                accepted keys.
        """
        routed: dict[str, dict[str, Any]] = {name: {} for name in self.SECTIONS}
        top: dict[str, Any] = {}
        unknown: list[str] = []

        for key, value in values.items():
            owner = self.field_owner(key)
            if owner is None:
                unknown.append(key)
            elif owner in self.TOP_LEVEL:
                top[owner] = value
            else:
                routed[owner][key] = value

        if unknown:
            accepted = ", ".join(sorted(self.known_fields()))
            raise ValueError(
                f"unknown online-train settings: {', '.join(sorted(unknown))}. "
                f"Accepted fields are: {accepted}"
            )

        sections = {
            name: cls(**routed[name]) for name, cls in self.SECTIONS.items()
        }
        return OnlineTrainSettings(**top, **sections)
