"""QSettings-backed persistence adapter for AppSettings.

This adapter provides persistent settings storage using Qt's QSettings,
allowing settings to persist across application restarts. This keeps
PySide6 dependencies out of the shared module.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings

from shared.app_settings import SettingsPersistence, AppSettings


class QSettingsPersistence(SettingsPersistence):
    """QSettings-backed persistence for GUI mode."""

    def __init__(
        self,
        organization: str = "SpikeHound",
        application: str = "SpikeHound",
    ) -> None:
        self._qsettings = QSettings(organization, application)
        self._field_names = [f.name for f in AppSettings.__dataclass_fields__.values()]

    def load(self) -> dict:
        """Load all settings from QSettings."""
        data = {}
        for name in self._field_names:
            val = self._qsettings.value(name)
            if val is not None:
                data[name] = val
        return data

    def save(self, data: dict) -> None:
        """Save settings to QSettings."""
        for key, val in data.items():
            if val is None:
                self._qsettings.remove(key)
            else:
                self._qsettings.setValue(key, val)


def create_gui_settings_store() -> "AppSettingsStore":
    """Factory function to create a settings store with QSettings persistence."""
    from shared.app_settings import AppSettingsStore
    return AppSettingsStore(persistence=QSettingsPersistence())


__all__ = ["QSettingsPersistence", "create_gui_settings_store"]
