from __future__ import annotations

import logging
from dataclasses import dataclass, replace, fields, asdict
import threading
from typing import Callable, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppSettings:
    plot_refresh_hz: float = 40.0
    default_window_sec: float = 1.0
    listen_output_key: Optional[str] = None
    list_all_audio_devices: bool = False
    load_config_on_launch: bool = False
    launch_config_path: Optional[str] = None
    recording_use_float32: bool = False
    recording_auto_increment: bool = True


@runtime_checkable
class SettingsPersistence(Protocol):
    """Protocol for settings persistence backends."""
    def load(self) -> dict:
        """Load settings from persistent storage. Returns dict of setting values."""
        ...
    
    def save(self, data: dict) -> None:
        """Save settings to persistent storage."""
        ...


class InMemoryPersistence:
    """In-memory persistence for headless mode (no persistence across restarts)."""
    
    def __init__(self) -> None:
        self._data: dict = {}
    
    def load(self) -> dict:
        return dict(self._data)
    
    def save(self, data: dict) -> None:
        self._data = dict(data)


class AppSettingsStore:
    """Thread-safe settings store for application-wide preferences.
    
    By default uses in-memory persistence (headless mode). For GUI mode,
    inject a QSettingsPersistence from gui.qsettings_adapter.
    """

    def __init__(self, *, persistence: Optional[SettingsPersistence] = None) -> None:
        self._lock = threading.Lock()
        self._subscribers: Dict[int, Callable[[AppSettings], None]] = {}
        self._next_token = 0
        self._persistence = persistence or InMemoryPersistence()
        self._settings = self._load_settings()

    def _load_settings(self) -> AppSettings:
        """Load settings from persistence backend."""
        data = self._persistence.load()
        if not data:
            return AppSettings()
        
        # Parse values with type coercion
        kwargs = {}
        for f in fields(AppSettings):
            if f.name not in data:
                continue
            val = data[f.name]
            if f.type == float or f.type == "float":
                try:
                    kwargs[f.name] = float(val)
                except (TypeError, ValueError):
                    pass
            elif f.type == bool or f.type == "bool":
                if isinstance(val, str):
                    kwargs[f.name] = bool(int(val))
                else:
                    kwargs[f.name] = bool(val)
            elif f.type == Optional[str] or f.type == "Optional[str]":
                kwargs[f.name] = str(val) if val is not None else None
            else:
                kwargs[f.name] = val
        
        return AppSettings(**kwargs)

    def get(self) -> AppSettings:
        with self._lock:
            return self._settings

    def update(self, **kwargs) -> AppSettings:
        with self._lock:
            new_settings = replace(self._settings, **kwargs)
            self._settings = new_settings
            callbacks = list(self._subscribers.values())
            self._persist(new_settings)
        for callback in callbacks:
            try:
                callback(new_settings)
            except Exception as exc:
                logger.debug("App settings subscriber callback failed: %s", exc)
                continue
        return new_settings

    def subscribe(self, callback: Callable[[AppSettings], None], *, replay: bool = True) -> Callable[[], None]:
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._subscribers[token] = callback
            snapshot = self._settings
        if replay:
            callback(snapshot)

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(token, None)

        return unsubscribe

    def _persist(self, settings: AppSettings) -> None:
        """Persist settings to the configured backend."""
        data = asdict(settings)
        # Convert booleans to int for compatibility
        for key, val in data.items():
            if isinstance(val, bool):
                data[key] = int(val)
        self._persistence.save(data)


__all__ = ["AppSettings", "AppSettingsStore", "SettingsPersistence", "InMemoryPersistence"]

