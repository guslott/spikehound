from __future__ import annotations

import logging
from dataclasses import dataclass, replace
import threading
from typing import Callable, Dict, Optional

from PySide6.QtCore import QSettings

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


class AppSettingsStore:
    """Thread-safe persistent settings store for application-wide preferences."""

    def __init__(self, *, organization: str = "SpikeHound", application: str = "SpikeHound") -> None:
        self._lock = threading.Lock()
        self._subscribers: Dict[int, Callable[[AppSettings], None]] = {}
        self._next_token = 0
        self._settings = self._load_settings(organization, application)
        self._qsettings = QSettings(organization, application)

    def _load_settings(self, organization: str, application: str) -> AppSettings:
        qsettings = QSettings(organization, application)
        try:
            refresh = float(qsettings.value("plot_refresh_hz", AppSettings.plot_refresh_hz))
        except (TypeError, ValueError):
            refresh = AppSettings.plot_refresh_hz
        try:
            window_sec = float(qsettings.value("default_window_sec", AppSettings.default_window_sec))
        except (TypeError, ValueError):
            window_sec = AppSettings.default_window_sec
        listen = qsettings.value("listen_output_key", AppSettings.listen_output_key)
        if listen is not None:
            listen = str(listen)
        list_all_audio = qsettings.value("list_all_audio_devices", AppSettings.list_all_audio_devices)
        list_all_audio = bool(int(list_all_audio)) if isinstance(list_all_audio, str) else bool(list_all_audio)
        
        load_launch = qsettings.value("load_config_on_launch", AppSettings.load_config_on_launch)
        load_launch = bool(int(load_launch)) if isinstance(load_launch, str) else bool(load_launch)
        
        launch_path = qsettings.value("launch_config_path", AppSettings.launch_config_path)
        if launch_path is not None:
            launch_path = str(launch_path)

        rec_float32 = qsettings.value("recording_use_float32", AppSettings.recording_use_float32)
        rec_float32 = bool(int(rec_float32)) if isinstance(rec_float32, str) else bool(rec_float32)

        rec_autoinc = qsettings.value("recording_auto_increment", AppSettings.recording_auto_increment)
        rec_autoinc = bool(int(rec_autoinc)) if isinstance(rec_autoinc, str) else bool(rec_autoinc)

        return AppSettings(
            plot_refresh_hz=refresh,
            default_window_sec=window_sec,
            listen_output_key=listen,
            list_all_audio_devices=list_all_audio,
            load_config_on_launch=load_launch,
            launch_config_path=launch_path,
            recording_use_float32=rec_float32,
            recording_auto_increment=rec_autoinc,
        )

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
        self._qsettings.setValue("plot_refresh_hz", settings.plot_refresh_hz)
        self._qsettings.setValue("default_window_sec", settings.default_window_sec)
        if settings.listen_output_key is None:
            self._qsettings.remove("listen_output_key")
        else:
            self._qsettings.setValue("listen_output_key", settings.listen_output_key)
        self._qsettings.setValue("list_all_audio_devices", int(bool(settings.list_all_audio_devices)))
        self._qsettings.setValue("load_config_on_launch", int(bool(settings.load_config_on_launch)))
        if settings.launch_config_path is None:
            self._qsettings.remove("launch_config_path")
        else:
            self._qsettings.setValue("launch_config_path", settings.launch_config_path)

        self._qsettings.setValue("recording_use_float32", int(bool(settings.recording_use_float32)))
        self._qsettings.setValue("recording_auto_increment", int(bool(settings.recording_auto_increment)))


__all__ = ["AppSettings", "AppSettingsStore"]
