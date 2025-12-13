"""ScopeConfigManager - Handles saving and loading scope configuration.

Extracted from MainWindow to provide a focused component for:
- Collecting current scope configuration
- Saving to JSON files
- Loading from JSON files
- Auto-loading default config on startup
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, TYPE_CHECKING

from PySide6 import QtGui, QtWidgets

if TYPE_CHECKING:
    from .types import ChannelConfig

logger = logging.getLogger(__name__)


class ScopeConfigProvider(Protocol):
    """Protocol defining what the config manager needs from its parent window."""
    
    def statusBar(self) -> QtWidgets.QStatusBar: ...
    
    @property
    def device_combo(self) -> QtWidgets.QComboBox: ...
    
    @property
    def window_combo(self) -> QtWidgets.QComboBox: ...
    
    @property
    def available_combo(self) -> QtWidgets.QComboBox: ...
    
    @property
    def active_combo(self) -> QtWidgets.QComboBox: ...
    
    @property
    def runtime(self) -> Any: ...
    
    @property
    def _current_window_sec(self) -> float: ...
    
    @property
    def _channel_ids_current(self) -> List[int]: ...
    
    @property
    def _channel_names(self) -> List[str]: ...
    
    @property
    def _channel_configs(self) -> Dict[int, "ChannelConfig"]: ...
    
    @property
    def _controller(self) -> Any: ...
    
    def _current_sample_rate_value(self) -> float: ...
    def _set_sample_rate_value(self, value: float) -> None: ...
    def _set_window_combo_value(self, value: float) -> None: ...
    def _on_available_channels(self, channels: Sequence[Any]) -> None: ...
    def _publish_active_channels(self) -> None: ...


class ScopeConfigManager:
    """Manages scope configuration save/load operations.
    
    Works with a provider (typically MainWindow) that exposes the necessary
    UI elements and callbacks via the ScopeConfigProvider protocol.
    """

    def __init__(self, provider: ScopeConfigProvider) -> None:
        self._provider = provider
        self._logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Color conversion utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def color_to_tuple(color: QtGui.QColor) -> tuple[int, int, int, int]:
        if not isinstance(color, QtGui.QColor):
            return (0, 0, 0, 255)
        return (color.red(), color.green(), color.blue(), color.alpha())

    @staticmethod
    def color_from_tuple(data: Sequence[int]) -> QtGui.QColor:
        try:
            r, g, b, a = (int(x) for x in data)
            return QtGui.QColor(r, g, b, a)
        except Exception as e:
            logger.debug("Failed to parse color tuple: %s", e)
            return QtGui.QColor(0, 0, 139)

    # -------------------------------------------------------------------------
    # Channel config serialization
    # -------------------------------------------------------------------------

    def channel_config_to_dict(self, config: "ChannelConfig") -> dict:
        """Convert a ChannelConfig to a JSON-serializable dict."""
        return {
            "color": self.color_to_tuple(config.color),
            "display_enabled": bool(config.display_enabled),
            "vertical_span_v": float(config.vertical_span_v),
            "screen_offset": float(config.screen_offset),
            "notch_enabled": bool(config.notch_enabled),
            "notch_freq_hz": float(config.notch_freq_hz),
            "highpass_enabled": bool(config.highpass_enabled),
            "highpass_hz": float(config.highpass_hz),
            "lowpass_enabled": bool(config.lowpass_enabled),
            "lowpass_hz": float(config.lowpass_hz),
            "listen_enabled": bool(config.listen_enabled),
            "analyze_enabled": bool(config.analyze_enabled),
            "channel_name": config.channel_name,
        }

    def channel_config_from_dict(
        self, payload: dict, *, fallback_name: str = ""
    ) -> "ChannelConfig":
        """Create a ChannelConfig from a dict, with defaults for missing fields."""
        from .types import ChannelConfig
        
        cfg = ChannelConfig()
        try:
            cfg.color = self.color_from_tuple(payload.get("color", (0, 0, 139, 255)))
            cfg.display_enabled = bool(payload.get("display_enabled", True))
            cfg.vertical_span_v = float(payload.get("vertical_span_v", payload.get("range_v", 1.0)))
            cfg.screen_offset = float(payload.get("screen_offset", payload.get("offset_v", 0.5)))
            cfg.notch_enabled = bool(payload.get("notch_enabled", False))
            cfg.notch_freq_hz = float(payload.get("notch_freq_hz", payload.get("notch_freq", 60.0)))
            cfg.highpass_enabled = bool(payload.get("highpass_enabled", False))
            cfg.highpass_hz = float(payload.get("highpass_hz", payload.get("highpass_freq", 10.0)))
            cfg.lowpass_enabled = bool(payload.get("lowpass_enabled", False))
            cfg.lowpass_hz = float(payload.get("lowpass_hz", payload.get("lowpass_freq", 1_000.0)))
            cfg.listen_enabled = bool(payload.get("listen_enabled", False))
            cfg.analyze_enabled = bool(payload.get("analyze_enabled", False))
            cfg.channel_name = str(payload.get("channel_name") or fallback_name or "")
        except Exception as exc:
            self._logger.debug("Failed to parse channel config: %s", exc)
            cfg.channel_name = fallback_name
        return cfg

    # -------------------------------------------------------------------------
    # Collect current configuration
    # -------------------------------------------------------------------------

    def collect_config(self) -> Optional[dict]:
        """Collect the current scope configuration as a dict.
        
        Returns None and shows a message if no device is selected.
        """
        p = self._provider
        device_key = p.device_combo.currentData()
        if device_key is None:
            QtWidgets.QMessageBox.information(
                p, "Save Config", "Select a device before saving a scope config."
            )
            return None
        
        window_value = float(p.window_combo.currentData() or p._current_window_sec or 0.0)
        payload = {
            "version": 2,
            "device_key": device_key,
            "sample_rate": float(p._current_sample_rate_value()),
            "window_sec": float(window_value),
            "channels": [],
        }
        
        channel_names = {cid: name for cid, name in zip(p._channel_ids_current, p._channel_names)}
        for cid in p._channel_ids_current:
            cfg = p._channel_configs.get(cid)
            if cfg is None:
                continue
            payload["channels"].append({
                "id": cid,
                "name": channel_names.get(cid) or cfg.channel_name or f"Channel {cid}",
                "config": self.channel_config_to_dict(cfg),
            })
        return payload

    # -------------------------------------------------------------------------
    # Find channel by ID
    # -------------------------------------------------------------------------

    def find_available_index_by_id(self, channel_id: int) -> int:
        """Find the index of a channel in the available_combo by its ID."""
        p = self._provider
        for idx in range(p.available_combo.count()):
            info = p.available_combo.itemData(idx)
            if getattr(info, "id", None) == channel_id:
                return idx
        return -1

    # -------------------------------------------------------------------------
    # Apply loaded configuration
    # -------------------------------------------------------------------------

    def apply_config_data(
        self, data: dict, source: str = "", *, show_dialogs: bool = True
    ) -> None:
        """Apply a loaded configuration to the scope.
        
        Args:
            data: The configuration dict (from JSON)
            source: Optional source path for status messages
            show_dialogs: If True, show message boxes; if False, use status bar only
        """
        p = self._provider

        def _info(title: str, message: str) -> None:
            if show_dialogs:
                QtWidgets.QMessageBox.information(p, title, message)
            else:
                p.statusBar().showMessage(message, 5000)

        def _warning(title: str, message: str) -> None:
            if show_dialogs:
                QtWidgets.QMessageBox.warning(p, title, message)
            else:
                p.statusBar().showMessage(message, 7000)

        def _critical(title: str, message: str) -> None:
            if show_dialogs:
                QtWidgets.QMessageBox.critical(p, title, message)
            else:
                p.statusBar().showMessage(message, 8000)

        version = int(data.get("version", 1) or 1)
        if version not in {1, 2}:
            _warning("Load Config", f"Unsupported config version: {version}")
            return
        
        device_key = data.get("device_key")
        sample_rate = float(data.get("sample_rate", p._current_sample_rate_value()))
        window_sec = float(data.get("window_sec", p._current_window_sec))
        channels_payload = data.get("channels") or []

        if device_key is not None:
            idx = p.device_combo.findData(device_key)
            if idx >= 0:
                p.device_combo.setCurrentIndex(idx)
            else:
                _warning(
                    "Load Config",
                    f"Device '{device_key}' is not available; cannot load configuration"
                    f"{f' from {source}' if source else ''}.",
                )
                return

        p._set_sample_rate_value(sample_rate)
        p._set_window_combo_value(window_sec)

        if device_key is None:
            _info("Load Config", "No device specified in the configuration.")
            return

        try:
            p.runtime.connect_device(device_key, sample_rate=p._current_sample_rate_value())
        except Exception as exc:
            _critical("Load Config", f"Failed to connect to device '{device_key}': {exc}")
            return

        # Refresh channel lists with the newly connected device
        try:
            available_channels = p.runtime.device_manager.get_available_channels()
            p._on_available_channels(available_channels)
        except Exception as exc:
            self._logger.debug("Failed to refresh channels after load: %s", exc)

        missing_channels: list[int] = []
        p.active_combo.blockSignals(True)
        p.available_combo.blockSignals(True)
        p.active_combo.clear()
        p._channel_configs.clear()
        try:
            for entry in channels_payload:
                cid = entry.get("id")
                if cid is None:
                    continue
                idx = self.find_available_index_by_id(int(cid))
                if idx < 0:
                    missing_channels.append(int(cid))
                    continue
                info = p.available_combo.itemData(idx)
                name = entry.get("name") or p.available_combo.itemText(idx)
                
                p.active_combo.addItem(name, info)
                p.available_combo.removeItem(idx)
                
                cfg = self.channel_config_from_dict(entry.get("config") or {}, fallback_name=name)
                cfg.channel_name = name
                p._channel_configs[int(cid)] = cfg
            if p.active_combo.count():
                p.active_combo.setCurrentIndex(0)
        finally:
            p.active_combo.blockSignals(False)
            p.available_combo.blockSignals(False)

        p._publish_active_channels()
        if missing_channels:
            missing_str = ", ".join(str(cid) for cid in missing_channels)
            _info("Load Config", f"Loaded with missing channels: {missing_str}")
        else:
            msg = f"Scope configuration loaded{f' from {source}' if source else ''}."
            p.statusBar().showMessage(msg, 5000)

    # -------------------------------------------------------------------------
    # Save/Load dialogs
    # -------------------------------------------------------------------------

    def save_config(self) -> None:
        """Show save dialog and save current configuration."""
        p = self._provider
        payload = self.collect_config()
        if payload is None:
            return
        
        default_path = Path.home() / "spikehound_scope.json"
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            p,
            "Save Scope Configuration",
            str(default_path),
            "JSON Files (*.json)",
        )
        if not path_str:
            return
        
        path = Path(path_str)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        try:
            path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                p,
                "Save Config",
                f"Failed to save configuration: {exc}",
            )
            return
        p.statusBar().showMessage(f"Saved scope configuration to {path}", 5000)

    def load_config(self) -> None:
        """Show load dialog and apply selected configuration."""
        p = self._provider
        start_dir = str(Path.home())
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            p,
            "Load Scope Configuration",
            start_dir,
            "JSON Files (*.json)",
        )
        if not path_str:
            return
        
        path = Path(path_str)
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                p,
                "Load Config",
                f"Failed to read configuration: {exc}",
            )
            return
        
        if not isinstance(data, dict):
            QtWidgets.QMessageBox.critical(p, "Load Config", "Configuration file is not valid JSON.")
            return
        
        self.apply_config_data(data, source=str(path))

    def try_load_default_config(self) -> None:
        """Try to load a default configuration on startup.
        
        Checks for:
        1. Launch config preference in app settings
        2. default_config.json in current working directory
        """
        p = self._provider

        # Check for launch config preference
        if p._controller and p._controller.app_settings_store:
            settings = p._controller.app_settings_store.get()
            if settings.load_config_on_launch and settings.launch_config_path:
                launch_path = Path(settings.launch_config_path)
                if launch_path.is_file():
                    try:
                        data = json.loads(launch_path.read_text())
                        if isinstance(data, dict):
                            self.apply_config_data(data, source=str(launch_path), show_dialogs=False)
                            p.statusBar().showMessage(f"Loaded launch config: {launch_path.name}", 5000)
                            return
                    except Exception as exc:
                        p.statusBar().showMessage(f"Failed to load launch config: {exc}", 7000)

        # Fallback to default_config.json in CWD
        default_path = Path.cwd() / "default_config.json"
        if not default_path.is_file():
            return
        
        try:
            data = json.loads(default_path.read_text())
        except Exception as exc:
            p.statusBar().showMessage(f"Failed to read default_config.json: {exc}", 7000)
            return
        
        if not isinstance(data, dict):
            p.statusBar().showMessage("default_config.json is not valid JSON.", 7000)
            return
        
        self.apply_config_data(data, source=str(default_path), show_dialogs=False)


__all__ = ["ScopeConfigManager", "ScopeConfigProvider"]
