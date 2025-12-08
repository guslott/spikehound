"""DeviceManager - Qt adapter for DeviceRegistry.

Provides Qt signals for GUI integration while delegating device operations
to the pure-Python DeviceRegistry in core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PySide6 import QtCore

if TYPE_CHECKING:  # pragma: no cover
    from daq.base_device import BaseDevice, ChannelInfo
    from core.device_registry import DeviceRegistry, RegistryEvent
else:
    BaseDevice = Any
    ChannelInfo = Any


class DeviceManager(QtCore.QObject):
    """Qt adapter that exposes DeviceRegistry events as Qt signals.
    
    This class wraps a pure-Python DeviceRegistry and converts its callback-based
    events into Qt signals for seamless GUI integration.
    """

    devicesChanged = QtCore.Signal(list)
    deviceConnected = QtCore.Signal(str)
    deviceDisconnected = QtCore.Signal()
    availableChannelsChanged = QtCore.Signal(list)

    def __init__(
        self,
        registry: "DeviceRegistry",
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._listener_token = registry.add_listener(self._on_registry_event)

    def _on_registry_event(self, event: "RegistryEvent") -> None:
        """Handle events from the DeviceRegistry and emit Qt signals."""
        from core.device_registry import RegistryEventType
        
        if event.event_type == RegistryEventType.DEVICES_CHANGED:
            self.devicesChanged.emit(event.data or [])
        elif event.event_type == RegistryEventType.DEVICE_CONNECTED:
            self.deviceConnected.emit(str(event.data) if event.data else "")
        elif event.event_type == RegistryEventType.DEVICE_DISCONNECTED:
            self.deviceDisconnected.emit()
        elif event.event_type == RegistryEventType.CHANNELS_CHANGED:
            self.availableChannelsChanged.emit(event.data or [])

    # -------------------------------------------------------------------------
    # Delegate methods to registry
    # -------------------------------------------------------------------------

    def set_list_all_audio_devices(self, enabled: bool, *, refresh: bool = False) -> None:
        """Configure whether to list all audio devices or just defaults."""
        self._registry.set_list_all_audio_devices(enabled, refresh=refresh)

    def refresh_devices(self) -> None:
        """Scan for available devices (emits devicesChanged signal)."""
        self._registry.refresh_devices()

    def get_device_list(self) -> List[Dict[str, object]]:
        """Return cached device list."""
        return self._registry.get_device_list()

    def connect_device(
        self,
        device_key: str,
        sample_rate: float,
        *,
        chunk_size: int = 1024,
        **driver_kwargs,
    ) -> "BaseDevice":
        """Connect to a device by key."""
        return self._registry.connect_device(
            device_key, sample_rate, chunk_size=chunk_size, **driver_kwargs
        )

    def disconnect_device(self) -> None:
        """Disconnect the currently connected device."""
        self._registry.disconnect_device()

    def get_available_channels(self) -> List["ChannelInfo"]:
        """Return channels available on the connected device."""
        return self._registry.get_available_channels()

    def active_key(self) -> Optional[str]:
        """Return the key of the currently connected device, or None."""
        return self._registry.active_key()

    def current_driver(self) -> Optional["BaseDevice"]:
        """Return the currently connected driver instance, or None."""
        return self._registry.current_driver()

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove listener from registry."""
        if self._listener_token is not None:
            self._registry.remove_listener(self._listener_token)
            self._listener_token = None


    def __del__(self) -> None:
        """Ensure local listener is removed if the object is garbage collected."""
        try:
            self.cleanup()
        except ImportError:
            # Can happen during interpreter shutdown
            pass
        except Exception:
            pass


__all__ = ["DeviceManager"]
