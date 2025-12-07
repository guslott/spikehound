"""DeviceRegistry - Pure-Python device discovery and connection management.

This module provides GUI-agnostic device management, enabling headless device
discovery and connection without Qt dependencies.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from daq.base_device import BaseDevice, ChannelInfo
    from daq.registry import DeviceDescriptor
else:  # pragma: no cover - runtime fallback
    BaseDevice = Any
    ChannelInfo = Any
    DeviceDescriptor = Any

logger = logging.getLogger(__name__)


def _registry():
    """Lazy import to avoid circular dependencies."""
    from daq import registry as reg
    return reg


class RegistryEventType(Enum):
    """Event types emitted by DeviceRegistry."""
    DEVICES_CHANGED = auto()
    DEVICE_CONNECTED = auto()
    DEVICE_DISCONNECTED = auto()
    CHANNELS_CHANGED = auto()


@dataclass
class RegistryEvent:
    """Event payload from DeviceRegistry."""
    event_type: RegistryEventType
    data: Any = None


# Listener callback signature: (event: RegistryEvent) -> None
RegistryListener = Callable[[RegistryEvent], None]


class DeviceRegistry:
    """Pure-Python device registry with callback-based notifications.
    
    Tracks available DAQ drivers and manages a single active connection.
    Uses callback pattern instead of Qt signals for GUI-agnostic operation.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._listeners: Dict[int, RegistryListener] = {}
        self._next_token: int = 0
        
        # Device state
        self._descriptors: Dict[str, DeviceDescriptor] = {}
        self._device_entries: Dict[str, Dict[str, object]] = {}
        self._active_key: Optional[str] = None
        self._driver: Optional[BaseDevice] = None
        self._channels: List[ChannelInfo] = []
        self._list_all_audio_devices: bool = False
        
        # Device list cache
        self._device_list: List[Dict[str, object]] = []

    # -------------------------------------------------------------------------
    # Listener Management
    # -------------------------------------------------------------------------

    def add_listener(self, callback: RegistryListener) -> int:
        """Register a listener for registry events.
        
        Args:
            callback: Function called with RegistryEvent on state changes
            
        Returns:
            Token for removing the listener
        """
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._listeners[token] = callback
            return token

    def remove_listener(self, token: int) -> None:
        """Unregister a listener by token."""
        with self._lock:
            self._listeners.pop(token, None)

    def _emit(self, event_type: RegistryEventType, data: Any = None) -> None:
        """Emit an event to all listeners."""
        event = RegistryEvent(event_type=event_type, data=data)
        with self._lock:
            listeners = list(self._listeners.values())
        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:
                logger.debug("Registry listener error: %s", exc)

    # -------------------------------------------------------------------------
    # Device Discovery
    # -------------------------------------------------------------------------

    def set_list_all_audio_devices(self, enabled: bool, *, refresh: bool = False) -> None:
        """Configure whether to list all audio devices or just defaults."""
        self._list_all_audio_devices = bool(enabled)
        if refresh:
            self.refresh_devices()

    def refresh_devices(self) -> List[Dict[str, object]]:
        """Scan for available devices and return the device list.
        
        Also emits DEVICES_CHANGED event to listeners.
        
        Returns:
            List of device info dictionaries
        """
        reg = _registry()
        reg.scan_devices(force=True)
        descriptors = reg.list_devices()
        
        # Configure soundcard listing preference
        try:
            from daq.soundcard_source import SoundCardSource
            SoundCardSource.set_list_all_devices(self._list_all_audio_devices)
        except Exception:
            pass
        
        self._descriptors = {d.key: d for d in descriptors}
        self._device_entries = {}
        payload: List[Dict[str, object]] = []
        
        for descriptor in descriptors:
            try:
                available = descriptor.cls.list_available_devices()
            except Exception:
                continue

            if not available:
                continue

            for device in sorted(available, key=lambda info: info.name.lower()):
                entry_key = f"{descriptor.key}::{device.id}"
                self._device_entries[entry_key] = {
                    "descriptor_key": descriptor.key,
                    "device_id": device.id,
                }
                capabilities = descriptor.capabilities
                try:
                    drv = reg.create_device(descriptor.key)
                    capabilities = drv.get_capabilities(device.id)
                except Exception:
                    pass
                    
                entry: Dict[str, object] = {
                    "key": entry_key,
                    "name": f"{descriptor.name} - {device.name}",
                    "module": descriptor.module,
                    "capabilities": capabilities,
                    "driver_key": descriptor.key,
                    "driver_name": descriptor.name,
                    "device_id": device.id,
                    "device_name": device.name,
                }
                vendor = getattr(device, "vendor", None)
                if vendor:
                    entry["device_vendor"] = vendor
                details = getattr(device, "details", None)
                if details:
                    entry["device_details"] = details
                payload.append(entry)

        # Sort by priority: Sound card > Simulated > Other
        def _priority(item: Dict[str, object]) -> tuple:
            driver = str(item.get("driver_name", "")).lower()
            module = str(item.get("module", "")).lower()
            name = str(item.get("device_name", item.get("name", ""))).lower()
            if "sound card" in driver or "soundcard" in module:
                rank = 0
            elif "simulated" in driver or "simulated" in module:
                rank = 1
            else:
                rank = 2
            return (rank, driver, name)

        payload.sort(key=_priority)
        self._device_list = payload
        self._emit(RegistryEventType.DEVICES_CHANGED, payload)
        return payload

    def get_device_list(self) -> List[Dict[str, object]]:
        """Return cached device list (call refresh_devices() to update)."""
        return list(self._device_list)

    def get_descriptors(self) -> List[DeviceDescriptor]:
        """Return list of device descriptors."""
        return list(self._descriptors.values())

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def connect_device(
        self,
        device_key: str,
        sample_rate: float,
        *,
        chunk_size: int = 1024,
        **driver_kwargs,
    ) -> BaseDevice:
        """Connect to a device by key.
        
        Args:
            device_key: Device key from the device list
            sample_rate: Requested sample rate
            chunk_size: Buffer chunk size
            **driver_kwargs: Additional driver-specific arguments
            
        Returns:
            Connected BaseDevice instance
            
        Raises:
            KeyError: Unknown device key
            RuntimeError: Connection failed
        """
        self.disconnect_device()

        entry = self._device_entries.get(device_key)
        if entry is None:
            raise KeyError(f"Unknown device key: {device_key!r}")

        descriptor_key = entry.get("descriptor_key")
        device_id = entry.get("device_id")
        if descriptor_key is None or not isinstance(descriptor_key, str):
            raise KeyError(f"Invalid descriptor for key: {device_key!r}")

        descriptor = self._descriptors.get(descriptor_key)
        if descriptor is None:
            raise KeyError(f"Unknown descriptor key: {descriptor_key!r}")

        if device_id is None or device_id == "":
            error_message = entry.get("error") or "No hardware devices detected for this driver."
            raise RuntimeError(error_message)

        reg = _registry()
        driver = reg.create_device(descriptor_key, **driver_kwargs)
        available = descriptor.cls.list_available_devices()
        target = next((dev for dev in available if str(dev.id) == str(device_id)), None)
        if target is None:
            raise RuntimeError(f"Selected device is no longer available: {device_id!r}")

        driver.open(target.id)
        channels = driver.list_available_channels(target.id)

        # Handle devices that determine sample rate after opening (e.g., FileSource)
        effective_sample_rate = sample_rate
        if effective_sample_rate <= 0:
            try:
                caps = driver.get_capabilities(target.id)
                if caps.sample_rates and len(caps.sample_rates) > 0:
                    effective_sample_rate = caps.sample_rates[0]
            except Exception:
                pass
        
        if effective_sample_rate <= 0:
            driver.close()
            raise RuntimeError("Could not determine sample rate for device")

        configure_kwargs = {
            "sample_rate": int(effective_sample_rate),
            "channels": [ch.id for ch in channels] if channels else None,
            "chunk_size": chunk_size,
        }
        configure_kwargs = {k: v for k, v in configure_kwargs.items() if v is not None}

        try:
            driver.configure(**configure_kwargs)
        except Exception:
            driver.close()
            raise

        self._driver = driver
        self._active_key = device_key
        self._channels = list(channels)
        self._emit(RegistryEventType.CHANNELS_CHANGED, list(self._channels))
        # NOTE: DEVICE_CONNECTED is emitted by caller (runtime) after dispatcher is wired
        return driver

    def disconnect_device(self) -> None:
        """Disconnect the currently connected device."""
        if self._driver is None:
            return
        try:
            if getattr(self._driver, "running", False):
                try:
                    self._driver.stop()
                except Exception:
                    pass
            try:
                self._driver.close()
            except Exception:
                pass
        finally:
            self._driver = None
            self._active_key = None
            self._channels = []
            self._emit(RegistryEventType.CHANNELS_CHANGED, [])
            self._emit(RegistryEventType.DEVICE_DISCONNECTED)

    def emit_device_connected(self, device_key: str) -> None:
        """Emit DEVICE_CONNECTED event (called by runtime after dispatcher is ready)."""
        self._emit(RegistryEventType.DEVICE_CONNECTED, device_key)

    # -------------------------------------------------------------------------
    # State Accessors
    # -------------------------------------------------------------------------

    def get_available_channels(self) -> List[ChannelInfo]:
        """Return channels available on the connected device."""
        return list(self._channels)

    def active_key(self) -> Optional[str]:
        """Return the key of the currently connected device, or None."""
        return self._active_key

    def current_driver(self) -> Optional[BaseDevice]:
        """Return the currently connected driver instance, or None."""
        return self._driver


__all__ = ["DeviceRegistry", "RegistryEvent", "RegistryEventType", "RegistryListener"]
