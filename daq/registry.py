"""DAQ plugin registry for SpikeHound backends.

Drivers live in :mod:`daq` and must subclass :class:`~daq.base_device.BaseDevice`.
Custom drivers should implement the BaseDevice contract (open/configure/start)
so they can be discovered at runtime. This module loads any ``*.py`` file in the
package (excluding ``base_device.py``, ``registry.py`` and module initialisers),
searches for concrete subclasses, and exposes helpers for listing and creating
known devices.

Example::

    from daq.registry import scan_devices, list_devices

    scan_devices()
    for device in list_devices():
        print(device.key, device.name)
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import pkgutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .base_device import BaseDevice, DeviceInfo

logger = logging.getLogger(__name__)

_EXCLUDE = {"base_device", "registry", "__init__"}
_REGISTRY: Dict[str, "DeviceDescriptor"] = {}
_scanned = False


@dataclass
class DeviceDescriptor:
    """Metadata for a discovered DAQ backend."""

    key: str
    name: str
    cls: Type[BaseDevice]
    module: str
    capabilities: Dict[str, object]


def scan_devices(force: bool = False) -> None:
    """Populate the device registry by inspecting modules under :mod:`daq`."""

    global _scanned
    if _scanned and not force:
        return

    _REGISTRY.clear()
    package = __name__.rsplit(".", 1)[0]
    for module_info in pkgutil.iter_modules([os.path.dirname(__file__)], package + "."):
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name in _EXCLUDE:
            continue
        try:
            module = importlib.import_module(module_info.name)
        except Exception as exc:
            logger.debug("Failed to import DAQ module %s: %s", module_info.name, exc)
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseDevice) or obj is BaseDevice:
                continue
            if inspect.isabstract(obj):
                continue
            key = f"{obj.__module__}.{obj.__name__}"
            if key in _REGISTRY:
                continue
            name = getattr(obj, "device_class_name", lambda: obj.__name__)()
            capabilities = _describe_capabilities(obj)
            _REGISTRY[key] = DeviceDescriptor(
                key=key,
                name=name,
                cls=obj,
                module=obj.__module__,
                capabilities=capabilities,
            )

    _scanned = True


def list_devices() -> List[DeviceDescriptor]:
    """Return descriptors for all discovered devices."""

    scan_devices()
    return list(_REGISTRY.values())


def create_device(key: str, **kwargs) -> BaseDevice:
    """Instantiate the backend associated with ``key``."""

    scan_devices()
    descriptor = _REGISTRY.get(key)
    if descriptor is None:
        raise KeyError(f"No DAQ backend registered for key {key!r}")
    return descriptor.cls(**kwargs)


def _describe_capabilities(cls: Type[BaseDevice]) -> Dict[str, object]:
    """Collect lightweight capability hints from the driver class."""

    capabilities: Dict[str, object] = {
        "supports_input": True,
    }
    doc = inspect.getdoc(cls)
    if doc:
        capabilities["description"] = doc.splitlines()[0]
    for attr in ("DEFAULT_SAMPLE_RATE", "MAX_CHANNELS", "NOTES"):
        if hasattr(cls, attr):
            capabilities[attr.lower()] = getattr(cls, attr)
    return capabilities


__all__ = ["scan_devices", "list_devices", "create_device", "DeviceDescriptor"]
