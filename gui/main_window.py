import importlib
import inspect
import pkgutil
from functools import partial
from typing import List, Type

from PySide6 import QtWidgets

import daq
from daq.base_source import BaseSource, DeviceInfo


def _load_device_classes() -> List[Type[BaseSource]]:
    classes: List[Type[BaseSource]] = []
    for module_info in pkgutil.iter_modules(daq.__path__, daq.__name__ + "."):
        if module_info.name.endswith("base_source"):
            continue
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseSource)
                and obj is not BaseSource
                and obj.__module__ == module.__name__
                and not inspect.isabstract(obj)
            ):
                classes.append(obj)
    return classes


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SpikeHound Oscilloscope")
        self.setFixedSize(800, 600)

        self.statusBar()
        self._active_source: BaseSource | None = None
        self._active_device: DeviceInfo | None = None

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        self._devices_menu = menubar.addMenu("Devices")
        self._populate_devices_menu()

    def _populate_devices_menu(self) -> None:
        self._devices_menu.clear()
        classes = sorted(_load_device_classes(), key=lambda cls: cls.device_class_name())
        for cls in classes:
            submenu = self._devices_menu.addMenu(cls.device_class_name())
            try:
                devices = cls.list_available_devices()
            except Exception as exc:  # pragma: no cover - GUI feedback only
                action = submenu.addAction(f"Unavailable ({exc})")
                action.setEnabled(False)
                continue

            if not devices:
                action = submenu.addAction("No devices detected")
                action.setEnabled(False)
                continue

            for device in devices:
                action = submenu.addAction(device.name)
                action.triggered.connect(partial(self._on_device_selected, cls, device))

    def _on_device_selected(self, source_cls: Type[BaseSource], device: DeviceInfo) -> None:
        try:
            source = source_cls()
        except Exception as exc:  # pragma: no cover - GUI feedback only
            QtWidgets.QMessageBox.critical(
                self,
                "Device Initialization Failed",
                f"Could not create driver for {device.name}: {exc}",
            )
            return

        self._active_source = source
        self._active_device = device
        self.statusBar().showMessage(f"Selected {source_cls.device_class_name()} → {device.name}", 4000)
