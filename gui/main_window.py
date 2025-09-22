import importlib
import inspect
import pkgutil
from typing import List, Type

from PySide6 import QtWidgets

import daq
from daq.base_source import BaseSource


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

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        self._devices_menu = menubar.addMenu("Devices")
        self._populate_devices_menu()

    def _populate_devices_menu(self) -> None:
        self._devices_menu.clear()
        classes = sorted(_load_device_classes(), key=lambda cls: cls.device_class_name())
        for cls in classes:
            submenu = self._devices_menu.addMenu(cls.device_class_name())
            submenu.setEnabled(False)
