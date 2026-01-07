from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import List, Type, Optional

from PySide6 import QtWidgets

logger = logging.getLogger(__name__)


class BaseTab(QtWidgets.QWidget):
    """
    Base class for all SpikeHound tab plugins.
    
    Plugins inheriting from this class and placed in the 'gui/tabs/' 
    directory will be automatically discovered and added to the 
    main workspace.
    """
    
    # Title used for the tab in the UI
    TAB_TITLE = "Plugin Tab"

    def __init__(self, runtime, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.runtime = runtime


class TabPluginManager:
    """Manages the discovery and instantiation of tab plugins."""

    def __init__(self, runtime, tab_directory: Optional[str] = None) -> None:
        self._runtime = runtime
        
        if tab_directory is None:
            # Default to gui/tabs relative to this file's package
            self._tab_directory = Path(__file__).resolve().parent / "tabs"
        else:
            self._tab_directory = Path(tab_directory)
            
        self._plugins: List[BaseTab] = []

    def discover_and_instantiate(self) -> List[BaseTab]:
        """
        Scan the tab directory, load modules, and instantiate BaseTab subclasses.
        """
        if not self._tab_directory.exists():
            try:
                self._tab_directory.mkdir(parents=True, exist_ok=True)
                # Create a placeholder __init__.py
                (self._tab_directory / "__init__.py").touch()
            except Exception as e:
                logger.error("Failed to create tab directory at %s: %s", self._tab_directory, e)
                return []

        instantiated_tabs: List[BaseTab] = []
        
        # Look for .py files
        for file_path in self._tab_directory.glob("*.py"):
            if file_path.name == "__init__.py":
                continue
            
            module_name = f"gui.tabs.{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    for _, obj in inspect.getmembers(module):
                        # Filter for classes that inherit from BaseTab but are not BaseTab itself
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseTab) and 
                            obj is not BaseTab):
                            
                            logger.info("Found tab plugin: %s", obj.__name__)
                            try:
                                # Instantiate with runtime
                                tab_instance = obj(self._runtime)
                                instantiated_tabs.append(tab_instance)
                            except Exception as instance_error:
                                logger.error("Failed to instantiate tab %s: %s", obj.__name__, instance_error)
                                
                except Exception as eval_error:
                    logger.error("Failed to execute module %s: %s", module_name, eval_error)
        
        return instantiated_tabs
