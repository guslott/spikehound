"""Verify core/shared modules are importable without PySide6.

These tests ensure the Qt decoupling is correctly implemented.
"""
from __future__ import annotations

import sys
import pytest


class TestHeadlessImports:
    """Test that core modules can be imported without PySide6."""

    def test_dispatcher_headless_import(self, monkeypatch):
        """Dispatcher should be importable without PySide6."""
        # Remove any cached imports
        modules_to_remove = [k for k in sys.modules if 'core.dispatcher' in k]
        for mod in modules_to_remove:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        # Block PySide6 imports
        monkeypatch.setitem(sys.modules, 'PySide6', None)
        monkeypatch.setitem(sys.modules, 'PySide6.QtCore', None)
        
        # This should not raise ImportError
        from core.dispatcher import Dispatcher, TickCallback
        
        assert Dispatcher is not None
        assert TickCallback is not None

    def test_app_settings_headless_import(self, monkeypatch):
        """AppSettingsStore should be importable without PySide6."""
        # Remove any cached imports
        modules_to_remove = [k for k in sys.modules if 'shared.app_settings' in k]
        for mod in modules_to_remove:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        # Block PySide6 imports
        monkeypatch.setitem(sys.modules, 'PySide6', None)
        monkeypatch.setitem(sys.modules, 'PySide6.QtCore', None)
        
        # This should not raise ImportError
        from shared.app_settings import AppSettings, AppSettingsStore, InMemoryPersistence
        
        assert AppSettings is not None
        assert AppSettingsStore is not None
        assert InMemoryPersistence is not None

    def test_app_settings_store_works_headless(self, monkeypatch):
        """AppSettingsStore should work in headless mode with InMemoryPersistence."""
        # Remove any cached imports
        modules_to_remove = [k for k in sys.modules if 'shared.app_settings' in k]
        for mod in modules_to_remove:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        # Block PySide6 imports
        monkeypatch.setitem(sys.modules, 'PySide6', None)
        monkeypatch.setitem(sys.modules, 'PySide6.QtCore', None)
        
        from shared.app_settings import AppSettingsStore
        
        store = AppSettingsStore()
        settings = store.get()
        
        assert settings.plot_refresh_hz == 40.0  # default value
        
        # Update and verify
        new_settings = store.update(plot_refresh_hz=60.0)
        assert new_settings.plot_refresh_hz == 60.0
        assert store.get().plot_refresh_hz == 60.0

    def test_controller_headless_import(self, monkeypatch):
        """PipelineController should be importable without PySide6."""
        # Remove any cached imports of core modules
        modules_to_remove = [
            k for k in sys.modules 
            if any(prefix in k for prefix in ['core.controller', 'core.dispatcher', 'shared.app_settings'])
        ]
        for mod in modules_to_remove:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        # Block PySide6 imports
        monkeypatch.setitem(sys.modules, 'PySide6', None)
        monkeypatch.setitem(sys.modules, 'PySide6.QtCore', None)
        
        # This should not raise ImportError
        from core.controller import PipelineController
        
        assert PipelineController is not None
