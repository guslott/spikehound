"""Robustness regression tests for ``daq.registry.scan_devices`` (finding 6.6a).

The per-*module* import is already wrapped in try/except, but the per-*class*
loop was not: a single malformed driver class (e.g. ``device_class_name``
shadowed by a non-callable) raised out of the loop and aborted discovery of
every sibling device. These tests pin that one bad class is skipped while valid
siblings are still registered.
"""
from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

import daq.registry as registry_mod
from daq.base_device import BaseDevice


def _make_concrete(name: str, **overrides):
    """Build a concrete ``BaseDevice`` subclass with all abstract methods stubbed.

    The stubs are never called during a scan (scan only reads the class object),
    so trivial no-op lambdas are sufficient to make the class non-abstract.
    """
    body = {m: (lambda self, *a, **k: None) for m in BaseDevice.__abstractmethods__}
    body.update(overrides)
    return type(name, (BaseDevice,), body)


@pytest.fixture
def _restore_registry():
    """Reset the module-global scan cache so real devices repopulate afterwards."""
    yield
    registry_mod._REGISTRY.clear()
    registry_mod._scanned = False


def _install_fake_module(monkeypatch, **classes):
    fake_module = ModuleType("daq._fake_devices")
    for attr, cls in classes.items():
        setattr(fake_module, attr, cls)
    monkeypatch.setattr(
        registry_mod.pkgutil,
        "iter_modules",
        lambda *a, **k: [SimpleNamespace(name="daq._fake_devices")],
    )
    monkeypatch.setattr(
        registry_mod.importlib,
        "import_module",
        lambda name: fake_module,
    )


def test_malformed_device_class_does_not_abort_sibling_discovery(
    monkeypatch, _restore_registry
):
    # device_class_name shadowed by a non-callable -> calling it raises TypeError.
    bad = _make_concrete("Bad", device_class_name="not-callable")
    good = _make_concrete(
        "Good", device_class_name=classmethod(lambda cls: "Good Device")
    )
    _install_fake_module(monkeypatch, Bad=bad, Good=good)

    # Must not raise, and must still register the valid sibling.
    registry_mod.scan_devices(force=True)

    keys = {d.key for d in registry_mod.list_devices()}
    assert any(k.endswith(".Good") for k in keys), "valid sibling must be discovered"
    assert not any(k.endswith(".Bad") for k in keys), "malformed class must be skipped"


def test_capability_introspection_failure_is_isolated(monkeypatch, _restore_registry):
    # A class whose capability introspection blows up must also be skipped, not fatal.
    good = _make_concrete(
        "Good", device_class_name=classmethod(lambda cls: "Good Device")
    )
    bad = _make_concrete(
        "Bad", device_class_name=classmethod(lambda cls: "Bad Device")
    )
    _install_fake_module(monkeypatch, Bad=bad, Good=good)

    def boom(cls):
        if cls is bad:
            raise RuntimeError("capability probe exploded")
        return {"supports_input": True}

    monkeypatch.setattr(registry_mod, "_describe_capabilities", boom)

    registry_mod.scan_devices(force=True)

    keys = {d.key for d in registry_mod.list_devices()}
    assert any(k.endswith(".Good") for k in keys)
    assert not any(k.endswith(".Bad") for k in keys)
