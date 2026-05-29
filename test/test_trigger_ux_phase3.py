"""Phase 3 UX (3C): viewport bounds, the on-plot volt readout, and a black
threshold line shown only in triggered modes. The line is hidden in
No-Trigger / Stream mode (no preview); 3C's "color to channel" is intentionally
not implemented (the line stays black).
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from gui.main_window import MainWindow
from gui.scope_widget import ScopeWidget
from gui.types import ChannelConfig
from core.runtime import SpikeHoundRuntime


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def main_window(monkeypatch):
    _app()
    monkeypatch.setattr(SpikeHoundRuntime, "scan_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_try_load_default_config", lambda self: None)
    window = MainWindow()
    yield window
    window.close()


def _setup_channel(window: MainWindow) -> None:
    window._channel_configs[0] = ChannelConfig(vertical_span_v=1.0, screen_offset=0.5)
    window.trigger_control.update_channels([("Ch 0", 0)])


# --- 3C: bounds + black line ------------------------------------------------

def test_threshold_line_is_clamped_to_viewport() -> None:
    _app()
    scope = ScopeWidget()
    assert list(scope.threshold_line.maxRange) == [0.0, 1.0]
    scope.threshold_line.setValue(2.0)
    assert float(scope.threshold_line.value()) == pytest.approx(1.0)
    scope.threshold_line.setValue(-0.5)
    assert float(scope.threshold_line.value()) == pytest.approx(0.0)


def test_threshold_line_is_black() -> None:
    _app()
    scope = ScopeWidget()
    c = scope.threshold_line.pen.color()
    assert (c.red(), c.green(), c.blue()) == (0, 0, 0)


# --- visibility: hidden in stream, solid black in triggered modes -----------

def test_stream_mode_hides_threshold_line(main_window) -> None:
    _setup_channel(main_window)
    main_window._trigger_controller.configure(
        mode="stream", channel_id=0, threshold=0.3, pre_seconds=0.0, window_sec=1.0,
    )
    assert main_window.scope.threshold_line.isVisible() is False
    assert main_window.scope.pretrigger_line.isVisible() is False


def test_no_channel_hides_threshold_line(main_window) -> None:
    # Stream + no channel -> hidden (channel_index None).
    main_window._trigger_controller.configure(
        mode="stream", channel_id=None, threshold=0.0, pre_seconds=0.0, window_sec=1.0,
    )
    assert main_window.scope.threshold_line.isVisible() is False


def test_triggered_mode_shows_solid_black_line(main_window) -> None:
    _setup_channel(main_window)
    main_window._trigger_controller.configure(
        mode="repeated", channel_id=0, threshold=0.3, pre_seconds=0.0, window_sec=1.0,
    )
    line = main_window.scope.threshold_line
    assert line.isVisible() is True
    assert line.pen.style() == QtCore.Qt.SolidLine
    c = line.pen.color()
    assert (c.red(), c.green(), c.blue()) == (0, 0, 0)


def test_threshold_label_shows_volts(main_window) -> None:
    _setup_channel(main_window)
    main_window._trigger_controller.configure(
        mode="repeated", channel_id=0, threshold=0.42, pre_seconds=0.0, window_sec=1.0,
    )
    assert "0.420" in main_window.scope.threshold_line.label.format
