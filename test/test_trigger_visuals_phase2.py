"""Phase 2 regression tests for the scope trigger: pretrigger placement,
value-vs-structural change semantics, and the centralized coordinate mapping.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

from gui.main_window import MainWindow
from gui.trigger_controller import TriggerController
from gui.trigger_control_widget import TriggerControlWidget
from gui.types import ChannelConfig
from gui.scope_coords import volts_to_norm, norm_to_volts
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


# --- 2C: coordinate mapping is a single, invertible source of truth ----------

def test_scope_coords_roundtrip_scalar_and_array() -> None:
    for span, offset, volts in [(1.0, 0.5, 0.3), (2.0, 0.5, -1.25), (0.5, 0.2, 0.0)]:
        norm = volts_to_norm(volts, span, offset)
        assert norm_to_volts(norm, span, offset) == pytest.approx(volts)

    arr = np.array([-1.0, 0.0, 0.75], dtype=np.float32)
    back = norm_to_volts(volts_to_norm(arr, 2.0, 0.5), 2.0, 0.5)
    assert np.allclose(back, arr, atol=1e-6)


def test_scope_coords_zero_span_is_safe() -> None:
    # Must not divide by zero; just needs to be finite.
    assert np.isfinite(volts_to_norm(1.0, 0.0, 0.5))


# --- 2A: pretrigger marker sits at the trigger instant, not the window start --

def test_pretrigger_marker_sits_at_trigger_instant(main_window) -> None:
    main_window._channel_configs[0] = ChannelConfig()
    main_window._trigger_controller.configure(
        mode="single", channel_id=0, threshold=0.0, pre_seconds=0.05, window_sec=1.0,
    )
    line = main_window.scope.pretrigger_line
    assert line.isVisible() is True
    # Used to be pinned at 0.0 (window start) by _update_trigger_visuals.
    assert float(line.value()) == pytest.approx(0.05)


def test_pretrigger_marker_hidden_in_stream_mode(main_window) -> None:
    main_window._trigger_controller.configure(
        mode="stream", channel_id=None, threshold=0.0, pre_seconds=0.05, window_sec=1.0,
    )
    assert main_window.scope.pretrigger_line.isVisible() is False


def test_threshold_line_position_tracks_configured_volts(main_window) -> None:
    cfg = ChannelConfig(vertical_span_v=2.0, screen_offset=0.5)
    main_window._channel_configs[0] = cfg
    main_window._trigger_controller.configure(
        mode="repeated", channel_id=0, threshold=1.0, pre_seconds=0.0, window_sec=1.0,
    )
    expected = volts_to_norm(1.0, cfg.vertical_span_v, cfg.screen_offset)  # 1/(2*2)+0.5 = 0.75
    assert float(main_window.scope.threshold_line.value()) == pytest.approx(expected)
    assert main_window.scope.threshold_line.isVisible() is True


# --- 2B: threshold value change preserves history; structural change resets ---

def _build_history(controller: TriggerController) -> None:
    controller.update_sample_rate(10_000.0)
    controller.push_samples(np.ones(200, dtype=np.float32), 10_000.0, 1.0)


def test_threshold_value_change_preserves_history_structural_resets() -> None:
    _app()
    controller = TriggerController()
    widget = TriggerControlWidget(controller)
    widget.update_channels([("Ch 0", 0)])

    widget.trigger_mode_single.setChecked(True)
    widget._on_config_changed()  # structural (stream->single): resets
    _build_history(controller)
    assert controller._history_length == 200

    # Pure threshold value change -> history preserved.
    widget.threshold_spin.setValue(0.42)
    assert controller.threshold == pytest.approx(0.42)
    assert controller._history_length == 200

    # Window (structural) change -> history reset.
    idx = widget.window_combo.findData(0.05)
    widget.window_combo.setCurrentIndex(idx)
    assert controller._history_length == 0


def test_threshold_line_drag_preserves_history(main_window) -> None:
    """Dragging the threshold line must not wipe the capture buffer each pixel."""
    tc = main_window._trigger_controller
    main_window._channel_configs[0] = ChannelConfig(vertical_span_v=1.0, screen_offset=0.5)
    main_window.trigger_control.update_channels([("Ch 0", 0)])
    main_window.trigger_control.trigger_mode_repeated.setChecked(True)
    main_window.trigger_control._on_config_changed()
    _build_history(tc)
    assert tc._history_length == 200

    # Simulate several drag updates of the threshold line.
    for norm_pos in (0.55, 0.6, 0.65):
        main_window.scope.threshold_line.setValue(norm_pos)  # emits sigPositionChanged
    assert tc._history_length == 200  # never reset by the drag
    # And the configured threshold followed the line.
    assert tc.threshold == pytest.approx(norm_to_volts(0.65, 1.0, 0.5))
