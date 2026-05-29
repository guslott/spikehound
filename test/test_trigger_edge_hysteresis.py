"""Phase 3: trigger edge selection (rising / falling / either) and hysteresis
(noise-reject band) in the TriggerController, plus the widget wiring.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from gui.trigger_controller import TriggerController
from gui.trigger_control_widget import TriggerControlWidget


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _ctrl(edge: str = "rising", hysteresis: float = 0.0) -> TriggerController:
    ctrl = TriggerController()
    ctrl.configure(
        mode="repeated", threshold=0.5, pre_seconds=0.0, window_sec=0.010,
        channel_id=0, edge=edge, hysteresis=hysteresis,
    )
    ctrl.update_sample_rate(10_000.0)
    return ctrl


# --- edge selection ----------------------------------------------------------

def test_rising_edge_only_detects_upward_crossing() -> None:
    ctrl = _ctrl(edge="rising")
    up = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)     # rises at idx 2
    down = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)   # falls at idx 2
    assert ctrl.detect_crossing(up) == 2
    ctrl.reset_state()
    ctrl.update_baseline(np.array([1.0], dtype=np.float32))   # signal already high
    assert ctrl.detect_crossing(down) is None


def test_falling_edge_only_detects_downward_crossing() -> None:
    ctrl = _ctrl(edge="falling")
    down = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    assert ctrl.detect_crossing(down) == 2
    ctrl.reset_state()
    assert ctrl.detect_crossing(up) is None


def test_either_edge_detects_first_of_either() -> None:
    ctrl = _ctrl(edge="either")
    assert ctrl.detect_crossing(np.array([0.0, 1.0, 0.0], dtype=np.float32)) == 1  # rise first
    ctrl.reset_state()
    ctrl.update_baseline(np.array([1.0], dtype=np.float32))   # start high
    assert ctrl.detect_crossing(np.array([1.0, 0.0, 1.0], dtype=np.float32)) == 1  # fall first


# --- hysteresis (noise reject) ----------------------------------------------

def test_hysteresis_blocks_retrigger_until_signal_retreats() -> None:
    # threshold 0.5, band 0.3 -> must dip below 0.2 to re-arm a rising trigger.
    ctrl = _ctrl(edge="rising", hysteresis=0.3)

    # Clean rising crossing fires (primed from reset).
    assert ctrl.detect_crossing(np.array([0.0, 1.0], dtype=np.float32)) == 1

    # Signal jitters around the threshold but never dips below 0.2 -> no re-fire.
    jitter = np.array([0.45, 0.55, 0.48, 0.6, 0.49, 0.7], dtype=np.float32)
    assert ctrl.detect_crossing(jitter) is None

    # Once it dips below the re-arm level (0.2) and rises again, it fires.
    rearm = np.array([0.1, 0.6], dtype=np.float32)
    assert ctrl.detect_crossing(rearm) == 1


def test_zero_hysteresis_matches_simple_crossing() -> None:
    ctrl = _ctrl(edge="rising", hysteresis=0.0)
    # Every clean upward crossing fires (no re-arm band).
    assert ctrl.detect_crossing(np.array([0.0, 1.0], dtype=np.float32)) == 1
    assert ctrl.detect_crossing(np.array([0.0, 1.0], dtype=np.float32)) == 1


def test_update_baseline_reprimes_without_firing() -> None:
    ctrl = _ctrl(edge="rising", hysteresis=0.3)
    assert ctrl.detect_crossing(np.array([0.0, 1.0], dtype=np.float32)) == 1  # fires, un-primes
    # Baseline maintenance sees a deep dip -> re-primes, but must not fire.
    ctrl.update_baseline(np.array([0.0, 0.0], dtype=np.float32))
    # Next rising crossing now fires because we were re-primed.
    assert ctrl.detect_crossing(np.array([0.1, 0.9], dtype=np.float32)) == 1


# --- widget wiring -----------------------------------------------------------

def test_widget_forwards_edge_and_hysteresis() -> None:
    _app()
    ctrl = TriggerController()
    widget = TriggerControlWidget(ctrl)
    widget.update_channels([("Ch 0", 0)])
    widget.trigger_mode_repeated.setChecked(True)

    idx = widget.edge_combo.findData("falling")
    widget.edge_combo.setCurrentIndex(idx)
    widget.hysteresis_spin.setValue(0.25)

    assert ctrl.edge == "falling"
    assert ctrl.hysteresis == 0.25


def test_edge_change_is_structural_and_resets_history() -> None:
    _app()
    ctrl = TriggerController()
    widget = TriggerControlWidget(ctrl)
    widget.update_channels([("Ch 0", 0)])
    widget.trigger_mode_repeated.setChecked(True)
    widget._on_config_changed()
    ctrl.update_sample_rate(10_000.0)
    ctrl.push_samples(np.ones(200, dtype=np.float32), 10_000.0, 1.0)
    assert ctrl._history_length == 200

    idx = widget.edge_combo.findData("either")
    widget.edge_combo.setCurrentIndex(idx)  # structural -> reset
    assert ctrl._history_length == 0
