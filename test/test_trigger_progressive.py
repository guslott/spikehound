"""Progressive ("sweep") trigger rendering: the captured window must draw
incrementally as samples arrive after the crossing, instead of waiting for the
whole window to buffer before anything appears.
"""
from __future__ import annotations

import time

import numpy as np

from gui.trigger_controller import TriggerController


def _armed_repeated() -> TriggerController:
    ctrl = TriggerController()
    ctrl.configure(
        mode="repeated", threshold=0.5, pre_seconds=0.002, window_sec=0.010, channel_id=0,
    )
    ctrl.update_sample_rate(10_000.0)  # window = 100 samples, pre = 20 samples
    return ctrl


def test_advance_capture_renders_partial_then_completes() -> None:
    ctrl = _armed_repeated()

    # First chunk: 30 samples, rising crossing at index 20.
    c1 = np.zeros(30, dtype=np.float32)
    c1[20:] = 1.0
    ctrl.push_samples(c1, 10_000.0, 0.010)
    assert ctrl.detect_crossing(c1) == 20
    ctrl.start_capture(0, 20)  # capture_start=0, capture_end=100

    # Only 30 of 100 window samples exist -> partial draw, not complete.
    assert ctrl.advance_capture() is False
    assert ctrl.display_data is not None
    assert ctrl.display_data.shape[0] == 30          # grows from the trigger point
    assert ctrl.display_data.shape[0] < ctrl._window_samples

    # A second partial as more data streams in.
    ctrl.push_samples(np.ones(40, dtype=np.float32), 10_000.0, 0.010)  # total 70
    assert ctrl.advance_capture() is False
    assert ctrl.display_data.shape[0] == 70

    # Window fills -> complete, full width, capture latched (pointers cleared).
    ctrl.push_samples(np.ones(30, dtype=np.float32), 10_000.0, 0.010)  # total 100
    assert ctrl.advance_capture() is True
    assert ctrl.display_data.shape[0] == 100
    assert ctrl._capture_start_abs is None
    assert ctrl._hold_until > 0.0


def test_partial_capture_preserves_pretrigger_alignment() -> None:
    ctrl = _armed_repeated()
    c1 = np.zeros(40, dtype=np.float32)
    c1[20:] = 1.0
    ctrl.push_samples(c1, 10_000.0, 0.010)
    ctrl.detect_crossing(c1)
    ctrl.start_capture(0, 20)
    ctrl.advance_capture()
    # The crossing still sits pre_samples (20) into the partial trace.
    assert ctrl.display_pre_samples == 20
    assert ctrl.display_data[20] >= 0.5
    assert ctrl.display_data[19] < 0.5


def test_in_progress_sweep_not_cleared_by_hold_check() -> None:
    """check_hold_expiry must not wipe a still-filling sweep (repeated mode)."""
    ctrl = _armed_repeated()
    c1 = np.zeros(30, dtype=np.float32)
    c1[20:] = 1.0
    ctrl.push_samples(c1, 10_000.0, 0.010)
    ctrl.detect_crossing(c1)
    ctrl.start_capture(0, 20)
    ctrl.advance_capture()  # partial; hold_until still 0.0, capture in progress

    cleared = ctrl.check_hold_expiry(time.perf_counter(), is_single_mode=False)
    assert cleared is False
    assert ctrl.display_data is not None
    assert ctrl._capture_start_abs is not None  # still sweeping


def test_no_capture_means_advance_is_noop() -> None:
    ctrl = _armed_repeated()
    ctrl.push_samples(np.zeros(50, dtype=np.float32), 10_000.0, 0.010)
    assert ctrl.advance_capture() is False
    assert ctrl.display_data is None
