"""Phase 4 (4A / F7): _render_trigger_display must not re-run the expensive
rebuild (setXRange + per-channel float32 copies + curve update) every frame
while a *completed* capture is merely being held. It should redraw only when
the captured data object or the window changes.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from gui.scope_widget import ScopeWidget
from gui.plot_manager import PlotManager
from gui.trigger_controller import TriggerController
from gui.types import ChannelConfig


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _completed_capture() -> tuple[PlotManager, ScopeWidget, TriggerController]:
    scope = ScopeWidget()
    tc = TriggerController()
    tc.configure(
        mode="repeated", threshold=0.5, pre_seconds=0.002, window_sec=0.010, channel_id=0,
    )
    tc.update_sample_rate(10_000.0)  # window = 100 samples, pre = 20
    pm = PlotManager(scope.plot_widget, tc)
    pm.ensure_renderers_for_ids([0], ["Ch 0"], {0: ChannelConfig()})

    chunk = np.zeros((200, 1), dtype=np.float32)
    chunk[20:, 0] = 1.0
    tc.push_samples(chunk, 10_000.0, 0.010)
    tc.detect_crossing(chunk[:, 0])
    tc.start_capture(0, 20)
    assert tc.advance_capture() is True          # full window captured
    assert tc.display_data is not None
    return pm, scope, tc


def test_held_capture_is_not_rerendered_each_frame() -> None:
    _app()
    pm, scope, tc = _completed_capture()

    calls: list[int] = []
    pm.renderers[0].update_data = lambda *a, **k: calls.append(1)  # type: ignore[assignment]

    pm._render_trigger_display([0], 0.010, scope.pretrigger_line)
    assert len(calls) == 1                         # first frame does the work

    pm._render_trigger_display([0], 0.010, scope.pretrigger_line)
    pm._render_trigger_display([0], 0.010, scope.pretrigger_line)
    assert len(calls) == 1                         # held, unchanged -> skipped


def test_window_change_forces_rerender() -> None:
    _app()
    pm, scope, tc = _completed_capture()

    calls: list[int] = []
    pm.renderers[0].update_data = lambda *a, **k: calls.append(1)  # type: ignore[assignment]

    pm._render_trigger_display([0], 0.010, scope.pretrigger_line)
    pm._render_trigger_display([0], 0.020, scope.pretrigger_line)  # window changed
    assert len(calls) == 2


def test_new_capture_object_forces_rerender() -> None:
    _app()
    pm, scope, tc = _completed_capture()

    calls: list[int] = []
    pm.renderers[0].update_data = lambda *a, **k: calls.append(1)  # type: ignore[assignment]
    pm._render_trigger_display([0], 0.010, scope.pretrigger_line)
    assert len(calls) == 1

    # A fresh capture produces a new display array (distinct object) -> re-render.
    tc.reset_state()
    chunk = np.zeros((200, 1), dtype=np.float32)
    chunk[20:, 0] = 1.0
    tc.push_samples(chunk, 10_000.0, 0.010)
    tc.detect_crossing(chunk[:, 0])
    tc.start_capture(0, 20)
    assert tc.advance_capture() is True
    pm._render_trigger_display([0], 0.010, scope.pretrigger_line)
    assert len(calls) == 2
