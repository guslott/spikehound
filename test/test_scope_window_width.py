from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtWidgets

from gui.main_window import MainWindow
from gui.analysis_tab import AnalysisTab
from core.runtime import SpikeHoundRuntime
from gui.trigger_control_widget import TriggerControlWidget
from gui.trigger_controller import TriggerController
from shared.models import ChunkPointer
from shared.types import AnalysisEvent
from shared.ring_buffer import SharedRingBuffer


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


def test_trigger_window_combo_includes_005_seconds() -> None:
    _app()
    widget = TriggerControlWidget(TriggerController())
    values = [float(widget.window_combo.itemData(i)) for i in range(widget.window_combo.count())]
    labels = [widget.window_combo.itemText(i) for i in range(widget.window_combo.count())]

    assert 0.05 in values
    assert labels[values.index(0.05)] == "0.05"


def test_analysis_tab_accepts_005_second_scope_width() -> None:
    _app()
    widget = AnalysisTab("Channel 1", 20_000.0)

    widget.update_scale(0.05, 1.0)

    assert widget._scope_window_sec == 0.05


def test_main_window_window_combo_updates_scope_range_and_trigger_samples(main_window, monkeypatch) -> None:
    calls: list[float] = []
    monkeypatch.setattr(main_window._controller, "update_window_span", lambda value: calls.append(float(value)))
    main_window._trigger_controller.update_sample_rate(20_000.0)

    idx = main_window.trigger_control.window_combo.findData(5.0)
    assert idx >= 0

    main_window.trigger_control.window_combo.setCurrentIndex(idx)

    x_range = main_window.scope.plot_widget.getPlotItem().viewRange()[0]
    assert main_window._window_combo_user_set is True
    assert main_window._current_window_sec == pytest.approx(5.0)
    assert main_window._trigger_controller.window_sec == pytest.approx(5.0)
    assert main_window._trigger_controller._window_samples == 100_000
    assert calls[-1] == pytest.approx(5.0)
    assert x_range[0] == pytest.approx(0.0)
    assert x_range[1] == pytest.approx(5.0)


def test_main_window_pretrigger_combo_recomputes_samples_at_fixed_rate(main_window) -> None:
    main_window._trigger_controller.update_sample_rate(10_000.0)

    idx = main_window.trigger_control.pretrigger_combo.findData(0.02)
    assert idx >= 0

    main_window.trigger_control.pretrigger_combo.setCurrentIndex(idx)

    assert main_window._trigger_controller.pre_seconds == pytest.approx(0.02)
    assert main_window._trigger_controller._pre_samples == 200


def test_trigger_controller_push_samples_refreshes_timing_without_sample_rate_change() -> None:
    controller = TriggerController()
    controller.configure(
        mode="repeated",
        threshold=0.5,
        pre_seconds=0.01,
        window_sec=0.1,
        channel_id=0,
    )
    controller.update_sample_rate(1_000.0)

    controller.configure(
        mode="repeated",
        threshold=0.5,
        pre_seconds=0.02,
        window_sec=0.2,
        channel_id=0,
        preserve_display_on_reset=True,
    )
    controller.push_samples(np.zeros((32, 1), dtype=np.float32), 1_000.0, 0.2)

    assert controller._pre_samples == 20
    assert controller._window_samples == 200


def test_trigger_mode_dispatcher_tick_keeps_selected_window_when_queue_batch_is_longer(
) -> None:
    sample_rate = 10.0
    buffer = SharedRingBuffer((1, 64), dtype=np.float32)
    samples = np.arange(50, dtype=np.float32).reshape(1, -1)
    start = buffer.write(samples)
    pointer = ChunkPointer(start_index=start, length=50, render_time=0.0, seq=0, start_sample=0)

    seen: dict[str, float] = {}

    def _capture_trigger_mode(
        data,
        times_arr,
        sr,
        window_sec,
        channel_ids,
        now,
        *,
        trigger_mode,
        trigger_channel_id,
        pretrigger_line,
    ) -> None:
        seen["window_sec"] = float(window_sec)
        plot_manager.sample_rate = float(sr)
        plot_manager.window_sec = float(window_sec)

    def _stream_should_not_run(*args, **kwargs) -> None:
        raise AssertionError("trigger-mode tick should not fall back to streaming")

    plot_manager = SimpleNamespace(
        renderers={0: object()},
        process_streaming=_stream_should_not_run,
        process_trigger_mode=_capture_trigger_mode,
        sample_rate=0.0,
        window_sec=0.0,
        chunk_rate=0.0,
        chunk_mean_samples=0.0,
        actual_plot_refresh_hz=0.0,
    )
    state = SimpleNamespace(
        _trigger_controller=SimpleNamespace(mode="repeated", channel_id=0),
        _drain_visualization_queue=lambda: [pointer],
        _controller=SimpleNamespace(viz_buffer=lambda: buffer),
        _maybe_update_analysis_sample_rate=lambda sr: None,
        _register_chunk=lambda data: None,
        _plot_manager=plot_manager,
        scope=SimpleNamespace(pretrigger_line=None),
        runtime=SimpleNamespace(update_metrics=lambda **kwargs: None),
        _update_status=lambda viz_depth: None,
        _current_sample_rate=0.0,
        _current_window_sec=1.0,
        _chunk_rate=0.0,
        _chunk_mean_samples=0.0,
        _channel_ids_current=[0],
        _logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    MainWindow._on_dispatcher_tick(state, {"status": {"sample_rate": sample_rate, "window_sec": 1.0}})

    assert seen["window_sec"] == pytest.approx(1.0)


def test_sta_process_event_aligns_to_detected_crossing_offset() -> None:
    _app()
    widget = AnalysisTab("Channel 1", 1_000.0)
    event = AnalysisEvent(
        id=1,
        channelId=0,
        thresholdValue=0.5,
        crossingIndex=500,
        crossingTimeSec=0.5,
        firstSampleTimeSec=0.498,
        sampleRateHz=1_000.0,
        windowMs=10.0,
        preMs=2.0,
        postMs=8.0,
        samples=np.zeros(10, dtype=np.float32),
    )
    window = np.array([0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    controller = SimpleNamespace(
        collect_trigger_window=lambda *args, **kwargs: (window.copy(), 0, 0),
    )

    status = widget._sta_process_event(
        controller,
        target_channel_id=0,
        channel_info=None,
        event=event,
        window_ms=50.0,
    )

    assert status == "added"
    assert widget._sta_windows
    normalized = widget._sta_windows[-1]
    assert normalized[2] == pytest.approx(0.0)
    assert normalized[5] == pytest.approx(-4.0)
