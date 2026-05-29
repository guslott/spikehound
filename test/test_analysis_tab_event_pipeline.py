from __future__ import annotations

import csv
import queue
import time
from collections import deque
from concurrent.futures import Future

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from analysis.settings import AnalysisSettingsStore
from analysis import AnalysisBatch
import gui.analysis_tab as analysis_tab_module
from gui.analysis_tab import AnalysisTab
from shared.models import ChannelInfo, Chunk
from shared.types import AnalysisEvent


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _DummyController:
    def __init__(self) -> None:
        self.analysis_settings_store = AnalysisSettingsStore()

    def active_channels(self):
        return []

    def collect_trigger_window(self, event, *, target_channel_id: int, window_ms: float):
        del event, target_channel_id, window_ms
        return np.ones(5, dtype=np.float32), 0, 0


def _make_chunk(
    *,
    samples: np.ndarray | None = None,
    start_time: float = 0.0,
    start_sample: int = 0,
    seq: int = 0,
) -> Chunk:
    chunk_samples = (
        np.zeros((1, 20), dtype=np.float32)
        if samples is None
        else np.asarray(samples, dtype=np.float32)
    )
    return Chunk(
        samples=chunk_samples,
        start_time=start_time,
        dt=0.001,
        seq=seq,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": start_sample},
    )


def _make_event(
    event_id: int = 1,
    *,
    samples: np.ndarray | None = None,
    crossing_index: int = 10,
    crossing_time_sec: float = 0.01,
    first_sample_time_sec: float = 0.0,
) -> AnalysisEvent:
    waveform = (
        np.array([0, 0, 0, 0, -1, 2, 0, 0, 0, 0], dtype=np.float32)
        if samples is None
        else np.asarray(samples, dtype=np.float32)
    )
    return AnalysisEvent(
        id=event_id,
        channelId=0,
        thresholdValue=0.5,
        crossingIndex=crossing_index,
        crossingTimeSec=crossing_time_sec,
        firstSampleTimeSec=first_sample_time_sec,
        sampleRateHz=1000.0,
        windowMs=10.0,
        preMs=5.0,
        postMs=5.0,
        samples=waveform,
        properties={
            "envelope": float(np.max(waveform) - np.min(waveform)),
            "peak_freq_hz": 100.0,
            "event_width_ms": 1.0,
            "min_to_max_width_ms": 1.0,
        },
        intervalSinceLastSec=float("nan"),
    )


def _make_tab(controller: _DummyController) -> AnalysisTab:
    _app()
    widget = AnalysisTab("ch0", 1000.0, controller=controller, channel_id=0)
    widget.set_sta_channels(
        (
            ChannelInfo(id=0, name="ch0"),
            ChannelInfo(id=1, name="ch1"),
        )
    )
    widget._update_timer.stop()
    return widget


def _wait_for(predicate, *, timeout: float = 2.0) -> bool:
    app = _app()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def _set_x_metric(widget: AnalysisTab, label_fragment: str) -> None:
    for index in range(widget.metric_xaxis_combo.count()):
        if label_fragment.lower() in widget.metric_xaxis_combo.itemText(index).lower():
            widget.metric_xaxis_combo.setCurrentIndex(index)
            _app().processEvents()
            return
    raise AssertionError(f"Could not find X-axis metric containing {label_fragment!r}")


def test_analysis_tab_processes_queue_events_once_and_delivers_async_results() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        q: "queue.Queue[AnalysisBatch]" = queue.Queue()
        widget.set_analysis_queue(q)

        event = _make_event(1)
        q.put(AnalysisBatch(chunk=_make_chunk(), events=(event,)))

        widget._last_window_start = 0.0
        widget._last_window_width = 1.0
        widget._window_start_index = 0
        widget._process_analysis_queue()

        assert _wait_for(lambda: len(widget._metric_events) == 1 and not widget._analysis_futures)
        assert [record["event_id"] for record in widget._metric_events] == [1]
        assert len(widget._event_overlays) == 1
    finally:
        widget.close()
        _app().processEvents()


class _FakeWorker:
    """Minimal stand-in so _notify_threshold_change reaches its body."""

    def __init__(self) -> None:
        self.configure_calls = 0

    def configure_threshold(self, *args, **kwargs) -> None:
        self.configure_calls += 1

    def update_sample_rate(self, sample_rate: float) -> None:
        del sample_rate


def test_threshold_value_nudge_preserves_history_but_mode_change_clears() -> None:
    """Finding 7.2: nudging a threshold VALUE (spinbox tick / line drag) must
    reconfigure detection going forward WITHOUT wiping accumulated events,
    clusters, or STA; only a detection-MODE change (or window-width / new
    worker) clears history.
    """
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        worker = _FakeWorker()
        widget.set_worker(worker)  # fresh worker -> clears (nothing accumulated yet)

        # Accumulate one event into the metric history.
        q: "queue.Queue[AnalysisBatch]" = queue.Queue()
        widget.set_analysis_queue(q)
        q.put(AnalysisBatch(chunk=_make_chunk(), events=(_make_event(1),)))
        widget._last_window_start = 0.0
        widget._last_window_width = 1.0
        widget._window_start_index = 0
        widget._process_analysis_queue()
        assert _wait_for(lambda: len(widget._metric_events) == 1 and not widget._analysis_futures)
        assert sorted(widget._event_details) == [1]

        # 1) A threshold VALUE nudge must NOT wipe history, but MUST reconfigure.
        configured_before = worker.configure_calls
        widget.threshold1_spin.setValue(0.25)  # fires valueChanged -> _notify_threshold_change()
        _app().processEvents()
        assert worker.configure_calls > configured_before, "value nudge should reconfigure the worker"
        assert len(widget._metric_events) == 1, "value nudge must NOT clear accumulated events (7.2)"
        assert sorted(widget._event_details) == [1], "value nudge must NOT clear event details (7.2)"

        # 2) A detection-MODE change (enabling auto-detect) SHOULD clear history.
        widget.auto_detect_check.setChecked(True)
        _app().processEvents()
        assert len(widget._metric_events) == 0, "mode change should clear accumulated events"
        assert len(widget._event_details) == 0, "mode change should clear event details"
    finally:
        widget.close()
        _app().processEvents()


def test_analysis_tab_requires_analysis_queue_for_live_event_processing() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        widget._last_window_start = 0.0
        widget._last_window_width = 1.0
        widget._window_start_index = 0
        widget._process_analysis_queue()

        assert len(widget._metric_events) == 0
        assert len(widget._event_overlays) == 0
    finally:
        widget.close()
        _app().processEvents()


def test_analysis_tab_ignores_legacy_chunk_payloads() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        q: "queue.Queue[object]" = queue.Queue()
        widget.set_analysis_queue(q)

        legacy_chunk = _make_chunk()
        legacy_chunk.meta["analysis_events"] = (_make_event(99),)
        q.put(legacy_chunk)

        widget._last_window_start = 0.0
        widget._last_window_width = 1.0
        widget._window_start_index = 0
        widget._process_analysis_queue()

        assert len(widget._metric_events) == 0
        assert len(widget._event_overlays) == 0
    finally:
        widget.close()
        _app().processEvents()


def test_async_analysis_update_uses_current_window_for_overlay_position() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        event = _make_event(
            101,
            crossing_index=550,
            crossing_time_sec=0.55,
            first_sample_time_sec=0.545,
        )
        future: Future = Future()
        future.set_result(widget._build_analysis_update((event,)))

        widget._window_start_time = 0.5
        widget._last_window_start = 0.5
        widget._last_window_width = 0.1
        widget._window_start_index = 500

        widget._on_analysis_update_ready(future, 0.0, 1.0, 0)

        overlay_item = widget._event_overlays[0]["item"]
        x_data, _ = overlay_item.getData()
        assert x_data is not None
        assert float(x_data[0]) == pytest.approx(0.045, rel=1e-6)
    finally:
        widget.close()
        _app().processEvents()


def test_pause_snapshot_keeps_raw_trace_frozen_during_overlay_recolor() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        batch = AnalysisBatch(
            chunk=_make_chunk(samples=np.zeros((1, 20), dtype=np.float32)),
            events=(_make_event(102),),
        )
        widget._render_batch(batch)
        widget._on_pause_viz_toggled(True)

        raw_snapshot = widget._pause_raw_snapshot_curve
        assert raw_snapshot is not None
        _, paused_y = raw_snapshot.getData()
        assert paused_y is not None

        widget._cached_raw_times = np.arange(20, dtype=np.float32) * 0.001
        widget._cached_raw_samples = np.full(20, 7.0, dtype=np.float32)
        widget._refresh_overlay_colors()

        frozen_snapshot = widget._pause_raw_snapshot_curve
        assert frozen_snapshot is raw_snapshot
        _, after_y = frozen_snapshot.getData()
        assert after_y is not None
        np.testing.assert_allclose(after_y, paused_y)
    finally:
        widget.close()
        _app().processEvents()


def test_cluster_membership_is_stable_when_axes_change() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        event = _make_event(11)
        update = widget._build_analysis_update((event,))
        widget._apply_analysis_update(update, 0.0, 1.0, 0)
        widget._update_metric_points()

        widget._on_add_class_clicked()
        cluster = widget._clusters[0]
        assert cluster.x_metric == "min"
        assert cluster.y_metric == "max"

        roi = cluster.roi
        assert roi is not None
        roi.setPos((-1.5, 1.5))
        roi.setSize((2.0, 1.0))
        widget._on_cluster_roi_changed()

        assert widget._event_cluster_labels == {11: cluster.id}
        assert roi.isVisible()

        _set_x_metric(widget, "Envelope")

        assert widget._event_cluster_labels == {11: cluster.id}
        assert not roi.isVisible()
    finally:
        widget.close()
        _app().processEvents()


def test_new_events_use_cluster_metric_pair_even_when_current_axes_differ() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        first_event = _make_event(21)
        widget._apply_analysis_update(widget._build_analysis_update((first_event,)), 0.0, 1.0, 0)
        widget._update_metric_points()

        widget._on_add_class_clicked()
        cluster = widget._clusters[0]
        roi = cluster.roi
        assert roi is not None
        roi.setPos((-1.5, 1.5))
        roi.setSize((2.0, 1.0))
        widget._on_cluster_roi_changed()
        assert widget._event_cluster_labels == {21: cluster.id}

        _set_x_metric(widget, "Envelope")

        second_event = _make_event(22)
        widget._apply_analysis_update(widget._build_analysis_update((second_event,)), 0.0, 1.0, 0)
        widget._update_metric_points()

        assert widget._event_cluster_labels[22] == cluster.id
    finally:
        widget.close()
        _app().processEvents()


def test_visible_overlays_repaint_when_class_membership_changes() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        event = _make_event(31)
        widget._apply_analysis_update(widget._build_analysis_update((event,)), 0.0, 1.0, 0)
        widget._update_metric_points()

        overlay_item = widget._event_overlays[0]["item"]
        assert overlay_item.opts["pen"].color().getRgb()[:3] == (220, 0, 0)

        widget._on_add_class_clicked()
        cluster = widget._clusters[0]
        roi = cluster.roi
        assert roi is not None
        roi.setPos((-1.5, 1.5))
        roi.setSize((2.0, 1.0))
        widget._on_cluster_roi_changed()

        assert widget._event_cluster_labels[31] == cluster.id
        assert overlay_item.opts["pen"].color().getRgb()[:3] == cluster.color.getRgb()[:3]

        widget._on_remove_class_clicked()

        assert 31 not in widget._event_cluster_labels
        assert overlay_item.opts["pen"].color().getRgb()[:3] == (220, 0, 0)
    finally:
        widget.close()
        _app().processEvents()


def test_event_details_retention_matches_metric_history_cap() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        widget._metric_events = deque(maxlen=2)

        event_1 = _make_event(41)
        event_2 = _make_event(42)
        event_3 = _make_event(43)

        widget._apply_analysis_update(widget._build_analysis_update((event_1,)), 0.0, 1.0, 0)
        widget._event_cluster_labels[41] = 9
        widget._sta_pending_events[41] = event_1
        widget._sta_records[41] = analysis_tab_module.CorrelationRecord(
            event_id=41,
            crossing_time_sec=event_1.crossingTimeSec,
            channel_windows={},
        )

        widget._apply_analysis_update(widget._build_analysis_update((event_2,)), 0.0, 1.0, 0)
        widget._apply_analysis_update(widget._build_analysis_update((event_3,)), 0.0, 1.0, 0)

        assert [record["event_id"] for record in widget._metric_events] == [42, 43]
        assert sorted(widget._event_details) == [42, 43]
        assert 41 not in widget._event_cluster_labels
        assert 41 not in widget._sta_records
        assert 41 not in widget._sta_pending_events
    finally:
        widget.close()
        _app().processEvents()


def test_event_details_bounded_independently_of_metric_history(monkeypatch) -> None:
    """Finding 7.4: the heavy _event_details waveform store is capped on its own
    small budget, independent of the (much larger, cheap) metric history."""
    monkeypatch.setattr(analysis_tab_module, "EVENT_DETAIL_CAPACITY", 3)
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        # Large metric history so it is NOT the binding constraint.
        widget._metric_events = deque(maxlen=1000)

        for eid in range(1, 11):  # 10 events, detail cap 3
            widget._apply_analysis_update(
                widget._build_analysis_update((_make_event(eid),)), 0.0, 1.0, 0
            )

        # Waveform store bounded to the small cap, keeping the most recent.
        assert len(widget._event_details) == 3
        assert sorted(widget._event_details) == [8, 9, 10]
        # The cheap metric history is unaffected by the waveform cap.
        assert len(widget._metric_events) == 10
    finally:
        widget.close()
        _app().processEvents()


def test_export_class_filters_by_current_class_membership(monkeypatch, tmp_path) -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        event_1 = _make_event(51)
        event_2 = _make_event(52, samples=np.array([0, 0, 0, 0, -4, 5, 0, 0, 0, 0], dtype=np.float32))
        widget._apply_analysis_update(widget._build_analysis_update((event_1, event_2)), 0.0, 1.0, 0)
        widget._update_metric_points()

        widget._on_add_class_clicked()
        cluster = widget._clusters[0]
        roi = cluster.roi
        assert roi is not None
        roi.setPos((-1.5, 1.5))
        roi.setSize((2.0, 1.0))
        widget._on_cluster_roi_changed()

        export_index = widget.export_class_combo.findData(cluster.id)
        assert export_index >= 0
        widget.export_class_combo.setCurrentIndex(export_index)

        export_path = tmp_path / "classified.csv"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *args, **kwargs: (str(export_path), "CSV Files (*.csv)")),
        )
        info_messages: list[str] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            staticmethod(lambda *args, **kwargs: info_messages.append(str(args[2]))),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "critical",
            staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected export error"))),
        )

        widget._on_export_class_clicked()

        with export_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["Class ID"] == "1"
        assert rows[0]["Max (V)"] == "2.0"
        assert info_messages
    finally:
        widget.close()
        _app().processEvents()


def test_view_class_waveforms_uses_current_class_membership(monkeypatch) -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        event_1 = _make_event(61)
        event_2 = _make_event(62, samples=np.array([0, 0, 0, 0, -4, 5, 0, 0, 0, 0], dtype=np.float32))
        widget._apply_analysis_update(widget._build_analysis_update((event_1, event_2)), 0.0, 1.0, 0)
        widget._update_metric_points()

        widget._on_add_class_clicked()
        cluster = widget._clusters[0]
        roi = cluster.roi
        assert roi is not None
        roi.setPos((-1.5, 1.5))
        roi.setSize((2.0, 1.0))
        widget._on_cluster_roi_changed()

        item = widget._cluster_items[cluster.id]
        widget.class_list.setCurrentItem(item)

        captured_event_ids: list[int] = []

        class FakeWaveformLoader(QtCore.QObject):
            progress = QtCore.Signal(int)
            data_ready = QtCore.Signal(str, object, list, object)

            def __init__(self, event_ids, event_details, class_name, class_color, parent=None):
                super().__init__(parent)
                del event_details, class_name, class_color
                captured_event_ids.extend(event_ids)

            def start(self) -> None:
                return None

            def wait(self) -> None:
                return None

            def cancel(self) -> None:
                return None

        monkeypatch.setattr(analysis_tab_module, "WaveformLoader", FakeWaveformLoader)

        widget._on_view_class_waveforms_clicked()

        assert captured_event_ids == [61]
    finally:
        widget.close()
        _app().processEvents()


def test_waveform_correlation_collects_all_active_channels() -> None:
    class _WindowController(_DummyController):
        def collect_trigger_window(self, event, *, target_channel_id: int, window_ms: float):
            del event, window_ms
            return np.full(5, float(target_channel_id + 1), dtype=np.float32), 0, 0

    controller = _WindowController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        event = _make_event(70)
        task = analysis_tab_module.CorrelationTask(
            events=(event,),
            channel_ids=(0, 1),
            source_channel_id=0,
            window_ms=50.0,
            mode="waveform",
        )

        widget._sta_handle_task(task)
        widget._sta_enabled = True
        widget._refresh_sta_plot()

        assert sorted(widget._sta_records[70].channel_windows) == [0, 1]
        assert sorted(widget._sta_curve_items) == [0, 1]
    finally:
        widget.close()
        _app().processEvents()


def test_waveform_correlation_plot_shows_recent_ghost_traces_and_summary_title() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        widget._sta_enabled = True
        widget._sta_mode = "waveform"
        widget._sta_source_cluster_id = None
        for event_id in range(1, 13):
            widget._sta_records[event_id] = analysis_tab_module.CorrelationRecord(
                event_id=event_id,
                crossing_time_sec=0.1 * event_id,
                channel_windows={
                    0: np.full(5, float(event_id), dtype=np.float32),
                    1: np.full(5, float(event_id + 100), dtype=np.float32),
                },
            )

        widget._refresh_sta_plot()

        assert sorted(widget._sta_curve_items) == [0, 1]
        assert len(widget._sta_ghost_curve_items) == 20
        title = widget.sta_plot.getPlotItem().titleLabel.text
        assert "Waveform average" in title
        assert "source: All events" in title
        assert "n=12" in title
    finally:
        widget.close()
        _app().processEvents()


def test_waveform_correlation_inspector_limits_to_recent_contributing_traces(monkeypatch) -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        widget._sta_enabled = True
        widget._sta_mode = "waveform"
        for event_id in range(1, 13):
            widget._sta_records[event_id] = analysis_tab_module.CorrelationRecord(
                event_id=event_id,
                crossing_time_sec=0.1 * event_id,
                channel_windows={
                    0: np.full(5, float(event_id), dtype=np.float32),
                    1: np.full(5, float(event_id + 100), dtype=np.float32),
                },
            )

        captured: dict[str, object] = {}

        class FakeDialog:
            def __init__(
                self,
                parent,
                class_name,
                waveforms,
                color,
                median_waveform=None,
                *,
                show_median=True,
                background_color=None,
                y_label="Amplitude",
                y_units="V",
                summary_waveforms=None,
                count_override=None,
            ) -> None:
                del parent, color, median_waveform, show_median, background_color, y_label, y_units
                captured["class_name"] = class_name
                captured["waveforms"] = list(waveforms)
                captured["summary_waveforms"] = list(summary_waveforms or [])
                captured["count_override"] = count_override

            def exec(self) -> int:
                return 0

        monkeypatch.setattr(analysis_tab_module, "ClusterWaveformDialog", FakeDialog)
        widget._refresh_sta_plot()

        widget._on_sta_view_waveforms_clicked()

        assert captured["class_name"] == "Correlation inspector – ch0"
        assert len(captured["summary_waveforms"]) == 2
        assert len(captured["waveforms"]) == 20
        assert captured["count_override"] == 12
    finally:
        widget.close()
        _app().processEvents()


def test_waveform_correlation_source_cluster_filters_events_by_current_class_membership() -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        event_1 = _make_event(71)
        event_2 = _make_event(72, samples=np.array([0, 0, 0, 0, -4, 5, 0, 0, 0, 0], dtype=np.float32))
        widget._apply_analysis_update(widget._build_analysis_update((event_1, event_2)), 0.0, 1.0, 0)
        widget._update_metric_points()

        widget._on_add_class_clicked()
        cluster = widget._clusters[0]
        roi = cluster.roi
        assert roi is not None
        roi.setPos((-1.5, 1.5))
        roi.setSize((2.0, 1.0))
        widget._on_cluster_roi_changed()
        assert widget._event_cluster_labels[71] == cluster.id

        widget._sta_source_cluster_id = cluster.id
        task = analysis_tab_module.CorrelationTask(
            events=(event_1, event_2),
            channel_ids=(0,),
            source_channel_id=0,
            window_ms=50.0,
            mode="waveform",
        )
        widget._sta_handle_task(task)

        assert sorted(widget._sta_records) == [71, 72]
        assert [record.event_id for record in widget._eligible_sta_records()] == [71]
    finally:
        widget.close()
        _app().processEvents()


def test_correlation_history_is_bounded(monkeypatch) -> None:
    controller = _DummyController()
    widget = _make_tab(controller)
    try:
        if widget._analysis_executor is not None:
            widget._analysis_executor.shutdown(wait=True)
            widget._analysis_executor = None

        monkeypatch.setattr(analysis_tab_module, "CORRELATION_HISTORY_CAPACITY", 2)

        task = analysis_tab_module.CorrelationTask(
            events=(_make_event(81), _make_event(82), _make_event(83)),
            channel_ids=(),
            source_channel_id=0,
            window_ms=20.0,
            mode="autocorrelogram",
        )
        widget._sta_handle_task(task)

        assert list(widget._sta_records) == [82, 83]
    finally:
        widget.close()
        _app().processEvents()
