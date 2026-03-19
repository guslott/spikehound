from __future__ import annotations

import csv
import queue
import time
from collections import deque

import numpy as np
from PySide6 import QtCore, QtWidgets

from analysis.settings import AnalysisSettingsStore
from analysis import AnalysisBatch
import gui.analysis_tab as analysis_tab_module
from gui.analysis_tab import AnalysisTab
from shared.models import Chunk
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


def _make_chunk() -> Chunk:
    samples = np.zeros((1, 20), dtype=np.float32)
    return Chunk(
        samples=samples,
        start_time=0.0,
        dt=0.001,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )


def _make_event(event_id: int = 1, *, samples: np.ndarray | None = None) -> AnalysisEvent:
    waveform = (
        np.array([0, 0, 0, 0, -1, 2, 0, 0, 0, 0], dtype=np.float32)
        if samples is None
        else np.asarray(samples, dtype=np.float32)
    )
    return AnalysisEvent(
        id=event_id,
        channelId=0,
        thresholdValue=0.5,
        crossingIndex=10,
        crossingTimeSec=0.01,
        firstSampleTimeSec=0.0,
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
    widget = AnalysisTab("ch0", 1000.0, controller=controller)
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
        widget._sta_pending_events[41] = (event_1, 1)

        widget._apply_analysis_update(widget._build_analysis_update((event_2,)), 0.0, 1.0, 0)
        widget._apply_analysis_update(widget._build_analysis_update((event_3,)), 0.0, 1.0, 0)

        assert [record["event_id"] for record in widget._metric_events] == [42, 43]
        assert sorted(widget._event_details) == [42, 43]
        assert 41 not in widget._event_cluster_labels
        assert 41 not in widget._sta_pending_events
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


def test_sta_source_cluster_filters_events_by_current_class_membership() -> None:
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
        widget._sta_windows.clear()

        task = analysis_tab_module.StaTask(
            events=(event_1, event_2),
            target_channel_id=0,
            channel_index=0,
            window_ms=50.0,
        )
        widget._sta_handle_task(task)

        assert len(widget._sta_windows) == 1
    finally:
        widget.close()
        _app().processEvents()
