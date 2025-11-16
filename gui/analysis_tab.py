from __future__ import annotations

import logging
import queue
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Sequence

import numpy as np

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui

from analysis.analysis_worker import _peak_frequency_sinc
from analysis.models import AnalysisBatch
from core.models import Chunk, EndOfStream
from shared.event_buffer import AnalysisEvents
from shared.types import Event


CLUSTER_COLORS: list[QtGui.QColor] = [
    QtGui.QColor(34, 139, 34),    # green
    QtGui.QColor(255, 140, 0),    # orange
    QtGui.QColor(148, 0, 211),    # purple
    QtGui.QColor(255, 215, 0),    # gold
    QtGui.QColor(199, 21, 133),   # magenta
]
UNCLASSIFIED_COLOR = QtGui.QColor(220, 0, 0)
MAX_VISIBLE_METRIC_EVENTS = 3000
METRIC_TIME_WINDOW_SEC = 60.0
STA_TRACE_PEN = pg.mkPen(90, 90, 90, 110)

def _baseline(samples: np.ndarray, pre_samples: int) -> float:
    arr = np.asarray(samples, dtype=np.float32)
    if pre_samples <= 0 or arr.size == 0:
        return 0.0
    return float(np.median(arr[: min(pre_samples, arr.size)]))


def _blackman(n: int) -> np.ndarray:
    return np.blackman(max(1, n))


def _energy_density(x: np.ndarray, sr: float) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0 or sr <= 0:
        return 0.0
    base = _baseline(arr, max(1, int(0.1 * arr.size)))
    x_detrend = arr - base
    window = _blackman(arr.size)
    weighted = x_detrend * window
    energy = np.sum(weighted * weighted, dtype=np.float64)
    window_sec = max(1e-12, arr.size / float(sr))
    return float(energy / window_sec)


def _min_max(x: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.max(arr)), float(np.min(arr))


if TYPE_CHECKING:
    from analysis.analysis_worker import AnalysisWorker
    from analysis.settings import AnalysisSettingsStore
    from core.controller import PipelineController
    from shared.event_buffer import EventRingBuffer
    from daq.base_source import ChannelInfo


class ClusterRectROI(pg.ROI):
    """Rectangular ROI used for metric clusters."""

    def __init__(self, pos, size, pen=None, **kwargs):
        super().__init__(pos, size, movable=False, rotatable=False, **kwargs)
        self.addScaleHandle((0.0, 0.5), (1.0, 0.5))
        self.addScaleHandle((1.0, 0.5), (0.0, 0.5))
        self.addScaleHandle((0.5, 0.0), (0.5, 1.0))
        self.addScaleHandle((0.5, 1.0), (0.5, 0.0))
        for handle in self.handles:
            item = handle.get("item")
            if item is not None:
                try:
                    item.setSize(10)
                except AttributeError:
                    pass
        if pen is None:
            pen = pg.mkPen(CLUSTER_COLORS[0])
        self.setPen(pen)
        self._dragging_with_shift = False

    def mouseDragEvent(self, ev):
        if ev.isStart():
            self._dragging_with_shift = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
            if self._dragging_with_shift:
                ev.accept()
                return
            return super().mouseDragEvent(ev)
        if self._dragging_with_shift:
            last = ev.lastPos()
            if last is None:
                ev.accept()
                return
            delta = pg.Point(ev.pos()) - pg.Point(last)
            self.translate(delta, snap=False)
            ev.accept()
            if ev.isFinish():
                self._dragging_with_shift = False
            return
        return super().mouseDragEvent(ev)


@dataclass
class MetricCluster:
    id: int
    name: str
    color: QtGui.QColor
    roi: pg.RectROI | None = None


@dataclass
class OverlayPayload:
    """Qt-free representation of a raw spike overlay."""

    event_id: int | None
    times: np.ndarray
    samples: np.ndarray
    last_time: float
    first_index: int | None
    sr: float
    pre_samples: int
    baseline: float
    peak_idx: int
    peak_time: float
    metrics: dict[str, float] | None
    metric_time: float


@dataclass
class StaTask:
    """Data packet describing which events to use for STA processing."""

    events: tuple[Event, ...]
    target_channel_id: int
    channel_index: int | None
    window_ms: float


@dataclass
class AnalysisUpdate:
    """Result bundle produced by preprocessing an analysis batch."""

    overlays: list[OverlayPayload]
    sta_windows: list[np.ndarray] | None = None
    sta_task: StaTask | None = None
    last_event_id: int | None = None


class AnalysisTab(QtWidgets.QWidget):
    """Simple analysis view with a top-half plot for a single channel."""

    def __init__(
        self,
        channel_name: str,
        sample_rate: float,
        parent: Optional[QtWidgets.QWidget] = None,
        controller: Optional["PipelineController"] = None,
    ) -> None:
        super().__init__(parent)
        self.channel_name = channel_name
        self.sample_rate = float(sample_rate)
        self._analysis_queue: Optional["queue.Queue"] = None
        self._dt = 1.0 / self.sample_rate if self.sample_rate > 0 else 1e-3
        self._buffer_span_sec = 10.0
        self._init_buffer()
        self._controller = controller
        self._analysis_settings: Optional["AnalysisSettingsStore"] = None
        self._event_buffer: Optional["EventRingBuffer"] = None
        self._analysis_events: Optional[AnalysisEvents] = None
        self._analysis_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        self._analysis_futures: set[Future] = set()
        self._worker: Optional["AnalysisWorker"] = None
        self._event_overlays: list[dict[str, object]] = []
        self._overlay_pool: list[pg.PlotCurveItem] = []
        self._overlay_pen = pg.mkPen(UNCLASSIFIED_COLOR, width=2)
        self._overlay_capacity: int = 64
        self._latest_sample_time: Optional[float] = None
        self._window_start_time: Optional[float] = None
        self._channel_index: Optional[int] = None
        self._latest_sample_index: Optional[int] = None
        self._window_start_index: Optional[int] = None
        if controller is not None:
            self._analysis_settings = getattr(controller, "analysis_settings_store", None)
            self._event_buffer = getattr(controller, "event_buffer", None)
            if self._event_buffer is not None:
                self._analysis_events = AnalysisEvents(self._event_buffer)
        self._event_window_ms = self._initial_event_window_ms()
        self._t0_event: Optional[float] = None
        self._metric_events: list[dict[str, float | int]] = []
        self._max_metric_events = 10_000
        self._metrics_dirty: bool = False
        self._last_event_id: Optional[int] = None
        self._viz_paused = False
        self._cached_raw_times: Optional[np.ndarray] = None
        self._cached_raw_samples: Optional[np.ndarray] = None
        self._last_window_start: float = 0.0
        self._last_window_width: float = 0.5
        self._fallback_start_sample: int = 0
        self._next_event_id: int = 0
        self._event_details: dict[int, dict[str, object]] = {}
        self._event_cluster_labels: dict[int, int] = {}
        self._clusters: list[MetricCluster] = []
        self._cluster_id_counter: int = 0
        self._cluster_items: dict[int, QtWidgets.QListWidgetItem] = {}
        self._selected_cluster_id: int | None = None
        self._sta_enabled: bool = False
        self._sta_windows: list[np.ndarray] = []
        self._sta_max_windows: int = 500
        self._sta_update_interval_ms: int = 100
        self._sta_dirty: bool = False
        self._sta_time_axis: np.ndarray | None = None
        self._sta_window_ms: float = 50.0
        self._sta_source_cluster_id: int | None = None
        self._sta_target_channel_id: int | None = None
        self._sta_pending_events: dict[int, tuple[Event, int]] = {}
        self._sta_retry_limit: int = 10

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel(self._title_text())
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.title_label)

        self.raw_row_widget = QtWidgets.QWidget(self)
        self.raw_row_layout = QtWidgets.QHBoxLayout(self.raw_row_widget)
        self.raw_row_layout.setContentsMargins(0, 0, 0, 0)
        self.raw_row_layout.setSpacing(6)

        self.plot_widget = pg.PlotWidget(enableMenu=False)
        try:
            self.plot_widget.hideButtons()
        except Exception:
            pass
        self.plot_widget.setBackground(pg.mkColor(236, 239, 244))
        self.plot_widget.setAntialiasing(False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Amplitude", units="V")
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, 1.0, padding=0.0)
        plot_item.setYRange(-1.0, 1.0, padding=0.0)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.raw_row_layout.addWidget(self.plot_widget, stretch=7)

        self.sta_plot = pg.PlotWidget(enableMenu=False)
        try:
            self.sta_plot.hideButtons()
        except Exception:
            pass
        self.sta_plot.setBackground(pg.mkColor(250, 250, 252))
        self.sta_plot.showGrid(x=True, y=True, alpha=0.25)
        self.sta_plot.setLabel("bottom", "Lag", units="ms")
        self.sta_plot.setLabel("left", "Amplitude", units="mV")
        self.sta_plot.setMouseEnabled(x=False, y=False)
        self.raw_row_layout.addWidget(self.sta_plot, stretch=3)
        self.sta_plot.setAntialiasing(False)
        self._sta_median_curve = pg.PlotCurveItem()
        self._sta_median_curve.setZValue(10)
        self.sta_plot.addItem(self._sta_median_curve)
        self._sta_trace_items: list[pg.PlotCurveItem] = []
        self._sta_max_traces: int = 250

        layout.addWidget(self.raw_row_widget, stretch=4)
        self._hide_sta_plot()

        self.metrics_plot = pg.PlotWidget(enableMenu=False)
        try:
            self.metrics_plot.hideButtons()
        except Exception:
            pass
        self.metrics_plot.setBackground(pg.mkColor(245, 246, 250))
        self.metrics_plot.setLabel("bottom", "Time (s)")
        self.metrics_plot.setLabel("left", "Max Amplitude (V)")
        self.metrics_plot.showGrid(x=True, y=True, alpha=0.3)
        metrics_item = self.metrics_plot.getPlotItem()
        vb = metrics_item.getViewBox()
        if vb is not None:
            vb.setMouseEnabled(x=False, y=False)
        energy_scatter_color = QtGui.QColor(UNCLASSIFIED_COLOR)
        energy_scatter_color.setAlpha(170)
        self.energy_scatter = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(energy_scatter_color), pen=None, name="Energy Density")
        self.metrics_plot.addItem(self.energy_scatter)
        self.energy_scatter.hide()

        controls = QtWidgets.QGroupBox("Display")
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(8, 10, 8, 10)
        controls_layout.setSpacing(12)

        size_layout = QtWidgets.QVBoxLayout()
        size_layout.setSpacing(4)

        self.pause_viz_btn = QtWidgets.QPushButton("Pause Viz")
        self.pause_viz_btn.setCheckable(True)
        self.pause_viz_btn.setToolTip("Pause/resume updating the raw trace (analysis continues).")
        self.pause_viz_btn.toggled.connect(self._on_pause_viz_toggled)
        self.pause_viz_btn.setFixedWidth(90)
        size_layout.addWidget(self.pause_viz_btn)

        width_row = QtWidgets.QHBoxLayout()
        width_row.setSpacing(6)
        width_row.addWidget(QtWidgets.QLabel("Width (s)"))
        self.width_combo = QtWidgets.QComboBox()
        self.width_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.width_combo.setFixedWidth(80)
        for value in (0.2, 0.5, 1.0, 2.0, 5.0):
            self.width_combo.addItem(f"{value:.1f}", value)
        self.width_combo.setCurrentIndex(1)
        width_row.addWidget(self.width_combo)
        size_layout.addLayout(width_row)

        height_row = QtWidgets.QHBoxLayout()
        height_row.setSpacing(6)
        height_row.addWidget(QtWidgets.QLabel("Height (±V)"))
        self.height_combo = QtWidgets.QComboBox()
        self.height_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.height_combo.setFixedWidth(80)
        for value in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
            self.height_combo.addItem(f"{value:.1f}", value)
        self.height_combo.setCurrentIndex(3)
        height_row.addWidget(self.height_combo)
        size_layout.addLayout(height_row)
        controls_layout.addLayout(size_layout)

        threshold_layout = QtWidgets.QVBoxLayout()
        threshold_layout.setSpacing(4)

        self.threshold1_check = QtWidgets.QCheckBox("Threshold 1")
        self.threshold1_spin = QtWidgets.QDoubleSpinBox()
        self.threshold1_spin.setDecimals(3)
        self.threshold1_spin.setMinimumWidth(90)
        self.threshold1_spin.setRange(-10.0, 10.0)
        self.threshold1_spin.setValue(0.5)
        t1_row = QtWidgets.QHBoxLayout()
        t1_row.setSpacing(6)
        t1_row.addWidget(self.threshold1_check)
        t1_row.addWidget(self.threshold1_spin)
        threshold_layout.addLayout(t1_row)

        self.threshold2_check = QtWidgets.QCheckBox("Threshold 2")
        self.threshold2_check.setEnabled(True)
        self.threshold2_spin = QtWidgets.QDoubleSpinBox()
        self.threshold2_spin.setDecimals(3)
        self.threshold2_spin.setMinimumWidth(90)
        self.threshold2_spin.setRange(-10.0, 10.0)
        self.threshold2_spin.setValue(-0.5)
        t2_row = QtWidgets.QHBoxLayout()
        t2_row.setSpacing(6)
        t2_row.addWidget(self.threshold2_check)
        t2_row.addWidget(self.threshold2_spin)
        threshold_layout.addLayout(t2_row)

        event_window_row = QtWidgets.QHBoxLayout()
        event_window_row.setSpacing(6)
        event_window_row.addWidget(QtWidgets.QLabel("Event window width"))
        self.event_window_combo = QtWidgets.QComboBox()
        self.event_window_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.event_window_combo.setMinimumWidth(110)
        for label, value in (("5 ms", 5.0), ("10 ms", 10.0), ("20 ms", 20.0)):
            self.event_window_combo.addItem(label, value)
        self._set_event_window_selection(self._event_window_ms)
        event_window_row.addWidget(self.event_window_combo)
        threshold_layout.addLayout(event_window_row)

        controls_layout.addLayout(threshold_layout)

        metrics_layout = QtWidgets.QVBoxLayout()
        metrics_layout.setSpacing(4)
        metrics_label = QtWidgets.QLabel("Metrics")
        metrics_label.setStyleSheet("font-weight: bold;")
        metrics_layout.addWidget(metrics_label)

        metric_row = QtWidgets.QHBoxLayout()
        metric_row.setSpacing(6)
        metric_row.addWidget(QtWidgets.QLabel("Vertical (Y) Metric"))
        self.metric_combo = QtWidgets.QComboBox()
        for label in (
            "Max in window (V)",
            "Min in window (V)",
            "Energy Density (V²/s)",
            "Peak Frequency (Hz)",
            "Peak Wavelength (s)",
            "Interval since last event (s)",
        ):
            self.metric_combo.addItem(label)
        self.metric_combo.setCurrentIndex(0)
        self.metric_combo.setMinimumWidth(180)
        self.metric_combo.currentIndexChanged.connect(self._on_axis_metric_changed)
        metric_row.addWidget(self.metric_combo)
        metrics_layout.addLayout(metric_row)

        xaxis_row = QtWidgets.QHBoxLayout()
        xaxis_row.setSpacing(6)
        xaxis_row.addWidget(QtWidgets.QLabel("Horizontal (X) Axis"))
        self.metric_xaxis_combo = QtWidgets.QComboBox()
        for label in ("Time (s)", "Max in window (V)", "Min in window (V)", "Energy Density (V²/s)", "Peak Frequency (Hz)", "Peak Wavelength (s)"):
            self.metric_xaxis_combo.addItem(label)
        self.metric_xaxis_combo.setMinimumWidth(180)
        self.metric_xaxis_combo.currentIndexChanged.connect(self._on_axis_metric_changed)
        xaxis_row.addWidget(self.metric_xaxis_combo)
        metrics_layout.addLayout(xaxis_row)

        self.metrics_clear_btn = QtWidgets.QPushButton("Clear metrics")
        self.metrics_clear_btn.clicked.connect(self._clear_metrics)
        metrics_layout.addWidget(self.metrics_clear_btn)
        self.clustering_enabled_check = QtWidgets.QCheckBox("Enable clustering")
        self.clustering_enabled_check.toggled.connect(self._on_clustering_toggled)
        metrics_layout.addWidget(self.clustering_enabled_check)
        metrics_layout.addStretch(1)

        controls_layout.addLayout(metrics_layout)

        sta_layout = QtWidgets.QVBoxLayout()
        sta_layout.setSpacing(4)

        self.sta_enable_check = QtWidgets.QCheckBox("Enable Cross Channel Correlation")
        self.sta_enable_check.setChecked(False)
        self.sta_enable_check.toggled.connect(self._on_sta_toggled)
        sta_layout.addWidget(self.sta_enable_check)

        sta_source_row = QtWidgets.QHBoxLayout()
        sta_source_row.setSpacing(6)
        sta_source_row.addWidget(QtWidgets.QLabel("Event source"))
        self.sta_source_combo = QtWidgets.QComboBox()
        self.sta_source_combo.addItem("All events")
        self.sta_source_combo.currentIndexChanged.connect(self._on_sta_source_changed)
        sta_source_row.addWidget(self.sta_source_combo)
        sta_layout.addLayout(sta_source_row)

        sta_channel_row = QtWidgets.QHBoxLayout()
        sta_channel_row.setSpacing(6)
        sta_channel_row.addWidget(QtWidgets.QLabel("Signal channel"))
        self.sta_channel_combo = QtWidgets.QComboBox()
        self.sta_channel_combo.currentIndexChanged.connect(self._on_sta_channel_changed)
        sta_channel_row.addWidget(self.sta_channel_combo)
        sta_layout.addLayout(sta_channel_row)

        sta_window_row = QtWidgets.QHBoxLayout()
        sta_window_row.setSpacing(6)
        sta_window_row.addWidget(QtWidgets.QLabel("Window (ms)"))
        self.sta_window_combo = QtWidgets.QComboBox()
        for label, value in (("20", 20.0), ("50", 50.0), ("100", 100.0)):
            self.sta_window_combo.addItem(label, value)
        self.sta_window_combo.setCurrentIndex(1)
        self.sta_window_combo.currentIndexChanged.connect(self._on_sta_window_changed)
        sta_window_row.addWidget(self.sta_window_combo)
        sta_layout.addLayout(sta_window_row)

        self.sta_clear_btn = QtWidgets.QPushButton("Clear STA")
        self.sta_clear_btn.clicked.connect(self._on_sta_clear_clicked)
        sta_layout.addWidget(self.sta_clear_btn)
        sta_layout.addStretch(1)

        controls_layout.addLayout(sta_layout)
        controls_layout.addStretch(1)

        controls.setLayout(controls_layout)
        layout.addWidget(controls, stretch=2)
        self._refresh_sta_source_options()

        self.metrics_container = QtWidgets.QWidget(self)
        metrics_container_layout = QtWidgets.QHBoxLayout(self.metrics_container)
        metrics_container_layout.setContentsMargins(0, 0, 0, 0)
        metrics_container_layout.setSpacing(10)
        metrics_container_layout.addWidget(self.metrics_plot, stretch=1)
        self.metrics_container_layout = metrics_container_layout

        self.cluster_panel = QtWidgets.QGroupBox("Event Clusters")
        cluster_layout = QtWidgets.QVBoxLayout()
        cluster_layout.setContentsMargins(12, 10, 12, 10)
        cluster_layout.setSpacing(6)
        self.add_class_btn = QtWidgets.QPushButton("Add class…")
        self.add_class_btn.clicked.connect(self._on_add_class_clicked)
        cluster_layout.addWidget(self.add_class_btn)
        self.remove_class_btn = QtWidgets.QPushButton("Remove class")
        self.remove_class_btn.clicked.connect(self._on_remove_class_clicked)
        cluster_layout.addWidget(self.remove_class_btn)
        self.export_class_btn = QtWidgets.QPushButton("Export class to CSV…")
        self.export_class_btn.clicked.connect(self._on_export_class_clicked)
        cluster_layout.addWidget(self.export_class_btn)
        self.view_class_waveforms_btn = QtWidgets.QPushButton("View waveforms…")
        self.view_class_waveforms_btn.clicked.connect(self._on_view_class_waveforms_clicked)
        cluster_layout.addWidget(self.view_class_waveforms_btn)
        self.class_list = QtWidgets.QListWidget()
        self.class_list.currentItemChanged.connect(self._on_class_selection_changed)
        cluster_layout.addWidget(self.class_list, stretch=1)
        cluster_layout.addStretch(1)
        self.cluster_panel.setLayout(cluster_layout)
        metrics_container_layout.addWidget(self.cluster_panel, stretch=0)
        self._set_cluster_panel_visible(False)

        layout.addWidget(self.metrics_container, stretch=4)

        self.raw_curve = self.plot_widget.plot(pen=pg.mkPen((30, 144, 255), width=2))
        self.event_curve = self.plot_widget.plot(pen=pg.mkPen((200, 0, 0), width=2))

        self.threshold1_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen((128, 0, 128), width=3))
        self.threshold1_line.setZValue(10)
        self.threshold1_line.setVisible(False)
        self.plot_widget.addItem(self.threshold1_line)

        self.threshold2_line = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen((150, 150, 150), width=3, style=QtCore.Qt.DashLine),
        )
        self.threshold2_line.setZValue(10)
        self.threshold2_line.setVisible(False)
        self.plot_widget.addItem(self.threshold2_line)

        layout.addStretch(1)

        self.width_combo.currentIndexChanged.connect(self._apply_ranges)
        self.height_combo.currentIndexChanged.connect(self._apply_ranges)
        self.threshold1_check.toggled.connect(lambda checked: self._toggle_threshold(self.threshold1_line, self.threshold1_spin, checked))
        self.threshold1_check.toggled.connect(self._on_threshold1_toggled)
        self.threshold1_spin.valueChanged.connect(lambda val: self._update_threshold_from_spin(self.threshold1_line, val))
        self.threshold1_spin.valueChanged.connect(lambda _: self._notify_threshold_change())
        self.threshold1_line.sigPositionChanged.connect(lambda _: self._update_spin_from_line(self.threshold1_line, self.threshold1_spin))
        self.threshold2_check.toggled.connect(self._on_threshold2_toggled)
        self.threshold2_check.toggled.connect(lambda checked: self._toggle_threshold(self.threshold2_line, self.threshold2_spin, checked))
        self.threshold2_spin.valueChanged.connect(lambda val: self._update_threshold_from_spin(self.threshold2_line, val))
        self.threshold2_spin.valueChanged.connect(lambda _: self._notify_threshold_change())
        self.threshold2_line.sigPositionChanged.connect(lambda _: self._update_spin_from_line(self.threshold2_line, self.threshold2_spin))
        self.event_window_combo.currentIndexChanged.connect(self._on_event_window_changed)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()
        self._metrics_timer = QtCore.QTimer(self)
        self._metrics_timer.setInterval(100)
        self._metrics_timer.timeout.connect(self._on_metrics_timer)
        self._metrics_timer.start()
        self._sta_timer = QtCore.QTimer(self)
        self._sta_timer.setInterval(self._sta_update_interval_ms)
        self._sta_timer.timeout.connect(self._on_sta_timer)
        self._sta_timer.start()
        self._in_threshold_update = False
        self._on_axis_metric_changed()
        self._apply_ranges()
        self._update_cluster_button_states()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        executor = getattr(self, "_analysis_executor", None)
        if executor is not None:
            try:
                for fut in list(self._analysis_futures):
                    fut.cancel()
                executor.shutdown(wait=False)
            except Exception:
                pass
            self._analysis_executor = None
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _init_buffer(self) -> None:
        samples = max(1, int(round(self.sample_rate * self._buffer_span_sec))) if self.sample_rate > 0 else 1
        self._buffer = np.zeros(samples, dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_filled = 0

    def _ensure_buffer_capacity(self, required_sec: float) -> None:
        if self.sample_rate <= 0:
            return
        required_sec = max(required_sec, 0.1)
        if required_sec <= self._buffer_span_sec and self._buffer.size:
            return
        self._buffer_span_sec = max(required_sec * 1.25, self._buffer_span_sec)
        new_size = max(1, int(round(self.sample_rate * self._buffer_span_sec)))
        recent = self._recent_data(min(self._buffer_filled, new_size))
        self._buffer = np.zeros(new_size, dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_filled = 0
        if recent.size:
            self._append_to_buffer(recent)

    def _append_to_buffer(self, data: np.ndarray) -> None:
        if data.size == 0:
            return
        if data.size >= self._buffer.size:
            self._buffer[:] = data[-self._buffer.size :]
            self._buffer_pos = 0
            self._buffer_filled = self._buffer.size
            return
        end = self._buffer_pos + data.size
        if end <= self._buffer.size:
            self._buffer[self._buffer_pos:end] = data
        else:
            first = self._buffer.size - self._buffer_pos
            self._buffer[self._buffer_pos:] = data[:first]
            self._buffer[: data.size - first] = data[first:]
        self._buffer_pos = (self._buffer_pos + data.size) % self._buffer.size
        self._buffer_filled = min(self._buffer_filled + data.size, self._buffer.size)

    def _recent_data(self, count: int) -> np.ndarray:
        if self._buffer_filled == 0 or count <= 0:
            return np.empty(0, dtype=np.float32)
        count = min(count, self._buffer_filled)
        start = (self._buffer_pos - count) % self._buffer.size
        if start + count <= self._buffer.size:
            return self._buffer[start:start + count].copy()
        first = self._buffer.size - start
        return np.concatenate((self._buffer[start:], self._buffer[:count - first]))

    def _extract_recent(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.float32)
        data = self._recent_data(count)
        if data.size == count:
            return data
        out = np.full(count, np.nan, dtype=np.float32)
        if data.size:
            out[-data.size :] = data
        return out

    def _title_text(self) -> str:
        return f"{self.channel_name} \u2013 {self.sample_rate:,.0f} Hz"

    def set_channel_info(self, channel_name: str, sample_rate: float) -> None:
        self.channel_name = channel_name
        self.sample_rate = float(sample_rate)
        self.title_label.setText(self._title_text())
        self._dt = 1.0 / self.sample_rate if self.sample_rate > 0 else self._dt
        self._init_buffer()
        self._latest_sample_time = None
        self._window_start_time = None
        self._latest_sample_index = None
        self._window_start_index = None
        self._clear_event_overlays()

    def set_analysis_queue(self, q: "queue.Queue") -> None:
        self._analysis_queue = q

    def set_worker(self, worker: "AnalysisWorker") -> None:
        self._worker = worker
        if self._worker is not None and self.sample_rate > 0:
            try:
                self._worker.update_sample_rate(self.sample_rate)
            except AttributeError:
                pass
        self._notify_threshold_change()

    def set_sta_channels(self, channels: Sequence["ChannelInfo"]) -> None:
        self._refresh_sta_channel_options(channels)

    def peek_all_events(self) -> list[Event]:
        if self._event_buffer is None:
            return []
        return self._event_buffer.peek_all()

    def drain_events(self) -> list[Event]:
        if self._event_buffer is None:
            return []
        return self._event_buffer.drain()

    def _toggle_threshold(self, line: pg.InfiniteLine, spin: QtWidgets.QDoubleSpinBox, checked: bool) -> None:
        line.setVisible(checked)
        if checked:
            self._in_threshold_update = True
            line.setValue(float(spin.value()))
            self._in_threshold_update = False
            line.setZValue(20)

    def _update_threshold_from_spin(self, line: pg.InfiniteLine, value: float) -> None:
        if not line.isVisible() or self._in_threshold_update:
            return
        self._in_threshold_update = True
        line.setValue(float(value))
        self._in_threshold_update = False

    def _update_spin_from_line(self, line: pg.InfiniteLine, spin: QtWidgets.QDoubleSpinBox) -> None:
        if not line.isVisible() or self._in_threshold_update:
            return
        self._in_threshold_update = True
        try:
            spin.blockSignals(True)
            spin.setValue(float(line.value()))
        finally:
            spin.blockSignals(False)
            self._in_threshold_update = False
        if spin in (self.threshold1_spin, self.threshold2_spin):
            self._notify_threshold_change()

    def _on_timer(self) -> None:
        if self._analysis_queue is None:
            return
        max_batches = 3
        processed = 0
        while processed < max_batches:
            try:
                item = self._analysis_queue.get_nowait()
            except queue.Empty:
                break
            if item is EndOfStream:
                continue
            if isinstance(item, AnalysisBatch):
                batch = item
            elif isinstance(item, Chunk):
                meta = getattr(item, "meta", None)
                events_from_meta = ()
                if meta is not None and hasattr(meta, "get"):
                    events_from_meta = tuple(meta.get("analysis_events") or ())
                batch = AnalysisBatch(chunk=item, events=events_from_meta)
            else:
                continue
            self._render_batch(batch)
            processed += 1
        if self._analysis_events is not None:
            new_events, new_last_id = self._analysis_events.pull_events(self._last_event_id)
            if new_events:
                window_start = self._last_window_start
                width = self._last_window_width or float(self.width_combo.currentData() or 0.5)
                self._handle_batch_events(new_events, window_start, width, self._window_start_index)
                self._last_event_id = new_last_id

    def _on_metrics_timer(self) -> None:
        if not self._metrics_dirty:
            return
        self._update_metric_points()
        self._metrics_dirty = False

    def _on_sta_timer(self) -> None:
        if not self._sta_enabled or not self._sta_dirty:
            return
        self._refresh_sta_plot()
        self._sta_dirty = False

    def _apply_ranges(self) -> None:
        width = float(self.width_combo.currentData() or 0.5)
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, width, padding=0.0)
        if self._selected_y_metric() in {"ed", "max", "min", "freq", "period", "interval"}:
            # Metrics overlay controls the Y range; leave amplitude height unchanged.
            pass
        else:
            height = float(self.height_combo.currentData() or 1.0)
            plot_item.setYRange(-height, height, padding=0.0)
        self._ensure_buffer_capacity(max(width, 1.0))
        if self.threshold1_check.isChecked():
            self.threshold1_line.setValue(float(self.threshold1_spin.value()))
        if self._window_start_time is not None:
            self._last_window_start = float(self._window_start_time)
        self._last_window_width = float(width)
        if not self._viz_paused:
            self._refresh_overlay_positions(self._last_window_start, self._last_window_width, self._window_start_index)

    def _initial_event_window_ms(self) -> float:
        if self._analysis_settings is None:
            return 10.0
        try:
            return float(self._analysis_settings.get().event_window_ms)
        except Exception:
            return 10.0

    def _set_event_window_selection(self, value_ms: float) -> None:
        target = float(value_ms)
        for idx in range(self.event_window_combo.count()):
            item_value = self.event_window_combo.itemData(idx)
            if item_value is None:
                continue
            if abs(float(item_value) - target) < 1e-6:
                try:
                    self.event_window_combo.blockSignals(True)
                    self.event_window_combo.setCurrentIndex(idx)
                finally:
                    self.event_window_combo.blockSignals(False)
                return

    def _on_event_window_changed(self, index: int) -> None:
        value = self.event_window_combo.itemData(index)
        if value is None:
            return
        self._event_window_ms = float(value)
        controller = self._controller
        if controller is not None:
            try:
                controller.update_analysis_settings(event_window_ms=self._event_window_ms)
            except AttributeError:
                pass
        self._notify_threshold_change()

    def _on_threshold1_toggled(self, checked: bool) -> None:
        self.threshold2_check.setEnabled(checked)
        self.threshold2_spin.setEnabled(checked and self.threshold2_check.isChecked())
        if not checked:
            self.threshold2_check.setChecked(False)
        self._notify_threshold_change()

    def _on_threshold2_toggled(self, checked: bool) -> None:
        self.threshold2_spin.setEnabled(checked)
        self._notify_threshold_change()

    def _notify_threshold_change(self) -> None:
        if self._worker is None:
            return
        self._clear_metrics()
        enabled = self.threshold1_check.isChecked()
        value = float(self.threshold1_spin.value())
        secondary_enabled = enabled and self.threshold2_check.isChecked()
        secondary_value = float(self.threshold2_spin.value())
        try:
            self._worker.configure_threshold(
                enabled,
                value,
                secondary_enabled=secondary_enabled,
                secondary_value=secondary_value,
            )
        except Exception:
            pass

    def _render_batch(self, batch: AnalysisBatch) -> None:
        chunk = batch.chunk
        try:
            data = np.array(chunk.samples, dtype=np.float32)
        except Exception:
            return
        if data.size == 0 or data.ndim != 2:
            self.raw_curve.clear()
            return
        channel = np.array(data[0], copy=False)
        frames = int(channel.size)
        if frames == 0:
            self.raw_curve.clear()
            return
        if chunk.dt > 0:
            self._dt = float(chunk.dt)
        self._append_to_buffer(channel.astype(np.float32, copy=False))
        meta = getattr(chunk, "meta", None)
        start_sample = None
        if meta is not None:
            try:
                idx = meta.get("source_channel_index")
            except AttributeError:
                idx = None
            if isinstance(idx, int):
                self._channel_index = idx
            try:
                start_sample_val = meta.get("start_sample")
            except AttributeError:
                start_sample_val = None
            try:
                start_sample = int(start_sample_val) if start_sample_val is not None else None
            except (TypeError, ValueError):
                start_sample = None
        if start_sample is None:
            start_sample = self._fallback_start_sample
        try:
            chunk_dt = float(chunk.dt)
        except Exception:
            chunk_dt = self._dt
        self._latest_sample_time = float(chunk.start_time) + frames * chunk_dt

        width = float(self.width_combo.currentData() or 0.5)
        self._ensure_buffer_capacity(max(width, 1.0))
        samples_needed = int(max(1, round(width / self._dt))) if self._dt > 0 else frames
        recent = self._extract_recent(samples_needed)
        span_samples = max(1, recent.size)
        window_duration = (span_samples - 1) * self._dt if span_samples > 1 else 0.0
        if self._latest_sample_time is not None:
            self._window_start_time = self._latest_sample_time - window_duration
        else:
            self._window_start_time = None
        if start_sample is not None and start_sample >= 0:
            self._latest_sample_index = start_sample + frames - 1
            self._window_start_index = self._latest_sample_index - (span_samples - 1)
            self._fallback_start_sample = start_sample + frames
        else:
            self._latest_sample_index = None
            self._window_start_index = None
        times = np.arange(recent.size, dtype=np.float32) * self._dt
        self._cached_raw_times = times
        self._cached_raw_samples = recent
        if not self._viz_paused:
            self.raw_curve.setData(times, recent, skipFiniteCheck=True)
            self.event_curve.clear()

        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, width, padding=0.0)
        height = float(self.height_combo.currentData() or 1.0)
        plot_item.setYRange(-height, height, padding=0.0)

        if self._window_start_time is not None:
            window_start = float(self._window_start_time)
        elif self._latest_sample_time is not None:
            window_start = float(self._latest_sample_time - width)
        else:
            window_start = 0.0
        if self._latest_sample_time is not None:
            width_in_use = max(width, float(self._latest_sample_time) - window_start)
        else:
            width_in_use = width
        self._last_window_start = float(window_start)
        self._last_window_width = float(width_in_use)

        events = batch.events or ()
        if not events:
            meta = getattr(batch.chunk, "meta", None)
            if meta is not None and hasattr(meta, "get"):
                events = tuple(meta.get("analysis_events") or ())
        self._handle_batch_events(events, window_start, width_in_use, self._window_start_index)

    def _on_pause_viz_toggled(self, checked: bool) -> None:
        self._viz_paused = bool(checked)
        if not self._viz_paused:
            self._refresh_raw_plot()
            self._refresh_overlay_positions(self._last_window_start, self._last_window_width, self._window_start_index)

    def _refresh_raw_plot(self) -> None:
        if self._viz_paused:
            return
        if self._cached_raw_times is None or self._cached_raw_samples is None:
            return
        self.raw_curve.setData(self._cached_raw_times, self._cached_raw_samples, skipFiniteCheck=True)
        self.event_curve.clear()

    def _acquire_overlay_item(self) -> pg.PlotCurveItem:
        if self._overlay_pool:
            item = self._overlay_pool.pop()
        else:
            item = pg.PlotCurveItem(pen=self._overlay_pen)
            item.setZValue(30)
            self.plot_widget.addItem(item)
        item.show()
        return item

    def _release_overlay_item(self, item: Optional[pg.PlotCurveItem]) -> None:
        if item is None:
            return
        item.hide()
        item.setData([], [])
        self._overlay_pool.append(item)

    def _apply_overlay_color(self, overlay: dict[str, object]) -> None:
        item = overlay.get("item") or overlay.get("curve")
        if not isinstance(item, pg.PlotCurveItem):
            return
        event_id = overlay.get("event_id")
        event_id_int = event_id if isinstance(event_id, int) else None
        color = self._get_event_color(event_id_int)
        base_pen = getattr(item, "opts", {}).get("pen") if hasattr(item, "opts") else None
        if base_pen is None:
            base_pen = getattr(self, "_overlay_pen", None)
        width = 2.0
        if base_pen is not None:
            width_attr = getattr(base_pen, "widthF", None)
            if callable(width_attr):
                try:
                    width = float(width_attr())
                except Exception:
                    width = 2.0
            else:
                try:
                    width = float(base_pen.width())
                except Exception:
                    width = 2.0
        item.setPen(pg.mkPen(color, width=width))

    def _refresh_overlay_colors(self) -> None:
        """Update the pens for all raw event overlays based on cluster labels."""
        if not self._event_overlays:
            return
        for overlay in self._event_overlays:
            self._apply_overlay_color(overlay)

    def _handle_batch_events(
        self,
        events: Sequence[Event],
        window_start: float,
        width: float,
        window_start_idx: Optional[int],
    ) -> None:
        pending_sta = bool(getattr(self, "_sta_pending_events", None))
        if events:
            self._submit_analysis_job(tuple(events), window_start, width, window_start_idx)
        elif self._sta_enabled and pending_sta:
            task = self._build_sta_task(())
            if task is not None:
                self._sta_handle_task(task)
        if not self._viz_paused:
            self._refresh_overlay_positions(window_start, width, window_start_idx)

    def _build_sta_task(self, events: Sequence[Event]) -> StaTask | None:
        if not self._sta_enabled:
            return None
        target_channel_id = self._sta_target_channel_id
        if target_channel_id is None:
            return None
        channel_info = int(self._channel_index) if getattr(self, "_channel_index", None) is not None else None
        return StaTask(
            events=tuple(events),
            target_channel_id=int(target_channel_id),
            channel_index=channel_info,
            window_ms=float(self._sta_window_ms),
        )

    def _submit_analysis_job(
        self,
        events: tuple[Event, ...],
        window_start: float,
        width: float,
        window_start_idx: Optional[int],
    ) -> None:
        if not events:
            return
        executor = self._analysis_executor
        if executor is None:
            self._apply_analysis_update(
                self._build_analysis_update(events),
                window_start,
                width,
                window_start_idx,
            )
            return
        if len(self._analysis_futures) >= 3:
            # Avoid dropping overlays: process inline if the worker is saturated.
            self._apply_analysis_update(
                self._build_analysis_update(events),
                window_start,
                width,
                window_start_idx,
            )
            return
        future = executor.submit(self._build_analysis_update, events)
        self._analysis_futures.add(future)

        def _on_done(fut: Future, params=(window_start, width, window_start_idx)) -> None:
            QtCore.QTimer.singleShot(
                0,
                lambda: self._on_analysis_update_ready(fut, *params),
            )

        future.add_done_callback(_on_done)

    def _on_analysis_update_ready(
        self,
        future: Future,
        window_start: float,
        width: float,
        window_start_idx: Optional[int],
    ) -> None:
        self._analysis_futures.discard(future)
        try:
            update = future.result()
        except Exception as exc:
            logging.getLogger(__name__).exception("Analysis worker failed: %s", exc)
            return
        self._apply_analysis_update(update, window_start, width, window_start_idx)

    def _build_analysis_update(self, events: tuple[Event, ...]) -> AnalysisUpdate:
        overlays: list[OverlayPayload] = []
        last_event_id: int | None = None
        for event in events:
            payload = self._build_overlay_payload(event)
            if payload is None:
                continue
            overlays.append(payload)
            event_id = payload.event_id
            if isinstance(event_id, int):
                last_event_id = event_id if last_event_id is None else max(last_event_id, event_id)
        sta_task = self._build_sta_task(events)
        return AnalysisUpdate(overlays=overlays, sta_task=sta_task, last_event_id=last_event_id)

    def _apply_analysis_update(
        self,
        update: AnalysisUpdate,
        window_start: float,
        width: float,
        window_start_idx: Optional[int],
    ) -> None:
        payloads = update.overlays or []
        for payload in payloads:
            reuse_overlay: dict[str, object] | None = None
            if len(self._event_overlays) >= self._overlay_capacity:
                reuse_overlay = self._event_overlays.pop(0)
            overlay = self._materialize_overlay(payload, reuse_overlay)
            if overlay is None:
                continue
            if not self._viz_paused:
                if not self._apply_overlay_view(overlay, window_start, width, window_start_idx):
                    self._release_overlay_item(overlay.get("item"))
                    continue
            self._event_overlays.append(overlay)
            self._record_overlay_metrics(overlay)
        if payloads:
            self._metrics_dirty = True
        if update.sta_windows:
            self._sta_windows.extend(update.sta_windows)
            if len(self._sta_windows) > self._sta_max_windows:
                self._sta_windows = self._sta_windows[-self._sta_max_windows :]
            self._sta_dirty = True
        elif update.sta_task is not None:
            self._sta_handle_task(update.sta_task)
        if isinstance(update.last_event_id, int):
            self._last_event_id = (
                update.last_event_id
                if self._last_event_id is None
                else max(self._last_event_id, update.last_event_id)
            )
        if not self._viz_paused and payloads:
            self._refresh_overlay_positions(window_start, width, window_start_idx)

    def _refresh_overlay_positions(self, window_start: float, width: float, window_start_idx: Optional[int]) -> None:
        if self._viz_paused:
            return
        if not self._event_overlays:
            return
        kept: list[dict[str, object]] = []
        for overlay in self._event_overlays:
            last_time = float(overlay.get("last_time", window_start))
            item = overlay.get("item")
            if last_time < window_start:
                self._release_overlay_item(item)
                continue
            if not self._apply_overlay_view(overlay, window_start, width, window_start_idx):
                self._release_overlay_item(item)
                continue
            self._apply_overlay_color(overlay)
            kept.append(overlay)
        self._event_overlays = kept

    def _clear_event_overlays(self) -> None:
        if not self._event_overlays:
            return
        for overlay in self._event_overlays:
            self._release_overlay_item(overlay.get("item"))
        self._event_overlays.clear()

    def _ensure_sta_trace_capacity(self, capacity: int) -> None:
        current = len(self._sta_trace_items)
        if current >= capacity:
            return
        for _ in range(capacity - current):
            item = pg.PlotCurveItem(pen=STA_TRACE_PEN)
            item.setZValue(-2)
            self.sta_plot.addItem(item)
            self._sta_trace_items.append(item)

    def _clear_sta_traces(self) -> None:
        for item in self._sta_trace_items:
            item.hide()
            item.setData([], [])

    def _clear_metrics(self) -> None:
        self._metric_events.clear()
        self._event_details.clear()
        self._event_cluster_labels.clear()
        self._refresh_overlay_colors()
        self._t0_event = None
        for cluster in self._clusters:
            item = self._cluster_items.get(cluster.id)
            if item is not None:
                item.setText(f"{cluster.name} (0 events)")
        if hasattr(self, "energy_scatter"):
            self.energy_scatter.clear()
            self.energy_scatter.hide()
        if hasattr(self, "metrics_plot"):
            self.metrics_plot.getPlotItem().enableAutoRange(y=True)
        self._update_metric_points()

    def get_energy_density_points(self) -> list[tuple[float, float]]:
        return [
            (event["time"], event["ed"])
            for event in self._metric_events
            if "time" in event and "ed" in event
        ]

    def _set_metrics_range(self, plot_item: pg.PlotItem, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        if max_x < min_x:
            min_x, max_x = max_x, min_x
        span_x = max(1e-6, max_x - min_x)
        padded_min_x = min_x - span_x * 0.02
        padded_max_x = max_x + span_x * 0.02
        span_y = max(1e-6, max_y - min_y)
        padded_min = min_y - span_y * 0.05
        padded_max = max_y + span_y * 0.05
        plot_item.setXRange(padded_min_x, padded_max_x, padding=0)
        plot_item.setYRange(padded_min, padded_max, padding=0.0)

    def _selected_y_metric(self) -> str:
        label = self.metric_combo.currentText().lower()
        if "interval" in label:
            return "interval"
        if "max" in label:
            return "max"
        if "min" in label:
            return "min"
        if "frequency" in label:
            return "freq"
        if "wavelength" in label:
            return "period"
        return "ed"

    def _selected_x_metric(self) -> str:
        label = self.metric_xaxis_combo.currentText().lower()
        if "time" in label:
            return "time"
        if "frequency" in label:
            return "freq"
        if "wavelength" in label:
            return "period"
        if "max" in label:
            return "max"
        if "min" in label:
            return "min"
        return "ed"

    def _on_axis_metric_changed(self) -> None:
        metric = self._selected_y_metric()
        if metric == "ed":
            self.metrics_plot.setLabel("left", "Energy Density (V²/s)")
        elif metric == "max":
            self.metrics_plot.setLabel("left", "Max Amplitude (V)")
        elif metric == "min":
            self.metrics_plot.setLabel("left", "Min Amplitude (V)")
        elif metric == "freq":
            self.metrics_plot.setLabel("left", "Peak Frequency (Hz)")
        elif metric == "period":
            self.metrics_plot.setLabel("left", "Peak Wavelength (s)")
        elif metric == "interval":
            self.metrics_plot.setLabel("left", "Interval (s)")
        else:
            self.metrics_plot.setLabel("left", "Value")
        x_metric = self._selected_x_metric()
        if x_metric == "time":
            self.metrics_plot.setLabel("bottom", "Time (s)")
        elif x_metric == "ed":
            self.metrics_plot.setLabel("bottom", "Energy Density (V²/s)")
        elif x_metric == "max":
            self.metrics_plot.setLabel("bottom", "Max Amplitude (V)")
        elif x_metric == "min":
            self.metrics_plot.setLabel("bottom", "Min Amplitude (V)")
        elif x_metric == "freq":
            self.metrics_plot.setLabel("bottom", "Peak Frequency (Hz)")
        else:
            self.metrics_plot.setLabel("bottom", "Peak Wavelength (s)")
        if metric in {"ed", "max", "min", "freq", "period", "interval"}:
            self.energy_scatter.show()
        else:
            self.energy_scatter.hide()
            self.energy_scatter.setData([], [])
        if self.clustering_enabled_check.isChecked():
            self._recompute_cluster_membership()
        self._update_metric_points()

    def _on_class_selection_changed(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        previous: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        del previous
        if current is None:
            self._selected_cluster_id = None
        else:
            data = current.data(QtCore.Qt.UserRole)
            self._selected_cluster_id = int(data) if isinstance(data, int) else None
        self._update_cluster_visuals()
        self._update_cluster_button_states()

    def _update_cluster_visuals(self) -> None:
        base_width = 1.5
        selected_id = self._selected_cluster_id
        for cluster in self._clusters:
            roi = cluster.roi
            if roi is None:
                continue
            width = base_width * 3.0 if cluster.id == selected_id else base_width
            roi.setPen(pg.mkPen(cluster.color, width=width))

    def _update_cluster_button_states(self) -> None:
        enabled = self.clustering_enabled_check.isChecked()
        has_selection = self.class_list.currentItem() is not None
        self.add_class_btn.setEnabled(enabled)
        self.remove_class_btn.setEnabled(enabled and has_selection)
        self.view_class_waveforms_btn.setEnabled(enabled and has_selection)
        self.export_class_btn.setEnabled(enabled and has_selection)

    def _set_cluster_panel_visible(self, visible: bool) -> None:
        if visible:
            self.cluster_panel.show()
            self.metrics_container_layout.setStretchFactor(self.metrics_plot, 7)
            self.metrics_container_layout.setStretchFactor(self.cluster_panel, 3)
        else:
            self.cluster_panel.hide()
            self.metrics_container_layout.setStretchFactor(self.metrics_plot, 1)
            self.metrics_container_layout.setStretchFactor(self.cluster_panel, 0)

    def _refresh_sta_source_options(self) -> None:
        """Populate the STA event source combo: 'All events' + spike classes."""
        if not hasattr(self, "sta_source_combo"):
            return
        previous_selection = self._sta_source_cluster_id
        was_blocked = self.sta_source_combo.blockSignals(True)
        self.sta_source_combo.clear()
        self.sta_source_combo.addItem("All events", None)
        for cluster in self._clusters:
            self.sta_source_combo.addItem(cluster.name, cluster.id)
        target_index = self.sta_source_combo.findData(previous_selection)
        if target_index >= 0:
            self.sta_source_combo.setCurrentIndex(target_index)
        else:
            self.sta_source_combo.setCurrentIndex(0)
        current_data = self.sta_source_combo.currentData()
        self._sta_source_cluster_id = current_data if isinstance(current_data, int) else None
        self.sta_source_combo.blockSignals(was_blocked)

    def _refresh_sta_channel_options(self, channels: Sequence["ChannelInfo"]) -> None:
        """Populate the STA signal channel combo from the active channels."""
        combo = getattr(self, "sta_channel_combo", None)
        if combo is None:
            return
        previous = self._sta_target_channel_id
        block_state = combo.blockSignals(True)
        combo.clear()
        for ch in channels:
            cid = getattr(ch, "id", None)
            name = getattr(ch, "name", f"Channel {cid}")
            combo.addItem(str(name), cid)
        combo.blockSignals(block_state)
        target_index = combo.findData(previous) if previous is not None else -1
        if target_index < 0 and self._channel_index is not None:
            target_index = combo.findData(int(self._channel_index))
        if target_index < 0 and combo.count():
            target_index = 0
        if target_index >= 0:
            combo.setCurrentIndex(target_index)
            data = combo.itemData(target_index)
            self._sta_target_channel_id = data if isinstance(data, int) else None
        else:
            self._sta_target_channel_id = None

    def _show_sta_plot(self) -> None:
        self.sta_plot.show()
        self.raw_row_layout.setStretchFactor(self.plot_widget, 7)
        self.raw_row_layout.setStretchFactor(self.sta_plot, 3)

    def _hide_sta_plot(self) -> None:
        self.sta_plot.hide()
        self.raw_row_layout.setStretchFactor(self.plot_widget, 10)
        self.raw_row_layout.setStretchFactor(self.sta_plot, 0)

    def _get_cluster_for_event(self, event_id: int | None) -> MetricCluster | None:
        if event_id is None:
            return None
        cluster_id = self._event_cluster_labels.get(event_id)
        if cluster_id is None:
            return None
        for cluster in self._clusters:
            if cluster.id == cluster_id:
                return cluster
        return None

    def _get_event_color(self, event_id: int | None) -> QtGui.QColor:
        """Return the pen colour for a given event_id.

        Unclassified events are red; classified events use their cluster colour.
        """
        cluster = self._get_cluster_for_event(event_id)
        if cluster is None:
            return QtGui.QColor(UNCLASSIFIED_COLOR)
        return cluster.color

    def _on_clustering_toggled(self, checked: bool) -> None:
        self._set_cluster_panel_visible(checked)
        if checked:
            self._recompute_cluster_membership()
        else:
            self._event_cluster_labels.clear()
            for cluster in self._clusters:
                item = self._cluster_items.get(cluster.id)
                if item is not None:
                    item.setText(f"{cluster.name} (0 events)")
        self._update_metric_points()
        self._update_cluster_visuals()
        self._update_cluster_button_states()
        self._refresh_overlay_colors()

    def _on_sta_toggled(self, checked: bool) -> None:
        self._sta_enabled = bool(checked)
        if self._sta_enabled:
            self._show_sta_plot()
            self._on_sta_clear_clicked()
        else:
            self._hide_sta_plot()
            self._sta_windows.clear()
            self._sta_time_axis = None
            self._sta_pending_events.clear()
            self._sta_dirty = False
            self._sta_median_curve.hide()
            self._sta_median_curve.clear()
            self._clear_sta_traces()

    def _on_sta_source_changed(self, index: int) -> None:
        if index < 0:
            self._sta_source_cluster_id = None
        else:
            data = self.sta_source_combo.itemData(index)
            self._sta_source_cluster_id = data if isinstance(data, int) else None
        self._on_sta_clear_clicked()

    def _on_sta_channel_changed(self, index: int) -> None:
        if index < 0:
            self._sta_target_channel_id = None
        else:
            data = self.sta_channel_combo.itemData(index)
            if isinstance(data, int):
                self._sta_target_channel_id = data
            else:
                self._sta_target_channel_id = index if self.sta_channel_combo.count() else None
        self._on_sta_clear_clicked()

    def _on_sta_window_changed(self, index: int) -> None:
        if index < 0:
            return
        value = self.sta_window_combo.itemData(index)
        if value is None:
            try:
                value = float(self.sta_window_combo.currentText())
            except ValueError:
                value = self._sta_window_ms
        self._sta_window_ms = float(value)
        self._on_sta_clear_clicked()

    def _on_sta_clear_clicked(self) -> None:
        self._sta_windows.clear()
        self._sta_time_axis = None
        self._sta_pending_events.clear()
        self._sta_dirty = False
        self._clear_sta_traces()
        self._sta_median_curve.hide()
        self._sta_median_curve.clear()

    def _sta_handle_task(self, task: StaTask) -> None:
        controller = self._controller
        if controller is None:
            return
        target_channel_id = task.target_channel_id
        channel_info = task.channel_index
        pending_items = list(self._sta_pending_events.values())
        self._sta_pending_events.clear()
        queue: list[tuple[Event, int]] = [(event, 0) for event in task.events]
        queue.extend(pending_items)
        updated = False
        for event, attempts in queue:
            status = self._sta_process_event(
                controller,
                target_channel_id,
                channel_info,
                event,
                task.window_ms,
            )
            if status == "added":
                updated = True
            elif status == "pending":
                event_id = getattr(event, "id", None)
                if not isinstance(event_id, int):
                    continue
                if attempts >= self._sta_retry_limit:
                    continue
                self._sta_pending_events[event_id] = (event, attempts + 1)
        if updated:
            self._sta_dirty = True

    def _sta_process_event(
        self,
        controller: "PipelineController",
        target_channel_id: int,
        channel_info: Optional[int],
        event: Event,
        window_ms: float,
    ) -> str:
        event_channel = getattr(event, "channelId", getattr(event, "channel_id", None))
        if event_channel is not None and channel_info is not None:
            try:
                if int(event_channel) != int(channel_info):
                    return "skip"
            except (TypeError, ValueError):
                return "skip"
        event_id = getattr(event, "id", None)
        if self._sta_source_cluster_id is not None:
            if not isinstance(event_id, int):
                return "skip"
            cluster = self._get_cluster_for_event(event_id)
            if cluster is None:
                return "pending"
            if cluster.id != self._sta_source_cluster_id:
                return "skip"
        window, miss_pre, miss_post = controller.collect_trigger_window(
            event,
            target_channel_id=target_channel_id,
            window_ms=window_ms,
        )
        if miss_pre > 0:
            return "skip"
        if miss_post > 0:
            return "pending"
        if window.size == 0:
            return "pending"
        pre_n = max(1, int(0.2 * window.size))
        baseline = float(np.median(window[:pre_n]))
        normalized = window.astype(np.float32, copy=False) - baseline
        self._sta_windows.append(normalized)
        if len(self._sta_windows) > self._sta_max_windows:
            self._sta_windows = self._sta_windows[-self._sta_max_windows :]
        self._sta_dirty = True
        return "added"

    def _refresh_sta_plot(self) -> None:
        """Redraw the spike-triggered average plot."""
        plot_item = self.sta_plot.getPlotItem()
        if not self._sta_windows:
            self._sta_median_curve.hide()
            self._sta_median_curve.clear()
            self._clear_sta_traces()
            return
        min_len = min((w.size for w in self._sta_windows if isinstance(w, np.ndarray)), default=0)
        if min_len <= 1:
            self._sta_median_curve.hide()
            self._sta_median_curve.clear()
            self._clear_sta_traces()
            return
        windows = np.stack([np.asarray(w[:min_len], dtype=np.float32) for w in self._sta_windows], axis=0)
        assert windows.ndim == 2, "STA windows must be 2D (events x samples)"
        num_events, num_samples = windows.shape
        if windows.size == 0:
            self._sta_median_curve.hide()
            self._sta_median_curve.clear()
            self._clear_sta_traces()
            return
        duration_ms = float(self._sta_window_ms)
        t = np.linspace(-duration_ms / 2.0, duration_ms / 2.0, num_samples, dtype=np.float32)
        assert t.shape[0] == num_samples, "Time axis must align with samples"
        self._sta_time_axis = t
        amp_min = float(np.min(windows))
        amp_max = float(np.max(windows))
        if not np.isfinite(amp_min) or not np.isfinite(amp_max):
            amp_min, amp_max = -1.0, 1.0
        if amp_max - amp_min < 1e-6:
            bound = max(1e-3, abs(amp_max))
            amp_min = -bound
            amp_max = bound
        max_traces = min(num_events, self._sta_max_traces)
        visible_windows = windows[-max_traces:] if max_traces else []
        self._ensure_sta_trace_capacity(max_traces)
        for idx, item in enumerate(self._sta_trace_items):
            if idx < max_traces:
                waveform = visible_windows[idx]
                item.setPen(STA_TRACE_PEN)
                item.setData(t, waveform)
                item.show()
            else:
                item.hide()
                item.setData([], [])
        median = np.median(windows, axis=0)
        # STA curves always use (time -> x, amplitude -> y)
        self._sta_median_curve.setData(t, median)
        self._sta_median_curve.setPen(pg.mkPen(0, 250, 30, 255, width=3))
        self._sta_median_curve.show()
        plot_item.setLabel("bottom", "Lag", units="ms")
        plot_item.setLabel("left", "Amplitude", units="mV")
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        plot_item.setXRange(t[0], t[-1], padding=0.0)
        plot_item.setYRange(amp_min, amp_max, padding=0.05)

    def _on_add_class_clicked(self) -> None:
        if not self.clustering_enabled_check.isChecked():
            return
        view_box = self.metrics_plot.getViewBox()
        if view_box is None:
            return
        rect = view_box.viewRect()
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            return
        x_span = rect.width()
        y_span = rect.height()
        default_width = x_span * 0.3
        default_height = y_span * 0.3
        x0 = rect.left() + (x_span - default_width) * 0.5
        y0 = rect.top() + (y_span - default_height) * 0.5
        cluster_id = self._cluster_id_counter
        self._cluster_id_counter += 1
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        cluster_name = f"Class {cluster_id + 1}"
        cluster = MetricCluster(id=cluster_id, name=cluster_name, color=color)
        roi = ClusterRectROI((x0, y0), (default_width, default_height), pen=pg.mkPen(color, width=1.5))
        roi.setZValue(50)
        cluster.roi = roi
        roi.sigRegionChanged.connect(lambda _: self._on_cluster_roi_changed())
        self.metrics_plot.addItem(roi)
        self._clusters.append(cluster)
        item = QtWidgets.QListWidgetItem(f"{cluster.name} (0 events)")
        item.setData(QtCore.Qt.UserRole, cluster_id)
        self.class_list.addItem(item)
        self.class_list.setCurrentItem(item)
        self._cluster_items[cluster_id] = item
        self._refresh_sta_source_options()
        self._recompute_cluster_membership()
        self._update_metric_points()
        self._update_cluster_visuals()
        self._update_cluster_button_states()

    def _on_remove_class_clicked(self) -> None:
        current_item = self.class_list.currentItem()
        if current_item is None:
            return
        cluster_id = current_item.data(QtCore.Qt.UserRole)
        if not isinstance(cluster_id, int):
            return
        cluster = next((c for c in self._clusters if c.id == cluster_id), None)
        if cluster is not None and cluster.roi is not None:
            try:
                self.metrics_plot.removeItem(cluster.roi)
            except Exception:
                pass
        self._clusters = [c for c in self._clusters if c.id != cluster_id]
        row = self.class_list.row(current_item)
        if row >= 0:
            self.class_list.takeItem(row)
        count = self.class_list.count()
        if count > 0:
            new_row = min(row, count - 1)
            if new_row >= 0:
                self.class_list.setCurrentRow(new_row)
        else:
            self.class_list.setCurrentRow(-1)
        self._cluster_items.pop(cluster_id, None)
        to_remove = [event_id for event_id, cid in self._event_cluster_labels.items() if cid == cluster_id]
        for event_id in to_remove:
            self._event_cluster_labels.pop(event_id, None)
        self._refresh_sta_source_options()
        self._recompute_cluster_membership()
        self._update_metric_points()
        self._update_cluster_visuals()
        self._update_cluster_button_states()

    def _on_export_class_clicked(self) -> None:
        pass

    def _on_view_class_waveforms_clicked(self) -> None:
        if not self.clustering_enabled_check.isChecked():
            QtWidgets.QMessageBox.information(self, "Waveforms", "Enable clustering to view class waveforms.")
            return
        current_item = self.class_list.currentItem()
        if current_item is None:
            QtWidgets.QMessageBox.information(self, "Waveforms", "Select a class to view its waveforms.")
            return
        cluster_id = current_item.data(QtCore.Qt.UserRole)
        if not isinstance(cluster_id, int):
            return
        cluster = next((c for c in self._clusters if c.id == cluster_id), None)
        if cluster is None:
            return
        event_ids = [event_id for event_id, cid in self._event_cluster_labels.items() if cid == cluster_id]
        waveforms: list[tuple[np.ndarray, np.ndarray]] = []
        for event_id in event_ids:
            details = self._event_details.get(event_id)
            if not details:
                continue
            times = details.get("times")
            samples = details.get("samples")
            if times is None or samples is None:
                continue
            arr_t = np.asarray(times, dtype=np.float64)
            arr_s = np.asarray(samples, dtype=np.float32)
            if arr_t.size == 0 or arr_s.size == 0 or arr_t.size != arr_s.size:
                continue
            t_rel = arr_t - arr_t[0]
            baseline = float(np.median(arr_s)) if arr_s.size else 0.0
            s_rel = arr_s - baseline
            waveforms.append((t_rel, s_rel))
        if not waveforms:
            QtWidgets.QMessageBox.information(self, "Waveforms", "No waveform data available for this class.")
            return
        dialog = ClusterWaveformDialog(self, cluster.name, waveforms, cluster.color)
        dialog.exec()

    def _release_metrics(self) -> None:
        self._metric_events = []
        self._t0_event = None
        self._update_metric_points()

    def _update_metric_points(self) -> None:
        y_key = self._selected_y_metric()
        x_key = self._selected_x_metric()
        if y_key not in {"ed", "max", "min", "freq", "period", "interval"}:
            self.energy_scatter.hide()
            return
        events = self._metric_events
        if not events:
            self.energy_scatter.hide()
            return
        visible_events = events
        if x_key == "time":
            last_time: float | None = None
            for event in reversed(events):
                t_val = event.get("time")
                if t_val is None:
                    continue
                try:
                    last_time = float(t_val)
                except (TypeError, ValueError):
                    continue
                break
            if last_time is not None:
                min_time = last_time - METRIC_TIME_WINDOW_SEC
                if min_time > 0:
                    start_idx = 0
                    for idx, event in enumerate(events):
                        t_val = event.get("time")
                        if t_val is None:
                            continue
                        try:
                            if float(t_val) >= min_time:
                                start_idx = idx
                                break
                        except (TypeError, ValueError):
                            continue
                    visible_events = events[start_idx:]
        if len(visible_events) > MAX_VISIBLE_METRIC_EVENTS:
            visible_events = visible_events[-MAX_VISIBLE_METRIC_EVENTS:]
        xs: list[float] = []
        ys: list[float] = []
        event_ids: list[Optional[int]] = []
        for event in visible_events:
            x_val = event.get(x_key)
            y_val = event.get(y_key)
            if x_val is None or y_val is None:
                continue
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue
            xs.append(float(x_val))
            ys.append(float(y_val))
            event_id = event.get("event_id")
            event_ids.append(event_id if isinstance(event_id, int) else None)
        if not xs:
            self.energy_scatter.hide()
            return
        if self.clustering_enabled_check.isChecked() and self._clusters:
            cluster_bounds: list[tuple[int, float, float, float, float]] = []
            for cluster in self._clusters:
                roi = cluster.roi
                if roi is None:
                    continue
                rect = roi.parentBounds()
                if rect is None:
                    continue
                min_x = float(rect.left())
                max_x = float(rect.right())
                if max_x < min_x:
                    min_x, max_x = max_x, min_x
                min_y = float(rect.top())
                max_y = float(rect.bottom())
                if max_y < min_y:
                    min_y, max_y = max_y, min_y
                cluster_bounds.append((cluster.id, min_x, max_x, min_y, max_y))
            if cluster_bounds:
                for idx, event_id in enumerate(event_ids):
                    if event_id is None:
                        continue
                    x_val = xs[idx]
                    y_val = ys[idx]
                    for cluster_id, min_x, max_x, min_y, max_y in cluster_bounds:
                        if min_x <= x_val <= max_x and min_y <= y_val <= max_y:
                            self._event_cluster_labels[event_id] = cluster_id
                            break
                counts: dict[int, int] = {cluster.id: 0 for cluster in self._clusters}
                for cid in self._event_cluster_labels.values():
                    counts[cid] = counts.get(cid, 0) + 1
                for cluster in self._clusters:
                    item = self._cluster_items.get(cluster.id)
                    if item is None:
                        continue
                    item.setText(f"{cluster.name} ({counts.get(cluster.id, 0)} events)")
                self._refresh_overlay_colors()
        default_brush_color = QtGui.QColor(UNCLASSIFIED_COLOR)
        default_brush_color.setAlpha(170)
        default_brush = pg.mkBrush(default_brush_color)
        brushes: list[pg.mkBrush] = []
        cluster_color_map = {cluster.id: cluster.color for cluster in self._clusters}
        for idx, eid in enumerate(event_ids):
            if eid is None:
                brushes.append(default_brush)
                continue
            cluster_id = self._event_cluster_labels.get(eid)
            if cluster_id is None:
                brushes.append(default_brush)
                continue
            color = cluster_color_map.get(cluster_id)
            if color is None:
                brushes.append(default_brush)
                continue
            brushes.append(pg.mkBrush(color))
        self.energy_scatter.setData(xs, ys, brush=brushes)
        self.energy_scatter.show()
        plot_item = self.metrics_plot.getPlotItem()
        self._set_metrics_range(plot_item, min(xs), max(xs), min(ys), max(ys))

    def _record_overlay_metrics(self, overlay: dict[str, object]) -> None:
        metrics = overlay.get("metrics")
        metric_time = overlay.get("metric_time")
        if metrics is None or metric_time is None:
            return
        if self._t0_event is None:
            self._t0_event = float(metric_time)
        rel_time = max(0.0, float(metric_time) - (self._t0_event or 0.0))
        record: dict[str, float | int] = {"time": rel_time}
        has_metric = False
        for key in ("ed", "max", "min", "freq", "period", "interval"):
            value = metrics.get(key)
            if value is None:
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            record[key] = val
            has_metric = True
        event_id = overlay.get("event_id")
        if isinstance(event_id, int):
            record["event_id"] = event_id
        if not has_metric:
            return
        self._metric_events.append(record)
        if len(self._metric_events) > self._max_metric_events:
            removed = self._metric_events.pop(0)
            removed_id = removed.get("event_id")
            if isinstance(removed_id, int):
                self._event_details.pop(removed_id, None)
                removed_cluster = self._event_cluster_labels.pop(removed_id, None) is not None
                if removed_cluster:
                    self._refresh_overlay_colors()

    def _build_overlay_payload(self, event: Event) -> Optional[OverlayPayload]:
        samples = np.asarray(event.samples, dtype=np.float32)
        if samples.size < 8:
            return None
        sr = float(event.sampleRateHz or 0.0)
        if sr <= 0:
            sr = self.sample_rate if self.sample_rate > 0 else 1.0
        if sr <= 0:
            return None
        pre_samples = max(0, int(round(float(event.preMs) * sr / 1000.0)))
        if pre_samples >= samples.size:
            pre_samples = samples.size - 1
        first_index = int(event.crossingIndex) - pre_samples if event.crossingIndex >= 0 else None
        times = float(event.firstSampleTimeSec) + (np.arange(samples.size, dtype=np.float64) / sr)
        last_time = float(times[-1]) if times.size else float(event.firstSampleTimeSec)
        baseline = _baseline(samples, pre_samples)
        x = samples.astype(np.float32) - baseline
        cross_idx = pre_samples
        search_radius = max(1, int(round(0.001 * sr)))
        i0 = max(0, cross_idx - search_radius)
        i1 = min(samples.size, cross_idx + search_radius + 1)
        if i1 <= i0:
            i1 = min(samples.size, i0 + 1)
        window_slice = x[i0:i1]
        if window_slice.size:
            local_idx = int(np.argmax(np.abs(window_slice)))
            peak_idx = i0 + local_idx
        else:
            peak_idx = cross_idx
        peak_idx = int(np.clip(peak_idx, 0, samples.size - 1))
        peak_time = float(times[peak_idx]) if times.size else float(event.crossingTimeSec)
        metrics: Optional[dict[str, float]] = None
        metric_values: dict[str, float] = {}
        if samples.size >= 4:
            ed = _energy_density(x, sr)
            mx, mn = _min_max(x)
            fpk = _peak_frequency_sinc(x, sr, center_index=cross_idx)
            period = 1.0 / fpk if fpk > 1e-9 else 0.0
            metric_values.update(
                {
                    "ed": float(ed),
                    "max": float(mx),
                    "min": float(mn),
                    "freq": float(fpk),
                    "period": float(period),
                }
            )
        interval_val = float(getattr(event, "intervalSinceLastSec", float("nan")))
        if not np.isfinite(interval_val):
            props = getattr(event, "properties", None)
            if isinstance(props, dict):
                try:
                    interval_candidate = props.get("interval_sec")
                    if interval_candidate is not None:
                        interval_val = float(interval_candidate)
                except (TypeError, ValueError):
                    interval_val = float("nan")
        if np.isfinite(interval_val) and interval_val >= 0.0:
            metric_values["interval"] = float(interval_val)
        if metric_values:
            metrics = metric_values
        raw_event_id = getattr(event, "id", None)
        if isinstance(raw_event_id, int):
            event_id = raw_event_id
        else:
            event_id = self._next_event_id
            self._next_event_id += 1
        return OverlayPayload(
            event_id=event_id,
            times=times,
            samples=samples,
            last_time=last_time,
            first_index=first_index,
            sr=sr,
            pre_samples=pre_samples,
            baseline=baseline,
            peak_idx=peak_idx,
            peak_time=peak_time,
            metrics=metrics,
            metric_time=peak_time,
        )

    def _materialize_overlay(
        self,
        payload: OverlayPayload,
        reuse_overlay: Optional[dict[str, object]] = None,
    ) -> Optional[dict[str, object]]:
        curve: Optional[pg.PlotCurveItem] = None
        overlay_data = reuse_overlay if reuse_overlay is not None else {}
        if reuse_overlay is not None:
            item = reuse_overlay.get("item")
            if isinstance(item, pg.PlotCurveItem):
                curve = item
        if curve is None:
            curve = self._acquire_overlay_item()
        overlay_data.update(
            {
                "item": curve,
                "times": payload.times,
                "samples": payload.samples,
                "last_time": payload.last_time,
                "first_index": payload.first_index,
                "sr": payload.sr,
                "pre_samples": payload.pre_samples,
                "baseline": payload.baseline,
                "peak_idx": payload.peak_idx,
                "peak_time": payload.peak_time,
                "metrics": payload.metrics,
                "metric_time": payload.metric_time,
                "event_id": payload.event_id,
            }
        )
        self._apply_overlay_color(overlay_data)
        details_entry: dict[str, object] = {
            "metric_time": float(payload.metric_time),
            "times": payload.times,
            "samples": payload.samples,
            "metrics": dict(payload.metrics) if isinstance(payload.metrics, dict) else {},
        }
        if isinstance(payload.event_id, int):
            self._event_details[payload.event_id] = details_entry
        return overlay_data

    def _apply_overlay_view(
        self,
        overlay: dict[str, object],
        window_start_time: float,
        width: float,
        window_start_idx: Optional[int],
    ) -> bool:
        times = overlay.get("times")
        samples = overlay.get("samples")
        item = overlay.get("item")
        if times is None or samples is None or item is None:
            return False
        arr_samples = np.asarray(samples, dtype=np.float32)
        relative: Optional[np.ndarray] = None
        first_index = overlay.get("first_index")
        sample_dt = float(self._dt) if self._dt > 0 else float(overlay.get("sr") or 0.0)
        if (
            window_start_idx is not None
            and first_index is not None
            and sample_dt > 0
        ):
            start_diff = int(first_index) - int(window_start_idx)
            offsets = np.arange(arr_samples.size, dtype=np.float64) + start_diff
            relative = offsets * sample_dt
        if relative is None:
            arr_times = np.asarray(times, dtype=np.float64)
            relative = arr_times - window_start_time
        mask = (relative >= 0.0) & (relative <= width)
        if not np.any(mask):
            return False
        item.setData(relative[mask].astype(np.float32), arr_samples[mask])
        return True

    def _on_cluster_roi_changed(self) -> None:
        self._recompute_cluster_membership()
        self._update_metric_points()

    def _recompute_cluster_membership(self) -> None:
        if not self._clusters:
            self._event_cluster_labels.clear()
            self._refresh_overlay_colors()
            return
        x_key = self._selected_x_metric()
        y_key = self._selected_y_metric()
        cluster_bounds: list[tuple[int, float, float, float, float]] = []
        for cluster in self._clusters:
            roi = cluster.roi
            if roi is None:
                continue
            rect = roi.parentBounds()
            if rect is None:
                continue
            min_x = float(rect.left())
            max_x = float(rect.right())
            if max_x < min_x:
                min_x, max_x = max_x, min_x
            min_y = float(rect.top())
            max_y = float(rect.bottom())
            if max_y < min_y:
                min_y, max_y = max_y, min_y
            cluster_bounds.append((cluster.id, min_x, max_x, min_y, max_y))
        self._event_cluster_labels.clear()
        if not cluster_bounds:
            self._refresh_overlay_colors()
            return
        counts: dict[int, int] = {cluster.id: 0 for cluster in self._clusters}
        for record in self._metric_events:
            event_id = record.get("event_id")
            if not isinstance(event_id, int):
                continue
            x_val = record.get(x_key)
            y_val = record.get(y_key)
            if x_val is None or y_val is None:
                continue
            try:
                x_num = float(x_val)
                y_num = float(y_val)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(x_num) and np.isfinite(y_num)):
                continue
            for cluster_id, min_x, max_x, min_y, max_y in cluster_bounds:
                if min_x <= x_num <= max_x and min_y <= y_num <= max_y:
                    self._event_cluster_labels[event_id] = cluster_id
                    counts[cluster_id] = counts.get(cluster_id, 0) + 1
                    break
        for cluster in self._clusters:
            item = self._cluster_items.get(cluster.id)
            if item is None:
                continue
            count = counts.get(cluster.id, 0)
            item.setText(f"{cluster.name} ({count} events)")
        self._refresh_overlay_colors()


class ClusterWaveformDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        class_name: str,
        waveforms: Sequence[tuple[np.ndarray, np.ndarray]],
        color: Optional[QtGui.QColor] = None,
    ) -> None:
        super().__init__(parent)
        count = len(waveforms)
        self.setWindowTitle(f"Waveforms \u2013 {class_name} ({count} events)")
        layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(enableMenu=False)
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "Amplitude (V)")
        plot_item = self.plot_widget.getPlotItem()
        plot_item.showGrid(x=True, y=True, alpha=0.15)
        for axis_name in ("bottom", "left"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen("black"))
            axis.setTextPen(pg.mkPen("black"))
        layout.addWidget(self.plot_widget, 1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        line_pen = pg.mkPen(QtGui.QColor(140, 140, 140, 90), width=1)
        lengths: list[int] = []
        trimmed_waveforms: list[tuple[np.ndarray, np.ndarray]] = []
        for times, samples in waveforms:
            length = int(min(times.size, samples.size))
            if length <= 0:
                continue
            trimmed_waveforms.append((times[:length], samples[:length]))
            lengths.append(length)
            self.plot_widget.plot(times, samples, pen=line_pen, clear=False)
        if not trimmed_waveforms:
            return
        length_counts = Counter(lengths)
        target_len = max(length_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        aligned = [(times[:target_len], samples[:target_len]) for times, samples in trimmed_waveforms if times.size >= target_len and samples.size >= target_len]
        if not aligned:
            return
        stack = np.stack([samples for _, samples in aligned], axis=0)
        median_wave = np.median(stack, axis=0)
        t_ref = aligned[0][0]
        median_pen = pg.mkPen(color if color is not None else QtGui.QColor(0, 0, 200), width=3)
        self.plot_widget.plot(t_ref, median_wave, pen=median_pen)
        self.plot_widget.setAntialiasing(False)
