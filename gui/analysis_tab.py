from __future__ import annotations

import csv
import logging
import queue
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Sequence

import numpy as np

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui

logger = logging.getLogger(__name__)


from analysis.models import AnalysisBatch
from shared.models import Chunk, EndOfStream
from shared.event_buffer import AnalysisEvents
from shared.types import AnalysisEvent


from gui.analysis.helpers import (
    CLUSTER_COLORS,
    UNCLASSIFIED_COLOR,
    WAVEFORM_MEDIAN_COLOR,
    MAX_VISIBLE_METRIC_EVENTS,
    METRIC_TIME_WINDOW_SEC,
    STA_TRACE_PEN,
    SCOPE_BACKGROUND_COLOR,
    _MeasureLine,
    ClusterRectROI,
    MetricCluster,
    OverlayPayload,
    StaTask,
    AnalysisUpdate,
)

from analysis.metrics import baseline, energy_density, min_max, peak_frequency_sinc

if TYPE_CHECKING:
    from analysis.analysis_worker import AnalysisWorker
    from analysis.settings import AnalysisSettingsStore
    from core.controller import PipelineController
    from shared.event_buffer import EventRingBuffer
    from daq.base_device import ChannelInfo

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
        self._metric_events: deque[dict[str, float | int]] = deque(maxlen=100_000)
        self._metrics_dirty: bool = False
        # Performance optimization: track state to skip unnecessary updates
        self._last_scatter_count: int = 0
        self._cluster_membership_dirty: bool = True  # Recompute classification on ROI change only
        self._brush_cache: dict[int, object] = {}  # cluster_id -> cached brush object
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
        self._unclassified_item: QtWidgets.QListWidgetItem | None = None  # Permanent "Unclassified" entry
        self._UNCLASSIFIED_ID: int = -1  # Special ID for Unclassified pseudo-class
        # Pause snapshot: when paused, we capture static curve data and colors
        self._pause_snapshot_curves: list[pg.PlotCurveItem] = []  # Static snapshot items
        self._sta_enabled: bool = False
        self._sta_windows: list[np.ndarray] = []
        self._sta_max_windows = float("inf")
        self._sta_update_interval_ms: int = 100
        self._sta_dirty: bool = False
        self._sta_time_axis: np.ndarray | None = None
        self._sta_aligned_windows: np.ndarray | None = None
        self._sta_window_ms: float = 50.0
        self._sta_source_cluster_id: int | None = None
        self._sta_target_channel_id: int | None = None
        self._sta_pending_events: dict[int, tuple[AnalysisEvent, int]] = {}
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
        except Exception as e:
            logger.debug("Failed to hide plot buttons: %s", e)
        self.plot_widget.setBackground(SCOPE_BACKGROUND_COLOR)
        self.plot_widget.setAntialiasing(False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Amplitude", units="V")
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, 1.0, padding=0.0)
        plot_item.setYRange(-1.0, 1.0, padding=0.0)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.pause_viz_btn = QtWidgets.QPushButton("Pause Display")
        self.pause_viz_btn.setCheckable(True)
        self.pause_viz_btn.setToolTip("Pause/resume updating the raw trace (analysis continues).")
        self.pause_viz_btn.toggled.connect(self._on_pause_viz_toggled)
        plot_container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(plot_container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(self.plot_widget, 0, 0)
        overlay = QtWidgets.QWidget()
        overlay_layout = QtWidgets.QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(8, 0, 0, 8)
        overlay_layout.setSpacing(0)
        overlay_layout.addWidget(self.pause_viz_btn, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        overlay_layout.addStretch(1)
        grid.addWidget(overlay, 0, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.raw_row_layout.addWidget(plot_container, stretch=7)

        self.sta_plot = pg.PlotWidget(enableMenu=False)
        try:
            self.sta_plot.hideButtons()
        except Exception as e:
            logger.debug("Failed to hide STA plot buttons: %s", e)
        self.sta_plot.setBackground(SCOPE_BACKGROUND_COLOR)
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
        self._sta_max_traces: int = 100

        layout.addWidget(self.raw_row_widget, stretch=4)
        self._hide_sta_plot()

        self.metrics_plot = pg.PlotWidget(enableMenu=False)
        try:
            self.metrics_plot.hideButtons()
        except Exception as e:
            logger.debug("Failed to hide metrics plot buttons: %s", e)
        self.metrics_plot.setBackground(SCOPE_BACKGROUND_COLOR)
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
        # Disable mouse event interception so cluster ROI handles can receive clicks
        # This fixes the issue where scatter points block ROI edge/handle dragging
        self.energy_scatter.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self.energy_scatter.setZValue(-10)  # Below ROI items (z=50) for visual order
        self.metrics_plot.addItem(self.energy_scatter)
        self.energy_scatter.hide()

        controls = QtWidgets.QGroupBox("Display")
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(8, 6, 8, 8)
        controls_layout.setSpacing(10)

        size_layout = QtWidgets.QVBoxLayout()
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(3)

        width_row = QtWidgets.QHBoxLayout()
        width_row.setSpacing(4)
        width_row.addWidget(QtWidgets.QLabel("Width (s)"))
        self.width_combo = QtWidgets.QComboBox()
        self.width_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.width_combo.setFixedWidth(72)
        for value in (0.2, 0.5, 1.0, 2.0, 5.0):
            self.width_combo.addItem(f"{value:.1f}", value)
        self.width_combo.setCurrentIndex(1)
        width_row.addWidget(self.width_combo)
        size_layout.addLayout(width_row)

        height_row = QtWidgets.QHBoxLayout()
        height_row.setSpacing(4)
        height_row.addWidget(QtWidgets.QLabel("Height (±V)"))
        self.height_combo = QtWidgets.QComboBox()
        self.height_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.height_combo.setFixedWidth(72)
        for value in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
            self.height_combo.addItem(f"{value:.1f}", value)
        self.height_combo.setCurrentIndex(3)
        height_row.addWidget(self.height_combo)
        size_layout.addLayout(height_row)

        event_window_row = QtWidgets.QHBoxLayout()
        event_window_row.setSpacing(4)
        event_window_row.addWidget(QtWidgets.QLabel("Event Width (ms)"))
        self.event_window_combo = QtWidgets.QComboBox()
        self.event_window_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.event_window_combo.setFixedWidth(72)
        for label, value in (("5", 5.0), ("10", 10.0), ("20", 20.0)):
            self.event_window_combo.addItem(label, value)
        self._set_event_window_selection(self._event_window_ms)
        event_window_row.addWidget(self.event_window_combo)
        size_layout.addLayout(event_window_row)
        controls_layout.addLayout(size_layout)

        threshold_layout = QtWidgets.QVBoxLayout()
        threshold_layout.setSpacing(4)

        self.auto_detect_check = QtWidgets.QCheckBox("Auto-detect events (5\u03c3)")
        self.auto_detect_check.setToolTip("Automatically detect events crossing 5 * MAD threshold (positive or negative).")
        self.auto_detect_check.toggled.connect(self._on_auto_detect_toggled)
        threshold_layout.addWidget(self.auto_detect_check)

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
        metric_model = QtGui.QStandardItemModel(self.metric_combo)
        for label in (
            "Max in window (V)",
            "Min in window (V)",
            "Energy Density",
            "Peak Frequency (Hz)",
            "Interval since last event (s)",
        ):
            item = QtGui.QStandardItem(label)
            metric_model.appendRow(item)
        self.metric_combo.setModel(metric_model)
        self.metric_combo.setCurrentIndex(0)
        self.metric_combo.setMinimumWidth(180)
        self.metric_combo.currentIndexChanged.connect(self._on_axis_metric_changed)
        metric_row.addWidget(self.metric_combo)
        metrics_layout.addLayout(metric_row)

        xaxis_row = QtWidgets.QHBoxLayout()
        xaxis_row.setSpacing(6)
        xaxis_row.addWidget(QtWidgets.QLabel("Horizontal (X) Axis"))
        self.metric_xaxis_combo = QtWidgets.QComboBox()
        metric_x_model = QtGui.QStandardItemModel(self.metric_xaxis_combo)
        for label in (
            "Time (s)",
            "Max in window (V)",
            "Min in window (V)",
            "Energy Density",
            "Peak Frequency (Hz)",
        ):
            item = QtGui.QStandardItem(label)
            metric_x_model.appendRow(item)
        self.metric_xaxis_combo.setModel(metric_x_model)
        self.metric_xaxis_combo.setMinimumWidth(180)
        self.metric_xaxis_combo.currentIndexChanged.connect(self._on_axis_metric_changed)
        xaxis_row.addWidget(self.metric_xaxis_combo)
        metrics_layout.addLayout(xaxis_row)

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.setSpacing(6)
        self.metrics_clear_btn = QtWidgets.QPushButton("Clear metrics (0)")
        self.metrics_clear_btn.setFixedWidth(150)
        self.metrics_clear_btn.clicked.connect(self._clear_metrics)
        buttons_row.addWidget(self.metrics_clear_btn)
        buttons_row.addStretch(1)
        metrics_layout.addLayout(buttons_row)
        self.clustering_enabled_check = QtWidgets.QCheckBox("Enable clustering")
        self.clustering_enabled_check.toggled.connect(self._on_clustering_toggled)
        metrics_layout.addWidget(self.clustering_enabled_check)
        metrics_layout.addStretch(1)

        controls_layout.addLayout(metrics_layout)

        sta_layout = QtWidgets.QVBoxLayout()
        sta_layout.setSpacing(4)

        self.sta_enable_check = QtWidgets.QCheckBox("Enable Spike Triggered Average (STA)")
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

        sta_buttons_row = QtWidgets.QHBoxLayout()
        sta_buttons_row.setSpacing(6)
        self.sta_clear_btn = QtWidgets.QPushButton("Clear STA")
        self.sta_clear_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.sta_clear_btn.clicked.connect(self._on_sta_clear_clicked)
        sta_buttons_row.addWidget(self.sta_clear_btn, 1)
        self.sta_view_waveforms_btn = QtWidgets.QPushButton("View waveforms…")
        self.sta_view_waveforms_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.sta_view_waveforms_btn.clicked.connect(self._on_sta_view_waveforms_clicked)
        self.sta_view_waveforms_btn.setEnabled(False)
        sta_buttons_row.addWidget(self.sta_view_waveforms_btn, 1)
        sta_layout.addLayout(sta_buttons_row)

        controls_layout.addLayout(sta_layout)
        # Note: No addStretch here - let widgets fill the available space

        controls.setLayout(controls_layout)
        # Controls panel has fixed content height - prevent vertical expansion
        controls.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        layout.addWidget(controls, stretch=0)
        self._refresh_cluster_options()

        self.metrics_container = QtWidgets.QWidget(self)
        # Set expanding policy to fill remaining vertical space
        self.metrics_container.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Expanding
        )
        metrics_container_layout = QtWidgets.QHBoxLayout(self.metrics_container)
        metrics_container_layout.setContentsMargins(0, 0, 0, 0)
        metrics_container_layout.setSpacing(10)
        metrics_container_layout.addWidget(self.metrics_plot, stretch=1)
        self.metrics_container_layout = metrics_container_layout

        self.cluster_panel = QtWidgets.QGroupBox("Event Clusters")
        cluster_layout = QtWidgets.QVBoxLayout()
        cluster_layout.setContentsMargins(12, 10, 12, 10)
        cluster_layout.setSpacing(6)
        
        # Add Class and Remove Class buttons side-by-side
        add_remove_row = QtWidgets.QHBoxLayout()
        add_remove_row.setSpacing(6)
        self.add_class_btn = QtWidgets.QPushButton("Add class")
        self.add_class_btn.clicked.connect(self._on_add_class_clicked)
        add_remove_row.addWidget(self.add_class_btn, stretch=1)
        self.remove_class_btn = QtWidgets.QPushButton("Remove class")
        self.remove_class_btn.clicked.connect(self._on_remove_class_clicked)
        add_remove_row.addWidget(self.remove_class_btn, stretch=1)
        cluster_layout.addLayout(add_remove_row)
        
        # Export row
        export_row = QtWidgets.QHBoxLayout()
        export_row.setSpacing(6)
        self.export_class_btn = QtWidgets.QPushButton("Export to CSV")
        self.export_class_btn.clicked.connect(self._on_export_class_clicked)
        export_row.addWidget(self.export_class_btn, stretch=1)
        
        self.export_class_combo = QtWidgets.QComboBox()
        self.export_class_combo.addItem("All events", None)
        export_row.addWidget(self.export_class_combo, stretch=1)
        cluster_layout.addLayout(export_row)
        
        self.view_class_waveforms_btn = QtWidgets.QPushButton("View waveforms")
        self.view_class_waveforms_btn.clicked.connect(self._on_view_class_waveforms_clicked)
        cluster_layout.addWidget(self.view_class_waveforms_btn)
        
        # Class list - gets more room now
        self.class_list = QtWidgets.QListWidget()
        self.class_list.currentItemChanged.connect(self._on_class_selection_changed)
        cluster_layout.addWidget(self.class_list, stretch=2)
        self.cluster_panel.setLayout(cluster_layout)
        metrics_container_layout.addWidget(self.cluster_panel, stretch=0)
        self._set_cluster_panel_visible(False)

        layout.addWidget(self.metrics_container, stretch=6)

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
        # Note: No addStretch at end - let metrics_container fill remaining space

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
        """Clean up analysis executor threads on window close."""
        executor = getattr(self, "_analysis_executor", None)
        if executor is not None:
            try:
                for fut in list(self._analysis_futures):
                    fut.cancel()
                executor.shutdown(wait=False)
            except Exception as e:
                logger.debug("Exception shutting down analysis executor: %s", e)
            self._analysis_executor = None
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _init_buffer(self) -> None:
        """Initialize the circular sample buffer based on current sample rate."""
        samples = max(1, int(round(self.sample_rate * self._buffer_span_sec))) if self.sample_rate > 0 else 1
        self._buffer = np.zeros(samples, dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_filled = 0

    def _ensure_buffer_capacity(self, required_sec: float) -> None:
        """Expand the circular buffer if needed to hold required_sec of data."""
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
        """Append samples to the circular buffer, wrapping at capacity."""
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
        """Return the most recent 'count' samples from the circular buffer."""
        if self._buffer_filled == 0 or count <= 0:
            return np.empty(0, dtype=np.float32)
        count = min(count, self._buffer_filled)
        start = (self._buffer_pos - count) % self._buffer.size
        if start + count <= self._buffer.size:
            return self._buffer[start:start + count].copy()
        first = self._buffer.size - start
        return np.concatenate((self._buffer[start:], self._buffer[:count - first]))

    def _extract_recent(self, count: int) -> np.ndarray:
        """Extract recent samples, padding with NaN if buffer has fewer than 'count'."""
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
        """Generate the window title from channel name and sample rate."""
        return f"{self.channel_name} \u2013 {self.sample_rate:,.0f} Hz"

    def set_channel_info(self, channel_name: str, sample_rate: float) -> None:
        """Update channel name and sample rate, reinitializing buffers."""
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
        """Set the queue from which analysis batches are consumed."""
        self._analysis_queue = q

    def set_worker(self, worker: "AnalysisWorker") -> None:
        """Bind the analysis worker for threshold configuration."""
        self._worker = worker
        if self._worker is not None and self.sample_rate > 0:
            try:
                self._worker.update_sample_rate(self.sample_rate)
            except AttributeError:
                pass
        self._notify_threshold_change()

    def set_sta_channels(self, channels: Sequence["ChannelInfo"]) -> None:
        """Update the list of available STA signal channels."""
        self._refresh_sta_channel_options(channels)

    def peek_all_events(self) -> list[AnalysisEvent]:
        """Return all events in the buffer without removing them."""
        if self._event_buffer is None:
            return []
        return self._event_buffer.peek_all()

    def drain_events(self) -> list[AnalysisEvent]:
        """Remove and return all events from the buffer."""
        if self._event_buffer is None:
            return []
        return self._event_buffer.drain()

    def _toggle_threshold(self, line: pg.InfiniteLine, spin: QtWidgets.QDoubleSpinBox, checked: bool) -> None:
        """Show/hide a threshold line and sync it with the spinbox value."""
        line.setVisible(checked)
        if checked:
            self._in_threshold_update = True
            line.setValue(float(spin.value()))
            self._in_threshold_update = False
            line.setZValue(20)

    def _update_threshold_from_spin(self, line: pg.InfiniteLine, value: float) -> None:
        """Update threshold line position when spinbox value changes."""
        if not line.isVisible() or self._in_threshold_update:
            return
        self._in_threshold_update = True
        line.setValue(float(value))
        self._in_threshold_update = False

    def _update_spin_from_line(self, line: pg.InfiniteLine, spin: QtWidgets.QDoubleSpinBox) -> None:
        """Update spinbox value when threshold line is dragged."""
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
        """Timer callback: drain analysis queue and render batches."""
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
        """Timer callback: refresh metric scatter plots if dirty."""
        if not self._metrics_dirty:
            return
        self._update_metric_points()
        self._metrics_dirty = False
    def _on_sta_timer(self) -> None:
        """Timer callback: refresh STA plot if enabled and dirty."""
        if not self._sta_enabled or not self._sta_dirty:
            return
        self._refresh_sta_plot()
        self._sta_dirty = False

    def _apply_ranges(self) -> None:
        """Apply width/height settings to the plot axes."""
        width = float(self.width_combo.currentData() or 0.5)
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, width, padding=0.0)
        if self._selected_y_metric() in {"ed", "max", "min", "freq", "interval"}:
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
        """Get initial event window duration from settings or default to 5ms."""
        if self._analysis_settings is None:
            return 5.0
        try:
            return float(self._analysis_settings.get().event_window_ms)
        except Exception as e:
            logger.debug("Failed to get event_window_ms from settings: %s", e)
            return 5.0

    def _set_event_window_selection(self, value_ms: float) -> None:
        """Set the event window combo to match value_ms."""
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
        """Handle event window combo selection change."""
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
        """Handle threshold 1 checkbox toggle, update dependent controls."""
        self.threshold2_check.setEnabled(checked)
        self.threshold2_spin.setEnabled(checked and self.threshold2_check.isChecked())
        if not checked:
            self.threshold2_check.setChecked(False)
        self._notify_threshold_change()

    def _on_threshold2_toggled(self, checked: bool) -> None:
        """Handle threshold 2 checkbox toggle."""
        self.threshold2_spin.setEnabled(checked)
        self._notify_threshold_change()

    def _on_auto_detect_toggled(self, checked: bool) -> None:
        """Handle auto-detect checkbox toggle, disable manual thresholds."""
        # Disable manual thresholds if auto-detect is on?
        # User said "add a third checkbox above...".
        # Let's disable manual controls to avoid confusion.
        self.threshold1_check.setEnabled(not checked)
        self.threshold1_spin.setEnabled(not checked and self.threshold1_check.isChecked())
        self.threshold2_check.setEnabled(not checked and self.threshold1_check.isChecked())
        self.threshold2_spin.setEnabled(not checked and self.threshold2_check.isChecked())
        self._notify_threshold_change()

    def _notify_threshold_change(self) -> None:
        """Push current threshold settings to the analysis worker."""
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
                auto_detect=self.auto_detect_check.isChecked(),
            )
        except Exception as e:
            logger.debug("Failed to configure worker threshold: %s", e)

    def _render_batch(self, batch: AnalysisBatch) -> None:
        """Process an analysis batch: update plots, overlays, and events."""
        chunk = batch.chunk
        try:
            data = np.array(chunk.samples, dtype=np.float32)
        except Exception as e:
            logger.debug("Failed to convert chunk samples: %s", e)
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
        except Exception as e:
            logger.debug("Failed to get chunk.dt: %s", e)
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
        """Handle visualization pause checkbox toggle.
        
        When paused: capture a static snapshot of the current display state
        and show that instead of dynamic items.
        When unpaused: clear the snapshot and restore dynamic visualization.
        """
        self._viz_paused = bool(checked)
        if self._viz_paused:
            self._capture_pause_snapshot()
        else:
            self._clear_pause_snapshot()
            self._refresh_raw_plot()
            self._refresh_overlay_positions(self._last_window_start, self._last_window_width, self._window_start_index)

    def _capture_pause_snapshot(self) -> None:
        """Capture a static snapshot of current curves for frozen display."""
        # Create static snapshot of raw trace
        if self._cached_raw_times is not None and self._cached_raw_samples is not None:
            raw_snapshot = pg.PlotCurveItem(
                self._cached_raw_times.copy(),
                self._cached_raw_samples.copy(),
                pen=self.raw_curve.opts.get("pen", pg.mkPen("b")),
            )
            raw_snapshot.setZValue(self.raw_curve.zValue())
            self.plot_widget.addItem(raw_snapshot)
            self._pause_snapshot_curves.append(raw_snapshot)
        
        # Create static snapshots of all overlay curves with their current colors
        # BEFORE hiding them so visibility check works
        for overlay in self._event_overlays:
            item = overlay.get("item")
            if not isinstance(item, pg.PlotCurveItem):
                continue
            # Get current data and pen from the overlay item
            x_data, y_data = item.getData()
            if x_data is None or y_data is None or len(x_data) == 0:
                continue
            overlay_snapshot = pg.PlotCurveItem(
                x_data.copy() if hasattr(x_data, "copy") else np.array(x_data),
                y_data.copy() if hasattr(y_data, "copy") else np.array(y_data),
                pen=item.opts.get("pen", self._overlay_pen),
            )
            overlay_snapshot.setZValue(item.zValue())
            self.plot_widget.addItem(overlay_snapshot)
            self._pause_snapshot_curves.append(overlay_snapshot)
        
        # Now hide the dynamic items
        self.raw_curve.hide()
        for overlay in self._event_overlays:
            item = overlay.get("item")
            if isinstance(item, pg.PlotCurveItem):
                item.hide()
    
    def _clear_pause_snapshot(self) -> None:
        """Remove all pause snapshot curves and restore dynamic items."""
        # Remove snapshot curves
        for curve in self._pause_snapshot_curves:
            try:
                self.plot_widget.removeItem(curve)
            except Exception as e:
                logger.debug("Failed to remove pause snapshot curve: %s", e)
        self._pause_snapshot_curves.clear()
        
        # Show the dynamic items again
        self.raw_curve.show()
        for overlay in self._event_overlays:
            item = overlay.get("item")
            if isinstance(item, pg.PlotCurveItem):
                item.show()

    def _refresh_raw_plot(self) -> None:
        """Redraw the raw signal curve from cached data."""
        if self._viz_paused:
            return
        if self._cached_raw_times is None or self._cached_raw_samples is None:
            return
        self.raw_curve.setData(self._cached_raw_times, self._cached_raw_samples, skipFiniteCheck=True)
        self.event_curve.clear()

    def _acquire_overlay_item(self) -> pg.PlotCurveItem:
        """Get a PlotCurveItem from the pool or create a new one."""
        if self._overlay_pool:
            item = self._overlay_pool.pop()
        else:
            item = pg.PlotCurveItem(pen=self._overlay_pen)
            item.setZValue(30)
            self.plot_widget.addItem(item)
        item.show()
        return item

    def _release_overlay_item(self, item: Optional[pg.PlotCurveItem]) -> None:
        """Return an overlay item to the pool for reuse."""
        if item is None:
            return
        item.hide()
        item.setData([], [])
        self._overlay_pool.append(item)

    def _apply_overlay_color(self, overlay: dict[str, object]) -> None:
        """Set overlay curve color based on event cluster membership."""
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
                except Exception as e:
                    logger.debug("Failed to get pen width: %s", e)
                    width = 2.0
            else:
                try:
                    width = float(base_pen.width())
                except Exception as e:
                    logger.debug("Failed to get pen width fallback: %s", e)
                    width = 2.0
        item.setPen(pg.mkPen(color, width=width))
    # Note: _refresh_overlay_colors was removed - overlay colors are set once at creation

    def _handle_batch_events(
        self,
        events: Sequence[AnalysisEvent],
        window_start: float,
        width: float,
        window_start_idx: Optional[int],
    ) -> None:
        """Process detected events: submit for analysis and update overlays."""
        pending_sta = bool(getattr(self, "_sta_pending_events", None))
        if events:
            self._submit_analysis_job(tuple(events), window_start, width, window_start_idx)
        elif self._sta_enabled and pending_sta:
            task = self._build_sta_task(())
            if task is not None:
                self._sta_handle_task(task)
        if not self._viz_paused:
            self._refresh_overlay_positions(window_start, width, window_start_idx)

    def _build_sta_task(self, events: Sequence[AnalysisEvent]) -> StaTask | None:
        """Create an STA task for the given events if STA is enabled."""
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
        events: tuple[AnalysisEvent, ...],
        window_start: float,
        width: float,
        window_start_idx: Optional[int],
    ) -> None:
        """Submit events to background executor or process inline."""
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
        """Callback when background analysis completes."""
        self._analysis_futures.discard(future)
        try:
            update = future.result()
        except Exception as exc:
            logging.getLogger(__name__).exception("Analysis worker failed: %s", exc)
            return
        self._apply_analysis_update(update, window_start, width, window_start_idx)

    def _build_analysis_update(self, events: tuple[AnalysisEvent, ...]) -> AnalysisUpdate:
        """Build overlay payloads and STA task for a batch of events."""
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
        """Apply analysis results: materialize overlays and update metrics."""
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
        """Update overlay positions and remove expired ones outside the window."""
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
            # Note: overlay color is set once at creation time, not refreshed per-frame
            kept.append(overlay)
        self._event_overlays = kept

    def _clear_event_overlays(self) -> None:
        """Remove and recycle all event overlay items."""
        if not self._event_overlays:
            return
        for overlay in self._event_overlays:
            self._release_overlay_item(overlay.get("item"))
        self._event_overlays.clear()

    def _ensure_sta_trace_capacity(self, capacity: int) -> None:
        """Ensure enough STA trace items exist for the given capacity."""
        current = len(self._sta_trace_items)
        if current >= capacity:
            return
        for _ in range(capacity - current):
            item = pg.PlotCurveItem(pen=STA_TRACE_PEN)
            item.setZValue(-2)
            self.sta_plot.addItem(item)
            self._sta_trace_items.append(item)

    def _clear_sta_traces(self) -> None:
        """Hide and clear all STA trace plot items."""
        for item in self._sta_trace_items:
            item.hide()
            item.setData([], [])

    def _clear_metrics(self) -> None:
        """Reset all metric data, scatter plots, and cluster counts."""
        self._metric_events.clear()
        self._event_details.clear()
        self._event_cluster_labels.clear()
        # Reset optimization tracking
        self._last_scatter_count = 0
        self._cluster_membership_dirty = True
        # Note: overlay colors are set at creation time, not refreshed
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
        self.metrics_clear_btn.setText("Clear metrics (0)")

    def get_energy_density_points(self) -> list[tuple[float, float]]:
        """Return (time, energy_density) pairs for all recorded events."""
        return [
            (event["time"], event["ed"])
            for event in self._metric_events
            if "time" in event and "ed" in event
        ]

    def _set_metrics_range(self, plot_item: pg.PlotItem, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        """Set the X and Y ranges for the metrics scatter plot with padding."""
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
        """Parse the Y-axis metric combo selection to a metric key."""
        label = self.metric_combo.currentText().lower()
        if "interval" in label:
            return "interval"
        if "max" in label:
            return "max"
        if "min" in label:
            return "min"
        if "frequency" in label:
            return "freq"
        return "ed"

    def _selected_x_metric(self) -> str:
        """Parse the X-axis metric combo selection to a metric key."""
        label = self.metric_xaxis_combo.currentText().lower()
        if "time" in label:
            return "time"
        if "frequency" in label:
            return "freq"
        if "max" in label:
            return "max"
        if "min" in label:
            return "min"
        return "ed"

    def _on_axis_metric_changed(self) -> None:
        """Handle axis metric combo changes and update plot labels."""
        metric = self._selected_y_metric()
        if metric == "ed":
            self.metrics_plot.setLabel("left", "Energy Density")
        elif metric == "max":
            self.metrics_plot.setLabel("left", "Max Amplitude (V)")
        elif metric == "min":
            self.metrics_plot.setLabel("left", "Min Amplitude (V)")
        elif metric == "freq":
            self.metrics_plot.setLabel("left", "Peak Frequency (Hz)")
        elif metric == "interval":
            self.metrics_plot.setLabel("left", "Interval (s)")
        else:
            self.metrics_plot.setLabel("left", "Value")
        x_metric = self._selected_x_metric()
        if x_metric == "time":
            self.metrics_plot.setLabel("bottom", "Time (s)")
        elif x_metric == "ed":
            self.metrics_plot.setLabel("bottom", "Energy Density")
        elif x_metric == "max":
            self.metrics_plot.setLabel("bottom", "Max Amplitude (V)")
        elif x_metric == "min":
            self.metrics_plot.setLabel("bottom", "Min Amplitude (V)")
        elif x_metric == "freq":
            self.metrics_plot.setLabel("bottom", "Peak Frequency (Hz)")
        else:
            self.metrics_plot.setLabel("bottom", "Energy Density")
        if metric in {"ed", "max", "min", "freq", "interval"}:
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
        """Handle cluster list selection change."""
        del previous
        if current is None:
            self._selected_cluster_id = None
        else:
            data = current.data(QtCore.Qt.UserRole)
            self._selected_cluster_id = int(data) if isinstance(data, int) else None
        self._update_cluster_visuals()
        self._update_cluster_button_states()

    def _update_cluster_visuals(self) -> None:
        """Update cluster ROI pen widths based on selection."""
        base_width = 1.5
        selected_id = self._selected_cluster_id
        for cluster in self._clusters:
            roi = cluster.roi
            if roi is None:
                continue
            width = base_width * 3.0 if cluster.id == selected_id else base_width
            roi.setPen(pg.mkPen(cluster.color, width=width))

    def _update_cluster_button_states(self) -> None:
        """Enable/disable cluster buttons based on state."""
        enabled = self.clustering_enabled_check.isChecked()
        current_item = self.class_list.currentItem()
        has_selection = current_item is not None
        
        # Check if the selected item is the Unclassified entry
        is_unclassified = False
        if has_selection:
            selected_id = current_item.data(QtCore.Qt.UserRole)
            is_unclassified = (selected_id == self._UNCLASSIFIED_ID)
        
        self.add_class_btn.setEnabled(enabled)
        # Remove class only enabled for user-added classes, not Unclassified
        self.remove_class_btn.setEnabled(enabled and has_selection and not is_unclassified)
        self.view_class_waveforms_btn.setEnabled(enabled and has_selection)
        self.export_class_btn.setEnabled(enabled and has_selection)

    def _update_sta_view_button(self) -> None:
        """Enable/disable STA view button based on data availability."""
        has_data = (
            self._sta_enabled
            and self._sta_aligned_windows is not None
            and getattr(self._sta_aligned_windows, "size", 0) > 0
            and self._sta_time_axis is not None
            and self._sta_time_axis.size > 0
        )
        if hasattr(self, "sta_view_waveforms_btn"):
            self.sta_view_waveforms_btn.setEnabled(bool(has_data))

    def _set_cluster_panel_visible(self, visible: bool) -> None:
        """Show/hide the cluster management panel with layout adjustments."""
        if visible:
            self.cluster_panel.show()
            self.metrics_container_layout.setStretchFactor(self.metrics_plot, 7)
            self.metrics_container_layout.setStretchFactor(self.cluster_panel, 3)
        else:
            self.cluster_panel.hide()
            self.metrics_container_layout.setStretchFactor(self.metrics_plot, 1)
            self.metrics_container_layout.setStretchFactor(self.cluster_panel, 0)

    def _refresh_cluster_options(self) -> None:
        """Populate the STA event source and export class combos."""
        # Update STA source combo
        if hasattr(self, "sta_source_combo"):
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

        # Update Export class combo
        if hasattr(self, "export_class_combo"):
            previous_export = self.export_class_combo.currentData()
            was_blocked = self.export_class_combo.blockSignals(True)
            self.export_class_combo.clear()
            self.export_class_combo.addItem("All events", None)
            for cluster in self._clusters:
                self.export_class_combo.addItem(cluster.name, cluster.id)
            target_index = self.export_class_combo.findData(previous_export)
            if target_index >= 0:
                self.export_class_combo.setCurrentIndex(target_index)
            else:
                self.export_class_combo.setCurrentIndex(0)
            self.export_class_combo.blockSignals(was_blocked)

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
        """Show the STA plot widget and adjust layout."""
        self.sta_plot.show()
        self.raw_row_layout.setStretchFactor(self.plot_widget, 7)
        self.raw_row_layout.setStretchFactor(self.sta_plot, 3)

    def _hide_sta_plot(self) -> None:
        """Hide the STA plot widget and restore layout."""
        self.sta_plot.hide()
        self.raw_row_layout.setStretchFactor(self.plot_widget, 10)
        self.raw_row_layout.setStretchFactor(self.sta_plot, 0)

    def _get_cluster_for_event(self, event_id: int | None) -> MetricCluster | None:
        """Look up the cluster containing the given event ID."""
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
        """Handle clustering enable checkbox toggle."""
        self._set_cluster_panel_visible(checked)
        if checked:
            # Add the permanent "Unclassified" entry at the top
            if self._unclassified_item is None:
                self._unclassified_item = QtWidgets.QListWidgetItem("Unclassified (0 events)")
                self._unclassified_item.setData(QtCore.Qt.UserRole, self._UNCLASSIFIED_ID)
            self.class_list.insertItem(0, self._unclassified_item)
            self._recompute_cluster_membership()
        else:
            # Remove the Unclassified entry
            if self._unclassified_item is not None:
                row = self.class_list.row(self._unclassified_item)
                if row >= 0:
                    self.class_list.takeItem(row)
            self._event_cluster_labels.clear()
            for cluster in self._clusters:
                item = self._cluster_items.get(cluster.id)
                if item is not None:
                    item.setText(f"{cluster.name} (0 events)")
        self._update_metric_points()
        self._update_cluster_visuals()
        self._update_cluster_button_states()
        # Note: overlay colors are set at creation time, not refreshed here

    def _on_sta_toggled(self, checked: bool) -> None:
        """Handle STA enable checkbox toggle."""
        self._sta_enabled = bool(checked)
        if self._sta_enabled:
            self._show_sta_plot()
            self._on_sta_clear_clicked()
        else:
            self._hide_sta_plot()
            self._sta_windows.clear()
            self._sta_time_axis = None
            self._sta_aligned_windows = None
            self._sta_pending_events.clear()
            self._sta_dirty = False
            self._sta_median_curve.hide()
            self._sta_median_curve.clear()
            self._clear_sta_traces()
        self._update_sta_view_button()

    def _on_sta_source_changed(self, index: int) -> None:
        """Handle STA event source combo change."""
        if index < 0:
            self._sta_source_cluster_id = None
        else:
            data = self.sta_source_combo.itemData(index)
            self._sta_source_cluster_id = data if isinstance(data, int) else None
        self._on_sta_clear_clicked()

    def _on_sta_channel_changed(self, index: int) -> None:
        """Handle STA signal channel combo change."""
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
        """Handle STA window duration combo change."""
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
        """Reset all STA state and clear traces."""
        self._sta_windows.clear()
        self._sta_time_axis = None
        self._sta_aligned_windows = None
        self._sta_pending_events.clear()
        self._sta_dirty = False
        self._clear_sta_traces()
        self._sta_median_curve.hide()
        self._sta_median_curve.clear()
        self._update_sta_view_button()

    def _sta_handle_task(self, task: StaTask) -> None:
        """Process pending STA events and collect trigger windows."""
        controller = self._controller
        if controller is None:
            return
        target_channel_id = task.target_channel_id
        channel_info = task.channel_index
        pending_items = list(self._sta_pending_events.values())
        self._sta_pending_events.clear()
        queue: list[tuple[AnalysisEvent, int]] = [(event, 0) for event in task.events]
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
        event: AnalysisEvent,
        window_ms: float,
    ) -> str:
        """Collect trigger window for a single event and add to STA."""
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
        # Align all traces so the trigger sample crosses zero
        center_idx = normalized.size // 2
        if 0 <= center_idx < normalized.size:
            normalized = normalized - float(normalized[center_idx])
        self._sta_windows.append(normalized)
        self._sta_dirty = True
        return "added"

    def _refresh_sta_plot(self) -> None:
        """Redraw the spike-triggered average plot."""
        plot_item = self.sta_plot.getPlotItem()
        self._sta_aligned_windows = None
        self._sta_time_axis = None
        if not self._sta_windows:
            self._sta_median_curve.hide()
            self._sta_median_curve.clear()
            self._clear_sta_traces()
            self._update_sta_view_button()
            return
        min_len = min((w.size for w in self._sta_windows if isinstance(w, np.ndarray)), default=0)
        if min_len <= 1:
            self._sta_median_curve.hide()
            self._sta_median_curve.clear()
            self._clear_sta_traces()
            self._update_sta_view_button()
            return
        windows = np.stack([np.asarray(w[:min_len], dtype=np.float32) for w in self._sta_windows], axis=0)
        assert windows.ndim == 2, "STA windows must be 2D (events x samples)"
        num_events, num_samples = windows.shape
        if windows.size == 0:
            self._sta_median_curve.hide()
            self._sta_median_curve.clear()
            self._clear_sta_traces()
            self._update_sta_view_button()
            return
        duration_ms = float(self._sta_window_ms)
        t = np.linspace(-duration_ms / 2.0, duration_ms / 2.0, num_samples, dtype=np.float32)
        assert t.shape[0] == num_samples, "Time axis must align with samples"
        self._sta_time_axis = t
        self._sta_aligned_windows = windows
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
        self._sta_median_curve.setPen(pg.mkPen(200, 0, 0, 255, width=3))
        self._sta_median_curve.show()
        plot_item.setLabel("bottom", "Lag", units="ms")
        plot_item.setLabel("left", "Amplitude", units="mV")
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        plot_item.setXRange(t[0], t[-1], padding=0.0)
        plot_item.setYRange(amp_min, amp_max, padding=0.05)
        self._update_sta_view_button()

    def _build_sta_waveform_payload(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Prepare STA waveforms for export/viewing."""
        if self._sta_aligned_windows is None or self._sta_time_axis is None:
            return []
        if self._sta_time_axis.size == 0 or self._sta_aligned_windows.size == 0:
            return []
        t_sec = np.asarray(self._sta_time_axis, dtype=np.float64) / 1000.0
        waveforms: list[tuple[np.ndarray, np.ndarray]] = []
        for row in self._sta_aligned_windows:
            samples = np.asarray(row, dtype=np.float32)
            length = min(samples.size, t_sec.size)
            if length <= 0:
                continue
            waveforms.append((t_sec[:length].copy(), samples[:length].copy()))
        return waveforms

    def _on_sta_view_waveforms_clicked(self) -> None:
        """Open dialog to view STA waveforms."""
        waveforms = self._build_sta_waveform_payload()
        if not waveforms:
            QtWidgets.QMessageBox.information(self, "Waveforms", "No cross-correlation data available.")
            return
        channel_label = self.sta_channel_combo.currentText().strip() if self.sta_channel_combo.count() else ""
        title = f"Cross correlation \u2013 {channel_label}" if channel_label else "Cross correlation"
        dialog = ClusterWaveformDialog(self, title, waveforms, None)
        dialog.exec()

    def _on_add_class_clicked(self) -> None:
        """Create a new cluster ROI when Add Class clicked."""
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
        self._refresh_cluster_options()
        self._recompute_cluster_membership()
        self._update_metric_points()
        self._update_cluster_visuals()
        self._update_cluster_button_states()

    def _on_remove_class_clicked(self) -> None:
        """Remove the selected cluster and its ROI."""
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
            except Exception as e:
                logger.debug("Failed to remove cluster ROI: %s", e)
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
        self._refresh_cluster_options()
        self._recompute_cluster_membership()
        self._update_metric_points()
        self._update_cluster_visuals()
        self._update_cluster_button_states()

    def _on_export_class_clicked(self) -> None:
        """Export events to CSV based on the selected class."""
        target_cluster_id = self.export_class_combo.currentData()
        
        # Filter events
        events_to_export = []
        for record in self._metric_events:
            event_id = record.get("event_id")
            if event_id is None:
                continue
            
            # Get raw cluster ID
            raw_cluster_id = self._event_cluster_labels.get(event_id) # None if unclassified
            
            # Filter
            if target_cluster_id is not None:
                # Specific class selected
                if raw_cluster_id != target_cluster_id:
                    continue
            
            # Map to user-facing Class ID (0=Unclassified, 1=Class 1, etc.)
            if raw_cluster_id is None:
                user_class_id = 0
            else:
                user_class_id = raw_cluster_id + 1
            
            # Prepare row data
            row = {
                "Class ID": user_class_id,
                "Time (s)": record.get("time"),
                "Max (V)": record.get("max"),
                "Min (V)": record.get("min"),
                "Energy Density": record.get("ed"),
                "Peak Frequency (Hz)": record.get("freq"),
                "Interval (s)": record.get("interval"),
            }
            events_to_export.append(row)
            
        if not events_to_export:
            QtWidgets.QMessageBox.information(self, "Export CSV", "No events found to export for the selected class.")
            return

        # Prompt for file
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Events to CSV", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return
            
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"
            
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "Class ID", 
                    "Time (s)", 
                    "Max (V)", 
                    "Min (V)", 
                    "Energy Density", 
                    "Peak Frequency (Hz)", 
                    "Interval (s)"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(events_to_export)
            QtWidgets.QMessageBox.information(self, "Export CSV", f"Successfully exported {len(events_to_export)} events.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to write CSV file:\n{e}")

    def _on_view_class_waveforms_clicked(self) -> None:
        """Open dialog to view waveforms for the selected cluster."""
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
        
        # Handle Unclassified group
        if cluster_id == self._UNCLASSIFIED_ID:
            class_name = "Unclassified"
            class_color = QtGui.QColor(UNCLASSIFIED_COLOR)
            # Find all events NOT in any cluster
            all_event_ids = {r.get("event_id") for r in self._metric_events if isinstance(r.get("event_id"), int)}
            classified_ids = set(self._event_cluster_labels.keys())
            event_ids = list(all_event_ids - classified_ids)
        else:
            # Find the cluster
            cluster = next((c for c in self._clusters if c.id == cluster_id), None)
            if cluster is None:
                return
            class_name = cluster.name
            class_color = cluster.color
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
        dialog = ClusterWaveformDialog(self, class_name, waveforms, class_color)
        dialog.exec()

    def _release_metrics(self) -> None:
        """Clear all metric event records."""
        self._metric_events = deque(maxlen=100_000)
        self._t0_event = None
        self._last_scatter_count = 0
        self._cluster_membership_dirty = True
        self._update_metric_points()

    def _update_metric_points(self) -> None:
        """Refresh the scatter plot with current metric data.
        
        Performance optimizations:
        - Skip update if event count unchanged since last call
        - Cache brush objects instead of creating fresh ones per event  
        - Only recompute cluster membership when ROI changes (dirty flag)
        """
        y_key = self._selected_y_metric()
        x_key = self._selected_x_metric()
        if y_key not in {"ed", "max", "min", "freq", "interval"}:
            self.energy_scatter.hide()
            return
        events = self._metric_events
        if not events:
            self.energy_scatter.hide()
            self._last_scatter_count = 0
            return
        
        current_count = len(events)
        
        # Determine visible events based on time window or max count
        visible_events = list(events)  # Convert deque to list for slicing
        if x_key == "time" and visible_events:
            # Find the last valid time
            last_time: float | None = None
            for event in reversed(visible_events):
                t_val = event.get("time")
                if t_val is not None:
                    try:
                        last_time = float(t_val)
                        break
                    except (TypeError, ValueError):
                        continue
            if last_time is not None:
                min_time = last_time - METRIC_TIME_WINDOW_SEC
                if min_time > 0:
                    # Binary search would be more efficient, but this is called infrequently now
                    for idx, event in enumerate(visible_events):
                        t_val = event.get("time")
                        if t_val is not None:
                            try:
                                if float(t_val) >= min_time:
                                    visible_events = visible_events[idx:]
                                    break
                            except (TypeError, ValueError):
                                continue
        
        if len(visible_events) > MAX_VISIBLE_METRIC_EVENTS:
            visible_events = visible_events[-MAX_VISIBLE_METRIC_EVENTS:]
        
        # Build coordinate and event_id lists
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
        
        # Update UI counts (cheap operation)
        self.metrics_clear_btn.setText(f"Clear metrics ({current_count})")
        
        # Only recompute cluster membership if dirty OR if we have new events
        need_reclassify = self._cluster_membership_dirty or (current_count != self._last_scatter_count)
        
        if self.clustering_enabled_check.isChecked() and self._clusters and need_reclassify:
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
                # Only classify events that don't already have a label
                for idx, event_id in enumerate(event_ids):
                    if event_id is None:
                        continue
                    # Skip if already classified and not a dirty full recompute
                    if not self._cluster_membership_dirty and event_id in self._event_cluster_labels:
                        continue
                    x_val = xs[idx]
                    y_val = ys[idx]
                    for cluster_id, min_x, max_x, min_y, max_y in cluster_bounds:
                        if min_x <= x_val <= max_x and min_y <= y_val <= max_y:
                            self._event_cluster_labels[event_id] = cluster_id
                            break
            
            # Update counts (use Counter for efficiency)
            counts = Counter(self._event_cluster_labels.values())
            for cluster in self._clusters:
                item = self._cluster_items.get(cluster.id)
                if item is not None:
                    item.setText(f"{cluster.name} ({counts.get(cluster.id, 0)} events)")
            
            # Note: overlay colors are set at creation time, not refreshed here
            
            self._cluster_membership_dirty = False
        
        # Always update Unclassified count when clustering is enabled (even with no user classes)
        if self.clustering_enabled_check.isChecked() and self._unclassified_item is not None:
            total_with_id = sum(1 for eid in event_ids if eid is not None)
            total_classified = len(self._event_cluster_labels)
            unclassified_count = total_with_id - total_classified
            self._unclassified_item.setText(f"Unclassified ({max(0, unclassified_count)} events)")
        
        # Build brush list with caching
        default_brush = self._brush_cache.get(-1)
        if default_brush is None:
            default_brush_color = QtGui.QColor(UNCLASSIFIED_COLOR)
            default_brush_color.setAlpha(170)
            default_brush = pg.mkBrush(default_brush_color)
            self._brush_cache[-1] = default_brush
        
        brushes: list[object] = []
        for eid in event_ids:
            if eid is None:
                brushes.append(default_brush)
                continue
            cluster_id = self._event_cluster_labels.get(eid)
            if cluster_id is None:
                brushes.append(default_brush)
                continue
            # Get cached brush or create and cache
            brush = self._brush_cache.get(cluster_id)
            if brush is None:
                cluster = next((c for c in self._clusters if c.id == cluster_id), None)
                if cluster is None:
                    brushes.append(default_brush)
                    continue
                brush = pg.mkBrush(cluster.color)
                self._brush_cache[cluster_id] = brush
            brushes.append(brush)
        
        self.energy_scatter.setData(xs, ys, brush=brushes)
        self.energy_scatter.show()
        self._last_scatter_count = current_count
        plot_item = self.metrics_plot.getPlotItem()
        self._set_metrics_range(plot_item, min(xs), max(xs), min(ys), max(ys))

    def _record_overlay_metrics(self, overlay: dict[str, object]) -> None:
        """Extract metrics from overlay and store for scatter plot."""
        metrics = overlay.get("metrics")
        metric_time = overlay.get("metric_time")
        if metrics is None or metric_time is None:
            return
        if self._t0_event is None:
            self._t0_event = float(metric_time)
        rel_time = max(0.0, float(metric_time) - (self._t0_event or 0.0))
        record: dict[str, float | int] = {"time": rel_time}
        has_metric = False
        for key in ("ed", "max", "min", "freq", "interval"):
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
        # deque with maxlen handles overflow automatically - O(1) instead of O(n)
        # Note: We don't clean up _event_cluster_labels here anymore; it's done lazily
        # in _update_metric_points when events are no longer visible.

    def _build_overlay_payload(self, event: AnalysisEvent) -> Optional[OverlayPayload]:
        """Build overlay data from a detected event for visualization."""
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
        baseline_val = baseline(samples, pre_samples)
        x = samples.astype(np.float32) - baseline_val
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
            ed = energy_density(samples, sr)
            mx, mn = min_max(samples)
            pf = peak_frequency_sinc(samples, sr, center_index=cross_idx)
            metric_values.update(
                {
                    "ed": float(ed),
                    "max": float(mx),
                    "min": float(mn),
                    "freq": float(pf),
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
            baseline=float(baseline_val),
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
        """Create a plot item from overlay payload data."""
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
        # Classify the event immediately before applying color
        # so overlay gets the correct color at creation time
        event_id = payload.event_id
        if isinstance(event_id, int) and self.clustering_enabled_check.isChecked() and self._clusters:
            metrics = payload.metrics
            if isinstance(metrics, dict):
                x_key = self._selected_x_metric()
                y_key = self._selected_y_metric()
                x_val = metrics.get(x_key)
                y_val = metrics.get(y_key)
                if x_val is not None and y_val is not None:
                    try:
                        x_num = float(x_val)
                        y_num = float(y_val)
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
                            if min_x <= x_num <= max_x and min_y <= y_num <= max_y:
                                self._event_cluster_labels[event_id] = cluster.id
                                break
                    except (TypeError, ValueError):
                        pass
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
        """Position and clip overlay data to current view window."""
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
        """Handle cluster ROI resize/move: recompute membership."""
        self._cluster_membership_dirty = True  # Force full reclassification
        self._brush_cache.clear()  # Clear brush cache since cluster bounds changed
        self._recompute_cluster_membership()
        self._update_metric_points()

    def _recompute_cluster_membership(self) -> None:
        """Recalculate which events belong to which clusters based on ROI bounds."""
        if not self._clusters:
            self._event_cluster_labels.clear()
            # Note: overlay colors are set at creation time, not refreshed here
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
        # When paused, don't clear labels - preserve them so visible overlays keep colors
        if not self._viz_paused:
            self._event_cluster_labels.clear()
        if not cluster_bounds:
            # Note: overlay colors are set at creation time, not refreshed here
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
        
        # Update Unclassified count
        total_events_with_id = sum(1 for r in self._metric_events if isinstance(r.get("event_id"), int))
        total_classified = sum(counts.values())
        unclassified_count = total_events_with_id - total_classified
        if self._unclassified_item is not None:
            self._unclassified_item.setText(f"Unclassified ({max(0, unclassified_count)} events)")
        
        # Note: overlay colors are set at creation time, not refreshed here


class ClusterWaveformDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        class_name: str,
        waveforms: Sequence[tuple[np.ndarray, np.ndarray]],
        color: Optional[QtGui.QColor] = None,
    ) -> None:
        super().__init__(parent)
        self._class_name = class_name
        self._cluster_color = color
        self._waveforms = self._sanitize_waveforms(waveforms)
        self._aligned_samples: list[np.ndarray] = []
        self._plot_time_axis: np.ndarray | None = None
        self._export_time_axis: np.ndarray | None = None
        self._median_waveform: np.ndarray | None = None
        self._measure_mode: str = "none"  # none | point | line
        self._measure_points: list[dict[str, object]] = []
        self._dragging_point_idx: int | None = None
        self._active_line: _MeasureLine | None = None
        self._measure_lines: list[_MeasureLine] = []
        self._line_anchor: QtCore.QPointF | None = None

        self._prepare_waveform_data()
        self._build_ui()
        self._plot_waveforms()
        self._update_title()

    def _sanitize_waveforms(self, waveforms: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
        sanitized: list[tuple[np.ndarray, np.ndarray]] = []
        for times, samples in waveforms:
            t_arr = np.asarray(times, dtype=np.float64)
            s_arr = np.asarray(samples, dtype=np.float32)
            length = int(min(t_arr.size, s_arr.size))
            if length <= 0:
                continue
            t_trim = t_arr[:length]
            sanitized.append((t_trim, s_arr[:length]))
        return sanitized

    def _prepare_waveform_data(self) -> None:
        if not self._waveforms:
            return
        lengths: list[int] = []
        for times, samples in self._waveforms:
            lengths.append(int(min(times.size, samples.size)))
        if not lengths:
            return
        length_counts = Counter(lengths)
        target_len = max(length_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        aligned: list[tuple[np.ndarray, np.ndarray]] = []
        for times, samples in self._waveforms:
            if times.size < target_len or samples.size < target_len or target_len <= 0:
                continue
            t_copy = np.array(times[:target_len], dtype=np.float64, copy=True)
            s_copy = np.array(samples[:target_len], dtype=np.float32, copy=True)
            aligned.append((t_copy, s_copy))
        if not aligned:
            return
        self._plot_time_axis = aligned[0][0]
        self._aligned_samples = [samples for _, samples in aligned]
        stack = np.stack(self._aligned_samples, axis=0)
        self._median_waveform = np.median(stack, axis=0)
        self._export_time_axis = self._build_export_time_axis(self._plot_time_axis)

    def _build_export_time_axis(self, reference: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if reference is None or reference.size == 0:
            return None
        if reference.size < 2:
            dt = 0.0
        else:
            diffs = np.diff(reference)
            finite = diffs[np.isfinite(diffs) & (diffs > 0)]
            dt = float(np.median(finite)) if finite.size else 0.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        start = float(reference[0])
        return start + (np.arange(reference.size, dtype=np.float64) * dt)

    def _safe_base_name(self) -> str:
        base = self._class_name.strip() or "waveforms"
        safe = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in base)
        return safe.lower()

    def _build_ui(self) -> None:
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor("white"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        menu_bar = QtWidgets.QMenuBar(self)
        file_menu = menu_bar.addMenu("File")
        layout.setMenuBar(menu_bar)

        save_image_action = QtGui.QAction("Save as image\u2026", self)
        save_image_action.triggered.connect(self._on_save_image)
        file_menu.addAction(save_image_action)

        save_traces_action = QtGui.QAction("Save traces\u2026", self)
        save_traces_action.triggered.connect(self._on_save_traces)
        file_menu.addAction(save_traces_action)

        file_menu.addSeparator()
        close_action = QtGui.QAction("Close", self)
        close_action.setShortcuts([QtGui.QKeySequence.Close, QtGui.QKeySequence("Ctrl+W")])
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        self.addAction(close_action)

        measure_menu = menu_bar.addMenu("Measure")
        point_action = QtGui.QAction("Point", self)
        point_action.triggered.connect(lambda: self._set_measure_mode("point"))
        measure_menu.addAction(point_action)

        measure_action = QtGui.QAction("Measure", self)
        measure_action.triggered.connect(lambda: self._set_measure_mode("line"))
        measure_menu.addAction(measure_action)

        measure_menu.addSeparator()
        clear_measure_action = QtGui.QAction("Clear", self)
        clear_measure_action.triggered.connect(self._clear_measurements)
        measure_menu.addAction(clear_measure_action)

        self.plot_widget = pg.PlotWidget(enableMenu=False)
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "Amplitude (V)")
        plot_item = self.plot_widget.getPlotItem()
        plot_item.showGrid(x=True, y=True, alpha=0.35)
        vb = plot_item.getViewBox()
        if vb is not None:
            vb.setMouseEnabled(x=True, y=True)
        for axis_name in ("bottom", "left"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen("black"))
            axis.setTextPen(pg.mkPen("black"))
        layout.addWidget(self.plot_widget, 1)
        # Accept gesture-based zoom (trackpad pinch) on the viewport.
        viewport = self.plot_widget.viewport()
        viewport.setAttribute(QtCore.Qt.WA_AcceptTouchEvents, True)
        viewport.grabGesture(QtCore.Qt.PinchGesture)
        viewport.installEventFilter(self)
        self.plot_widget.scene().installEventFilter(self)

    def _plot_waveforms(self) -> None:
        if not self._aligned_samples or self._plot_time_axis is None:
            return
        line_color = QtGui.QColor(STA_TRACE_PEN.color())
        line_pen = pg.mkPen(line_color, width=1)
        for samples in self._aligned_samples:
            self.plot_widget.plot(self._plot_time_axis, samples, pen=line_pen, clear=False)
        if self._median_waveform is not None:
            median_pen = pg.mkPen(WAVEFORM_MEDIAN_COLOR, width=3)
            self.plot_widget.plot(self._plot_time_axis, self._median_waveform, pen=median_pen)
        self.plot_widget.setAntialiasing(False)
        self._set_measure_mode("none")

    def _update_title(self) -> None:
        count = len(self._aligned_samples) if self._aligned_samples else len(self._waveforms)
        self.setWindowTitle(f"Waveforms \u2013 {self._class_name} ({count} events)")

    def _on_save_image(self) -> None:
        if self._plot_time_axis is None or not self._aligned_samples:
            QtWidgets.QMessageBox.information(self, "Save image", "No waveform data available to save.")
            return
        suggested = f"{self._safe_base_name()}_waveforms.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save as image", suggested, "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path = f"{path}.png"
        pixmap = self.grab()
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(self, "Save image", "Unable to capture the waveform window.")
            return
        if not pixmap.save(path, "PNG"):
            QtWidgets.QMessageBox.critical(self, "Save image", "Failed to save the image.")

    def _on_save_traces(self) -> None:
        if not self._aligned_samples or self._export_time_axis is None or self._median_waveform is None:
            QtWidgets.QMessageBox.information(self, "Save traces", "No waveform data available to export.")
            return
        suggested = f"{self._safe_base_name()}_waveforms.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save traces", suggested, "CSV file (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path = f"{path}.csv"
        try:
            with open(path, "w", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(self._export_time_axis)
                writer.writerow(self._median_waveform)
                for samples in self._aligned_samples:
                    writer.writerow(samples)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save traces", f"Failed to save traces:\n{exc}")

    def _set_measure_mode(self, mode: str) -> None:
        self._measure_mode = mode
        vb = self.plot_widget.getViewBox()
        if vb is not None:
            vb.setMouseEnabled(mode == "none", mode == "none")
        if mode == "none":
            self.plot_widget.unsetCursor()
        else:
            self.plot_widget.setCursor(QtCore.Qt.CrossCursor)
        self._dragging_point_idx = None
        self._active_line = None
        self._line_anchor = None

    def _handle_mouse_press(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> bool:
        if event.button() != QtCore.Qt.LeftButton:
            return False
        scene_pos = event.scenePos()
        vb = self.plot_widget.getViewBox()
        if vb is None:
            return False
        hit_point = self._hit_test_point(scene_pos)
        data_pos = vb.mapSceneToView(scene_pos)
        if not np.isfinite(data_pos.x()) or not np.isfinite(data_pos.y()):
            return False
        if self._measure_mode == "none":
            if hit_point is not None:
                self._dragging_point_idx = hit_point
                event.accept()
                return True
            return False
        if self._measure_mode == "point":
            if hit_point is not None:
                self._dragging_point_idx = hit_point
                event.accept()
                return True
            self._add_measure_point(data_pos)
            self._dragging_point_idx = len(self._measure_points) - 1
            event.accept()
            return True
        # If clicking on an existing line handle, let ROI handle manage it.
        if self._line_handle_hit(scene_pos):
            return False
        # Begin a new measurement line
        line = _MeasureLine(self.plot_widget.getPlotItem(), data_pos, data_pos, mode=self._measure_mode)
        self._measure_lines.append(line)
        self._active_line = line
        self._line_anchor = QtCore.QPointF(data_pos)
        event.accept()
        return True

    def _handle_mouse_move(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> bool:
        if self._measure_mode == "none" and self._dragging_point_idx is None and self._active_line is None:
            return False
        scene_pos = event.scenePos()
        vb = self.plot_widget.getViewBox()
        if vb is None:
            return False
        data_pos = vb.mapSceneToView(scene_pos)
        if not np.isfinite(data_pos.x()) or not np.isfinite(data_pos.y()):
            return False
        if self._dragging_point_idx is not None:
            self._update_measure_point(self._dragging_point_idx, data_pos)
            event.accept()
            return True
        if self._active_line is not None:
            anchor = self._line_anchor if self._line_anchor is not None else data_pos
            self._active_line.set_points(anchor, data_pos)
            event.accept()
            return True
        return False

    def _handle_mouse_release(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> bool:
        if self._measure_mode == "none" and self._dragging_point_idx is None and self._active_line is None:
            return False
        if event.button() != QtCore.Qt.LeftButton:
            return False
        if self._dragging_point_idx is not None:
            self._dragging_point_idx = None
            if self._measure_mode != "none":
                self._set_measure_mode("none")
            event.accept()
            return True
        if self._active_line is not None:
            self._active_line = None
            self._line_anchor = None
            if self._measure_mode != "none":
                self._set_measure_mode("none")
            event.accept()
            return True
        return False

    def _add_measure_point(self, pos: QtCore.QPointF) -> None:
        scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], symbol="+", size=14, pen=pg.mkPen("black"), brush=None)
        label = pg.TextItem(color=(20, 20, 20))
        self.plot_widget.addItem(scatter)
        self.plot_widget.addItem(label)
        entry = {"item": scatter, "label": label}
        self._measure_points.append(entry)
        self._update_measure_point(len(self._measure_points) - 1, pos)

    def _update_measure_point(self, idx: int, pos: QtCore.QPointF) -> None:
        if idx < 0 or idx >= len(self._measure_points):
            return
        entry = self._measure_points[idx]
        item = entry.get("item")
        label = entry.get("label")
        if isinstance(item, pg.ScatterPlotItem):
            item.setData([pos.x()], [pos.y()])
            item.setToolTip(f"({pos.x():.4g} s, {pos.y():.4g} V)")
        if isinstance(label, pg.TextItem):
            label.setText(f"({pos.x():.4g} s, {pos.y():.4g} V)")
            label.setPos(pos.x(), pos.y())

    def _hit_test_point(self, scene_pos: QtCore.QPointF, pixel_radius: float = 8.0) -> int | None:
        vb = self.plot_widget.getViewBox()
        if vb is None:
            return None
        for idx, entry in enumerate(self._measure_points):
            item = entry.get("item")
            if not isinstance(item, pg.ScatterPlotItem):
                continue
            spots = item.points()
            if not spots:
                continue
            spot = spots[0]
            pt = vb.mapViewToScene(spot.pos())
            dist = (QtCore.QPointF(pt) - scene_pos)
            if (dist.x() ** 2 + dist.y() ** 2) ** 0.5 <= pixel_radius:
                return idx
        return None

    def _line_handle_hit(self, scene_pos: QtCore.QPointF, pixel_radius: float = 8.0) -> bool:
        for line in self._measure_lines:
            handles = line.roi.getSceneHandlePositions()
            for _, h_pos in handles:
                dist = QtCore.QPointF(h_pos) - scene_pos
                if (dist.x() ** 2 + dist.y() ** 2) ** 0.5 <= pixel_radius:
                    return True
        return False

    def _clear_measurements(self) -> None:
        for entry in self._measure_points:
            item = entry.get("item")
            label = entry.get("label")
            try:
                if isinstance(item, pg.ScatterPlotItem):
                    self.plot_widget.removeItem(item)
            except Exception as e:
                logger.debug("Failed to remove scatter item: %s", e)
            try:
                if isinstance(label, pg.TextItem):
                    self.plot_widget.removeItem(label)
            except Exception as e:
                logger.debug("Failed to remove text label: %s", e)
        self._measure_points.clear()
        for line in self._measure_lines:
            line.remove()
        self._measure_lines.clear()
        self._dragging_point_idx = None
        self._active_line = None
        self._line_anchor = None

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802 - Qt API
        if obj is self.plot_widget.viewport():
            if event.type() == QtCore.QEvent.Gesture:
                pinch = event.gesture(QtCore.Qt.PinchGesture)
                if pinch is not None:
                    self._handle_pinch_gesture(pinch)
                    return True
            if event.type() == QtCore.QEvent.NativeGesture:
                if event.gestureType() == QtCore.Qt.NativeGestureType.Zoom:
                    delta = float(event.value())
                    factor = 1.0 + delta
                    self._apply_zoom_factor(factor)
                    return True
        if obj is self.plot_widget.scene():
            if event.type() == QtCore.QEvent.GraphicsSceneMousePress:
                return self._handle_mouse_press(event)
            if event.type() == QtCore.QEvent.GraphicsSceneMouseMove:
                return self._handle_mouse_move(event)
            if event.type() == QtCore.QEvent.GraphicsSceneMouseRelease:
                return self._handle_mouse_release(event)
        return super().eventFilter(obj, event)

    def _handle_pinch_gesture(self, pinch: QtGui.QPinchGesture) -> None:
        try:
            factor = float(pinch.scaleFactor())
        except Exception as e:
            logger.debug("Failed to get pinch scale factor: %s", e)
            return
        if not np.isfinite(factor) or factor <= 0.0:
            return
        self._apply_zoom_factor(factor)

    def _apply_zoom_factor(self, factor: float) -> None:
        if not np.isfinite(factor) or factor <= 0.0:
            return
        plot_item = self.plot_widget.getPlotItem()
        vb = plot_item.getViewBox() if plot_item is not None else None
        if vb is None:
            return
        # Scale equally on both axes; invert because scaleBy zooms the view rect.
        inv = 1.0 / factor
        vb.scaleBy((inv, inv))
