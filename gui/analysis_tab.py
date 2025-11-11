from __future__ import annotations

import math
import queue
from typing import Optional, TYPE_CHECKING

import numpy as np

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from core.models import Chunk, EndOfStream
from shared.types import Event


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


def _peak_frequency_sinc(x: np.ndarray, sr: float) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0 or sr <= 0:
        return 0.0
    samples = arr.astype(np.float64)
    samples -= np.median(samples)
    window = _blackman(samples.size)
    tapered = samples * window
    n_fft_min = max(2048, samples.size * 8)
    n_fft = 1 << max(0, int(math.ceil(math.log2(n_fft_min))))
    spectrum = np.fft.rfft(tapered, n=n_fft)
    mags = np.abs(spectrum)
    mags[0] = 0.0
    if mags.size <= 1:
        return 0.0
    peak_idx = int(np.argmax(mags))
    if peak_idx <= 0 or peak_idx >= mags.size - 1:
        peak_freq = peak_idx * sr / n_fft
        return float(peak_freq)
    alpha = mags[peak_idx - 1]
    beta = mags[peak_idx]
    gamma = mags[peak_idx + 1]
    denom = max((alpha - 2 * beta + gamma), 1e-12)
    delta = 0.5 * (alpha - gamma) / denom
    peak_bin = peak_idx + delta
    peak_freq = peak_bin * sr / n_fft
    return float(max(0.0, peak_freq))

if TYPE_CHECKING:
    from analysis.analysis_worker import AnalysisWorker
    from analysis.settings import AnalysisSettingsStore
    from core.controller import PipelineController
    from shared.event_buffer import AnalysisEvents, EventRingBuffer


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
        self._analysis_events: Optional["AnalysisEvents"] = None
        self._event_buffer: Optional["EventRingBuffer"] = None
        self._worker: Optional["AnalysisWorker"] = None
        self._last_event_id: Optional[int] = None
        self._event_overlays: list[dict[str, object]] = []
        self._overlay_pool: list[pg.PlotCurveItem] = []
        self._overlay_pen = pg.mkPen((220, 0, 0), width=2)
        self._latest_sample_time: Optional[float] = None
        self._window_start_time: Optional[float] = None
        self._channel_index: Optional[int] = None
        self._latest_sample_index: Optional[int] = None
        self._window_start_index: Optional[int] = None
        if controller is not None:
            self._analysis_settings = getattr(controller, "analysis_settings_store", None)
            self._analysis_events = getattr(controller, "analysis_events", None)
            self._event_buffer = getattr(controller, "event_buffer", None)
        self._event_window_ms = self._initial_event_window_ms()
        self._t0_event: Optional[float] = None
        self._metric_events: list[dict[str, float]] = []
        self._max_metric_events = 10_000

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel(self._title_text())
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.title_label)

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
        self.plot_widget.getPlotItem().setXRange(0.0, 1.0, padding=0.0)
        self.plot_widget.getPlotItem().setYRange(-1.0, 1.0, padding=0.0)
        layout.addWidget(self.plot_widget, stretch=4)

        self.metrics_plot = pg.PlotWidget(enableMenu=False)
        try:
            self.metrics_plot.hideButtons()
        except Exception:
            pass
        self.metrics_plot.setBackground(pg.mkColor(245, 246, 250))
        self.metrics_plot.setLabel("bottom", "Time (s)")
        self.metrics_plot.setLabel("left", "Energy Density (V^2/s)")
        self.metrics_plot.showGrid(x=True, y=True, alpha=0.3)
        self.energy_scatter = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(220, 0, 0, 170), pen=None, name="Energy Density")
        self.metrics_plot.addItem(self.energy_scatter)
        self.energy_scatter.hide()

        controls = QtWidgets.QGroupBox("Display")
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(14, 14, 14, 12)
        controls_layout.setSpacing(16)

        size_layout = QtWidgets.QVBoxLayout()
        size_layout.setSpacing(8)
        width_row = QtWidgets.QHBoxLayout()
        width_row.setSpacing(6)
        width_row.addWidget(QtWidgets.QLabel("Width (s)"))
        self.width_combo = QtWidgets.QComboBox()
        self.width_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.width_combo.setMinimumWidth(110)
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
        self.height_combo.setMinimumWidth(110)
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

        # TODO: Second threshold temporarily disabled; will re-enable in a later feature.
        self.threshold2_check = QtWidgets.QCheckBox("Threshold 2")
        self.threshold2_check.setEnabled(False)
        self.threshold2_check.setStyleSheet("color: rgb(130, 130, 130);")
        self.threshold2_spin = QtWidgets.QDoubleSpinBox()
        self.threshold2_spin.setDecimals(3)
        self.threshold2_spin.setMinimumWidth(90)
        self.threshold2_spin.setRange(-10.0, 10.0)
        self.threshold2_spin.setValue(-0.5)
        self.threshold2_spin.setEnabled(False)
        self.threshold2_spin.setStyleSheet("color: rgb(130, 130, 130);")
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

        self.metrics_autorange = QtWidgets.QCheckBox("Auto-range")
        self.metrics_autorange.setChecked(True)
        self.metrics_autorange.toggled.connect(lambda _: self._update_metric_points())
        metrics_layout.addWidget(self.metrics_autorange)

        metric_row = QtWidgets.QHBoxLayout()
        metric_row.setSpacing(6)
        metric_row.addWidget(QtWidgets.QLabel("Vertical (Y) Metric"))
        self.metric_combo = QtWidgets.QComboBox()
        for label in ("Max in window (V)", "Energy Density (V²/s)", "Min in window (V)"):
            self.metric_combo.addItem(label)
        self.metric_combo.setCurrentIndex(0)
        self.metric_combo.currentIndexChanged.connect(self._on_axis_metric_changed)
        metric_row.addWidget(self.metric_combo)
        metrics_layout.addLayout(metric_row)

        xaxis_row = QtWidgets.QHBoxLayout()
        xaxis_row.setSpacing(6)
        xaxis_row.addWidget(QtWidgets.QLabel("Horizontal (X) Axis"))
        self.metric_xaxis_combo = QtWidgets.QComboBox()
        for label in ("Time (s)", "Energy Density (V²/s)", "Max in window (V)", "Min in window (V)"):
            self.metric_xaxis_combo.addItem(label)
        self.metric_xaxis_combo.currentIndexChanged.connect(self._on_axis_metric_changed)
        xaxis_row.addWidget(self.metric_xaxis_combo)
        metrics_layout.addLayout(xaxis_row)

        self.metrics_clear_btn = QtWidgets.QPushButton("Clear metrics")
        self.metrics_clear_btn.clicked.connect(self._clear_metrics)
        metrics_layout.addWidget(self.metrics_clear_btn)
        metrics_layout.addStretch(1)

        controls_layout.addLayout(metrics_layout)
        controls_layout.addStretch(1)

        controls.setLayout(controls_layout)
        layout.addWidget(controls, stretch=2)

        layout.addWidget(self.metrics_plot, stretch=4)

        self.raw_curve = self.plot_widget.plot(pen=pg.mkPen((30, 144, 255), width=2))
        self.event_curve = self.plot_widget.plot(pen=pg.mkPen((200, 0, 0), width=2))

        self.threshold1_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen((128, 0, 128), width=2))
        self.threshold1_line.setZValue(10)
        self.threshold1_line.setVisible(False)
        self.plot_widget.addItem(self.threshold1_line)

        self.threshold2_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen((150, 150, 150), width=2, style=QtCore.Qt.DashLine),
        )
        self.threshold2_line.setZValue(10)
        self.threshold2_line.setVisible(False)
        self.plot_widget.addItem(self.threshold2_line)

        layout.addStretch(1)

        self.width_combo.currentIndexChanged.connect(self._apply_ranges)
        self.height_combo.currentIndexChanged.connect(self._apply_ranges)
        self.threshold1_check.toggled.connect(lambda checked: self._toggle_threshold(self.threshold1_line, self.threshold1_spin, checked))
        self.threshold1_check.toggled.connect(lambda _: self._notify_threshold_change())
        self.threshold1_spin.valueChanged.connect(lambda val: self._update_threshold_from_spin(self.threshold1_line, val))
        self.threshold1_spin.valueChanged.connect(lambda _: self._notify_threshold_change())
        self.threshold1_line.sigPositionChanged.connect(lambda _: self._update_spin_from_line(self.threshold1_line, self.threshold1_spin))
        self.event_window_combo.currentIndexChanged.connect(self._on_event_window_changed)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()
        self._in_threshold_update = False
        self._apply_ranges()

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
        if spin is self.threshold1_spin:
            self._notify_threshold_change()

    def _on_timer(self) -> None:
        if self._analysis_queue is None:
            return
        latest: Optional[Chunk] = None
        while True:
            try:
                item = self._analysis_queue.get_nowait()
            except queue.Empty:
                break
            if item is EndOfStream:
                continue
            if isinstance(item, Chunk):
                latest = item
        if latest is not None:
            self._render_chunk(latest)
        else:
            self._update_event_overlays()

    def _apply_ranges(self) -> None:
        width = float(self.width_combo.currentData() or 0.5)
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, width, padding=0.0)
        if self._selected_y_metric() in {"ed", "max", "min"}:
            # Metrics overlay controls the Y range; leave amplitude height unchanged.
            pass
        else:
            height = float(self.height_combo.currentData() or 1.0)
            plot_item.setYRange(-height, height, padding=0.0)
        self._ensure_buffer_capacity(max(width, 1.0))
        if self.threshold1_check.isChecked():
            self.threshold1_line.setValue(float(self.threshold1_spin.value()))
        self._update_event_overlays()

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

    def _notify_threshold_change(self) -> None:
        if self._worker is None:
            return
        self._clear_metrics()
        enabled = self.threshold1_check.isChecked()
        value = float(self.threshold1_spin.value())
        # Threshold 2 remains disabled; only Threshold 1 updates the worker.
        try:
            self._worker.configure_threshold(enabled, value)
        except Exception:
            pass

    def _render_chunk(self, chunk: Chunk) -> None:
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
        else:
            self._latest_sample_index = None
            self._window_start_index = None
        times = np.arange(recent.size, dtype=np.float32) * self._dt
        self.raw_curve.setData(times, recent, skipFiniteCheck=True)
        self.event_curve.clear()

        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, width, padding=0.0)
        height = float(self.height_combo.currentData() or 1.0)
        plot_item.setYRange(-height, height, padding=0.0)
        self._update_event_overlays()

    def _pull_new_events(self) -> list[Event]:
        if self._analysis_events is None:
            return []
        events, last_id = self._analysis_events.pull_events(self._last_event_id)
        if last_id is not None:
            self._last_event_id = last_id
        return events

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

    def _update_event_overlays(self) -> None:
        if self._analysis_events is None or self._latest_sample_time is None:
            return
        width_setting = float(self.width_combo.currentData() or 0.5)
        if self._window_start_time is not None:
            window_start = self._window_start_time
        else:
            window_start = self._latest_sample_time - width_setting
        width_in_use = max(width_setting, self._latest_sample_time - window_start)
        window_start_idx = self._window_start_index
        channel_idx = self._channel_index
        new_events = self._pull_new_events()
        for event in new_events:
            if channel_idx is not None and event.channelId != channel_idx:
                continue
            overlay = self._build_overlay(event)
            if overlay is None:
                continue
            if not self._apply_overlay_view(overlay, window_start, width_in_use, window_start_idx):
                self._release_overlay_item(overlay.get("item"))
                continue
            self._event_overlays.append(overlay)
            self._record_overlay_metrics(overlay)
        self._refresh_overlay_positions(window_start, width_in_use, window_start_idx)
        self._update_metric_points()

    def _refresh_overlay_positions(self, window_start: float, width: float, window_start_idx: Optional[int]) -> None:
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
            kept.append(overlay)
        self._event_overlays = kept
        self._update_metric_points()

    def _clear_event_overlays(self) -> None:
        if not self._event_overlays:
            return
        for overlay in self._event_overlays:
            self._release_overlay_item(overlay.get("item"))
        self._event_overlays.clear()

    def _clear_metrics(self) -> None:
        self._metric_events.clear()
        self._t0_event = None
        if hasattr(self, "energy_scatter"):
            self.energy_scatter.clear()
            self.energy_scatter.hide()
        if hasattr(self, "metrics_plot"):
            self.metrics_plot.getPlotItem().enableAutoRange(y=True)

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
        if "max" in label:
            return "max"
        if "min" in label:
            return "min"
        return "ed"

    def _selected_x_metric(self) -> str:
        label = self.metric_xaxis_combo.currentText().lower()
        if "time" in label:
            return "time"
        if "max" in label:
            return "max"
        if "min" in label:
            return "min"
        return "ed"

    def _on_axis_metric_changed(self) -> None:
        metric = self._selected_y_metric()
        if metric == "ed":
            self.metrics_plot.setLabel("left", "Energy Density (V²/s)")
        else:
            self.metrics_plot.setLabel("left", "Amplitude (V)")
        x_metric = self._selected_x_metric()
        if x_metric == "time":
            self.metrics_plot.setLabel("bottom", "Time (s)")
        elif x_metric == "ed":
            self.metrics_plot.setLabel("bottom", "Energy Density (V²/s)")
        else:
            self.metrics_plot.setLabel("bottom", "Amplitude (V)")
        if metric in {"ed", "max", "min"}:
            self.energy_scatter.show()
            self._update_metric_points()
        else:
            self.energy_scatter.hide()
            self.energy_scatter.setData([], [])

    def _release_metrics(self) -> None:
        self._metric_events = []
        self._t0_event = None
        self._update_metric_points()

    def _update_metric_points(self) -> None:
        y_key = self._selected_y_metric()
        x_key = self._selected_x_metric()
        if y_key not in {"ed", "max", "min"}:
            self.energy_scatter.hide()
            return
        xs: list[float] = []
        ys: list[float] = []
        for event in self._metric_events:
            x_val = event.get(x_key)
            y_val = event.get(y_key)
            if x_val is None or y_val is None:
                continue
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue
            xs.append(float(x_val))
            ys.append(float(y_val))
        if not xs:
            self.energy_scatter.hide()
            return
        self.energy_scatter.setData(xs, ys)
        self.energy_scatter.show()
        if self.metrics_autorange.isChecked():
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
        record: dict[str, float] = {"time": rel_time}
        for key in ("ed", "max", "min"):
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
        if len(record) <= 1:
            return
        self._metric_events.append(record)
        if len(self._metric_events) > self._max_metric_events:
            self._metric_events.pop(0)

    def _build_overlay(self, event: Event) -> Optional[dict[str, object]]:
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
        curve = self._acquire_overlay_item()
        metrics = None
        if samples.size >= 4:
            ed = _energy_density(x, sr)
            mx, mn = _min_max(x)
            fpk = _peak_frequency_sinc(x, sr)
            metrics = {
                "ed": float(ed),
                "max": float(mx),
                "min": float(mn),
                "peak_freq_hz": float(fpk),
            }
        return {
            "item": curve,
            "times": times,
            "samples": samples,
            "last_time": last_time,
            "first_index": first_index,
            "sr": sr,
            "pre_samples": pre_samples,
            "baseline": baseline,
            "peak_idx": peak_idx,
            "peak_time": peak_time,
            "metrics": metrics,
            "metric_time": peak_time,
        }

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
        relative: np.ndarray
        if window_start_idx is not None and overlay.get("first_index") is not None:
            sr = float(overlay.get("sr") or 0.0)
            if sr <= 0:
                sr = self.sample_rate if self.sample_rate > 0 else 1.0
            first_index = int(overlay["first_index"])
            offsets = np.arange(arr_samples.size, dtype=np.float64) + (first_index - window_start_idx)
            relative = offsets / sr
        else:
            arr_times = np.asarray(times, dtype=np.float64)
            relative = arr_times - window_start_time
        mask = (relative >= 0.0) & (relative <= width)
        if not np.any(mask):
            return False
        item.setData(relative[mask].astype(np.float32), arr_samples[mask])
        return True
