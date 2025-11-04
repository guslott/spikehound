from __future__ import annotations

import queue
import threading
import time
from collections import deque

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from core import DeviceManager, PipelineController
from core.conditioning import ChannelFilterSettings, FilterSettings
from core.models import Chunk, EndOfStream
from audio.player import AudioPlayer, AudioConfig, list_output_devices
from .analysis_dock import AnalysisDock


@dataclass
class ChannelConfig:
    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(0, 0, 139))
    display_enabled: bool = True
    range_v: float = 1.0
    offset_v: float = 0.0
    notch_enabled: bool = False
    notch_freq: float = 60.0
    highpass_enabled: bool = False
    highpass_freq: float = 10.0
    lowpass_enabled: bool = False
    lowpass_freq: float = 1_000.0
    listen_enabled: bool = False
    analyze_enabled: bool = False
    channel_name: str = ""


class ChannelOptionsPanel(QtWidgets.QWidget):
    """Per-channel configuration widget."""

    configChanged = QtCore.Signal(ChannelConfig)
    analysisRequested = QtCore.Signal()

    def __init__(self, channel_id: int, channel_name: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._channel_id = channel_id
        self._config = ChannelConfig(channel_name=channel_name)
        self._block_updates = False

        self._build_ui()
        self.set_channel_name(channel_name)
        self.set_config(self._config)

    def _apply_toggle_style(self, button: QtWidgets.QPushButton) -> None:
        button.setCheckable(True)
        button.setStyleSheet(
            """
QPushButton {
    background-color: rgb(180, 180, 180);
    border: 1px solid rgb(90, 90, 90);
    padding: 4px 10px;
}
QPushButton:checked {
    background-color: rgb(30, 144, 255);
    color: rgb(255, 255, 255);
    font-weight: bold;
}
"""
        )

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._title_label = QtWidgets.QLabel("")
        self._title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._title_label)

        # Visual identity controls
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(QtWidgets.QLabel("Color"))
        self.color_btn = QtWidgets.QPushButton()
        self.color_btn.setFixedWidth(48)
        self.color_btn.clicked.connect(self._choose_color)
        color_row.addWidget(self.color_btn)
        color_row.addSpacing(12)
        self.display_check = QtWidgets.QCheckBox("Display")
        self.display_check.setChecked(True)
        color_row.addWidget(self.display_check)
        color_row.addStretch(1)
        layout.addLayout(color_row)

        # Plot scaling
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("Vertical Range (±V)"))
        self.range_combo = QtWidgets.QComboBox()
        self.range_combo.setMaximumWidth(120)
        for value in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
            self.range_combo.addItem(f"{value:.1f}", value)
        range_row.addWidget(self.range_combo, 1)
        layout.addLayout(range_row)

        # DC offset adjustment
        offset_row = QtWidgets.QHBoxLayout()
        offset_row.addWidget(QtWidgets.QLabel("Vertical Offset (V)"))
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-10_000.0, 10_000.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSingleStep(0.1)
        self.offset_spin.setMaximumWidth(110)
        offset_row.addWidget(self.offset_spin, 1)
        layout.addLayout(offset_row)

        # Filtering controls
        notch_row = QtWidgets.QHBoxLayout()
        self.notch_check = QtWidgets.QCheckBox("Notch Filter")
        notch_row.addWidget(self.notch_check)
        notch_row.addStretch(1)
        notch_row.addWidget(QtWidgets.QLabel("Hz"))
        self.notch_spin = QtWidgets.QDoubleSpinBox()
        self.notch_spin.setRange(1.0, 1_000.0)
        self.notch_spin.setValue(60.0)
        self.notch_spin.setDecimals(1)
        self.notch_spin.setSingleStep(1.0)
        self.notch_spin.setMaximumWidth(110)
        notch_row.addWidget(self.notch_spin)
        layout.addLayout(notch_row)

        hp_row = QtWidgets.QHBoxLayout()
        self.highpass_check = QtWidgets.QCheckBox("High-pass")
        hp_row.addWidget(self.highpass_check)
        hp_row.addStretch(1)
        hp_row.addWidget(QtWidgets.QLabel("Hz"))
        self.highpass_spin = QtWidgets.QDoubleSpinBox()
        self.highpass_spin.setRange(0.1, 10_000.0)
        self.highpass_spin.setValue(10.0)
        self.highpass_spin.setDecimals(1)
        self.highpass_spin.setSingleStep(1.0)
        self.highpass_spin.setMaximumWidth(110)
        hp_row.addWidget(self.highpass_spin)
        layout.addLayout(hp_row)

        lp_row = QtWidgets.QHBoxLayout()
        self.lowpass_check = QtWidgets.QCheckBox("Low-pass")
        lp_row.addWidget(self.lowpass_check)
        lp_row.addStretch(1)
        lp_row.addWidget(QtWidgets.QLabel("Hz"))
        self.lowpass_spin = QtWidgets.QDoubleSpinBox()
        self.lowpass_spin.setRange(1.0, 50_000.0)
        self.lowpass_spin.setValue(1_000.0)
        self.lowpass_spin.setDecimals(1)
        self.lowpass_spin.setSingleStep(10.0)
        self.lowpass_spin.setMaximumWidth(110)
        lp_row.addWidget(self.lowpass_spin)
        layout.addLayout(lp_row)

        layout.addStretch(1)

        # Downstream feature toggles
        toggle_row = QtWidgets.QHBoxLayout()
        self.listen_btn = QtWidgets.QPushButton("Listen")
        self._apply_toggle_style(self.listen_btn)
        toggle_row.addWidget(self.listen_btn, 1)
        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.setStyleSheet(
            "background-color: rgb(180, 180, 180); border: 1px solid rgb(90,90,90); padding: 4px 10px;"
        )
        toggle_row.addWidget(self.analyze_btn, 1)
        toggle_row.addStretch(1)
        layout.addLayout(toggle_row)

        # Wire signals
        self.range_combo.currentIndexChanged.connect(self._on_widgets_changed)
        self.notch_check.toggled.connect(self._on_notch_toggled)
        self.notch_spin.valueChanged.connect(self._on_widgets_changed)
        self.highpass_check.toggled.connect(self._on_highpass_toggled)
        self.highpass_spin.valueChanged.connect(self._on_widgets_changed)
        self.lowpass_check.toggled.connect(self._on_lowpass_toggled)
        self.lowpass_spin.valueChanged.connect(self._on_widgets_changed)
        self.offset_spin.valueChanged.connect(self._on_widgets_changed)
        self.listen_btn.toggled.connect(self._on_widgets_changed)
        self.display_check.toggled.connect(self._on_display_toggled)
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)

    def set_channel_name(self, name: str) -> None:
        self._config.channel_name = name
        self._title_label.setText(f"Channel: {name}")

    def set_config(self, config: ChannelConfig) -> None:
        self._config = replace(config)
        self._block_updates = True
        self._apply_color(config.color)
        idx = self.range_combo.findData(config.range_v)
        if idx >= 0:
            self.range_combo.setCurrentIndex(idx)
        else:
            self.range_combo.addItem(f"{config.range_v:.3f}", config.range_v)
            self.range_combo.setCurrentIndex(self.range_combo.count() - 1)
        self.notch_check.setChecked(config.notch_enabled)
        self.notch_spin.setValue(config.notch_freq)
        self.notch_spin.setEnabled(config.notch_enabled)
        self.highpass_check.setChecked(config.highpass_enabled)
        self.highpass_spin.setValue(config.highpass_freq)
        self.highpass_spin.setEnabled(config.highpass_enabled)
        self.lowpass_check.setChecked(config.lowpass_enabled)
        self.lowpass_spin.setValue(config.lowpass_freq)
        self.lowpass_spin.setEnabled(config.lowpass_enabled)
        self.offset_spin.setValue(config.offset_v)
        self.listen_btn.setChecked(config.listen_enabled)
        self.display_check.setChecked(config.display_enabled)
        self._block_updates = False

    def _apply_color(self, color: QtGui.QColor) -> None:
        qcolor = QtGui.QColor(color)
        if not qcolor.isValid():
            qcolor = QtGui.QColor(0, 0, 139)
        self._config.color = qcolor
        self.color_btn.setStyleSheet(
            f"background-color: {qcolor.name()}; border: 1px solid rgb(40,40,40);"
        )

    def _choose_color(self) -> None:
        initial = QtGui.QColor(self._config.color)
        color = QtWidgets.QColorDialog.getColor(initial, self, "Select Channel Color")
        if color.isValid():
            self._apply_color(color)
            self._emit_config()

    def _on_notch_toggled(self, checked: bool) -> None:
        self.notch_spin.setEnabled(checked)
        self._on_widgets_changed()

    def _on_highpass_toggled(self, checked: bool) -> None:
        self.highpass_spin.setEnabled(checked)
        self._on_widgets_changed()

    def _on_lowpass_toggled(self, checked: bool) -> None:
        self.lowpass_spin.setEnabled(checked)
        self._on_widgets_changed()

    def _on_widgets_changed(self) -> None:
        if self._block_updates:
            return
        self._config.range_v = float(self.range_combo.currentData())
        self._config.offset_v = float(self.offset_spin.value())
        self._config.notch_enabled = self.notch_check.isChecked()
        self._config.notch_freq = float(self.notch_spin.value())
        self._config.highpass_enabled = self.highpass_check.isChecked()
        self._config.highpass_freq = float(self.highpass_spin.value())
        self._config.lowpass_enabled = self.lowpass_check.isChecked()
        self._config.lowpass_freq = float(self.lowpass_spin.value())
        self._config.listen_enabled = self.listen_btn.isChecked()
        self._config.display_enabled = self.display_check.isChecked()
        self._emit_config()

    def _emit_config(self) -> None:
        if self._block_updates:
            return
        self.configChanged.emit(replace(self._config))

    def _on_display_toggled(self, checked: bool) -> None:
        if self._block_updates:
            return
        self._config.display_enabled = checked
        self._emit_config()

    def _on_analyze_clicked(self) -> None:
        self.analysisRequested.emit()


class ChannelViewBox(pg.ViewBox):
    """Custom ViewBox to expose click & drag events for channel selection."""

    channelClicked = QtCore.Signal(float, QtCore.Qt.MouseButton)
    channelDragged = QtCore.Signal(float)
    channelDragFinished = QtCore.Signal()

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("enableMenu", False)
        super().__init__(*args, **kwargs)
        self._dragging = False
        self._drag_button: Optional[QtCore.Qt.MouseButton] = None

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapSceneToView(event.scenePos())
            self.channelClicked.emit(pos.y(), event.button())
            self._dragging = True
            self._drag_button = event.button()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._dragging and self._drag_button == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapSceneToView(event.scenePos())
            self.channelDragged.emit(pos.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._dragging and event.button() == self._drag_button:
            self.channelDragFinished.emit()
            self._dragging = False
            self._drag_button = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window orchestrating plotting, device control, and audio monitoring."""

    startRequested = QtCore.Signal()
    stopRequested = QtCore.Signal()
    recordToggled = QtCore.Signal(bool)
    startRecording = QtCore.Signal(str, bool)
    stopRecording = QtCore.Signal()
    triggerConfigChanged = QtCore.Signal(dict)

    def __init__(self, controller: Optional[PipelineController] = None) -> None:
        super().__init__()
        if controller is None:
            controller = PipelineController()
        self._controller: Optional[PipelineController] = None
        self.setWindowTitle("SpikeHound - Manlius Pebble Hill School & Cornell University")
        self.resize(1100, 720)
        self.statusBar()

        # Persistent view/model state for the plotting surface and UI panels
        self._curves: List[pg.PlotCurveItem] = []
        self._channel_names: List[str] = []
        self._chunk_rate: float = 0.0
        self._device_map: Dict[str, dict] = {}
        self._device_connected = False
        self._active_channel_infos: List[object] = []
        self._channel_ids_current: List[int] = []
        self._curve_map: Dict[int, pg.PlotCurveItem] = {}
        self._channel_configs: Dict[int, ChannelConfig] = {}
        self._channel_panels: Dict[int, ChannelOptionsPanel] = {}
        self._channel_last_samples: Dict[int, np.ndarray] = {}
        self._channel_display_buffers: Dict[int, np.ndarray] = {}
        self._last_times: np.ndarray = np.zeros(0, dtype=np.float32)
        self._channel_color_cycle: List[QtGui.QColor] = [
            QtGui.QColor(0, 0, 139),
            QtGui.QColor(178, 34, 34),
            QtGui.QColor(0, 105, 148),
            QtGui.QColor(34, 139, 34),
            QtGui.QColor(128, 0, 128),
            QtGui.QColor(255, 140, 0),
        ]
        self._next_color_index = 0
        self._current_sample_rate: float = 0.0
        self._current_window_sec: float = 1.0
        self._dispatcher_signals: Optional[QtCore.QObject] = None
        self._drag_channel_id: Optional[int] = None
        self._active_channel_id: Optional[int] = None
        # Audio monitoring pipeline (listen button -> AudioPlayer)
        self._listen_channel_id: Optional[int] = None
        self._audio_router_thread: Optional[threading.Thread] = None
        self._audio_router_stop = threading.Event()
        self._audio_player: Optional[AudioPlayer] = None
        self._audio_player_queue: Optional["queue.Queue"] = None
        self._audio_input_samplerate: float = 0.0
        self._audio_lock = threading.Lock()
        self._audio_devices: List[Dict[str, object]] = []
        self._audio_current_device: Optional[object] = None
        self._trigger_mode: str = "stream"
        self._trigger_channel_id: Optional[int] = None
        self._trigger_threshold: float = 0.0
        self._trigger_pre_seconds: float = 0.01
        self._trigger_pre_samples: int = 0
        self._trigger_window_samples: int = 1
        self._trigger_last_sample_rate: float = 0.0
        self._trigger_history: deque[np.ndarray] = deque()
        self._trigger_history_length: int = 0
        self._trigger_history_total: int = 0
        self._trigger_max_chunk: int = 0
        self._trigger_prev_value: float = 0.0
        self._trigger_capture_start_abs: Optional[int] = None
        self._trigger_capture_end_abs: Optional[int] = None
        self._trigger_display: Optional[np.ndarray] = None
        self._trigger_display_times: Optional[np.ndarray] = None
        self._trigger_hold_until: float = 0.0
        self._trigger_single_armed: bool = False
        self._trigger_display_pre_samples: int = 0
        self._plot_refresh_hz = 40.0
        self._plot_interval = 1.0 / self._plot_refresh_hz
        self._last_plot_refresh = 0.0
        self._chunk_mean_samples: float = 0.0
        self._chunk_accum_count: int = 0
        self._chunk_accum_samples: int = 0
        self._chunk_rate_window: float = 1.0
        self._chunk_last_rate_update = time.perf_counter()

        self._apply_palette()
        self._style_plot()

        scope_widget = self._create_scope_widget()
        self.setCentralWidget(QtWidgets.QWidget(self))
        self._analysis_dock = AnalysisDock(parent=self)
        self._analysis_dock.set_scope_widget(scope_widget, "Scope")
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self._analysis_dock)
        self._analysis_dock.select_scope()
        self._wire_placeholders()
        self._update_trigger_controls()
        self._device_manager = DeviceManager(self)
        self._device_manager.devicesChanged.connect(self._on_devices_changed)
        self._device_manager.deviceConnected.connect(self._on_device_connected)
        self._device_manager.deviceDisconnected.connect(self._on_device_disconnected)
        self._device_manager.availableChannelsChanged.connect(self._on_available_channels)
        self._apply_device_state(False)
        self._populate_audio_outputs()
        self._device_manager.refresh_devices()
        self.attach_controller(controller)
        self._emit_trigger_config()

        # Global shortcuts for quitting/closing
        quit_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Quit), self)
        quit_shortcut.activated.connect(self._quit_application)
        close_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Close), self)
        close_shortcut.activated.connect(self.close)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _create_scope_widget(self) -> QtWidgets.QWidget:
        scope = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(scope)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)

        # Upper-left: plot area for multi-channel traces.
        self._view_box = ChannelViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self._view_box, enableMenu=False)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setBackground(QtGui.QColor(211, 230, 204))
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Amplitude", units="V")
        plot_item = self.plot_widget.getPlotItem()
        plot_item.getAxis("left").setPen(pg.mkPen((0, 0, 139)))
        plot_item.getAxis("bottom").setPen(pg.mkPen((0, 0, 139)))
        plot_item.showGrid(x=True, y=True, alpha=0.4)
        plot_item.vb.setBorder(pg.mkPen((0, 0, 139)))
        grid.addWidget(self.plot_widget, 0, 0)

        self.threshold_line = pg.InfiniteLine(angle=0, pen=pg.mkPen((178, 34, 34), width=3), movable=True)
        self.threshold_line.setVisible(False)
        self.plot_widget.addItem(self.threshold_line)
        try:
            self.threshold_line.setZValue(100)
        except AttributeError:
            pass

        self.pretrigger_line = pg.InfiniteLine(angle=90, pen=pg.mkPen((0, 0, 139), style=QtCore.Qt.DashLine), movable=False)
        self.pretrigger_line.setVisible(False)
        self.plot_widget.addItem(self.pretrigger_line)
        self._view_box.channelClicked.connect(self._on_plot_channel_clicked)
        self._view_box.channelDragged.connect(self._on_plot_channel_dragged)
        self._view_box.channelDragFinished.connect(self._on_plot_drag_finished)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(12)
        self._status_labels = {}
        for key in ("sr", "chunk", "queues", "drops"):
            label = QtWidgets.QLabel("SR: 0 Hz" if key == "sr" else "…")
            label.setStyleSheet("color: rgb(50,50,50); font-size: 11px;")
            label.setWordWrap(True)
            label.setMinimumHeight(28)
            status_row.addWidget(label)
            self._status_labels[key] = label
        status_row.addStretch(1)
        grid.addLayout(status_row, 1, 0, 1, 2)

        # Upper-right: stacked control boxes (Recording, Trigger, Channel Options).
        side_panel = QtWidgets.QWidget()
        side_panel.setMaximumWidth(320)
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        self.record_group = QtWidgets.QGroupBox("Recording")
        record_layout = QtWidgets.QVBoxLayout(self.record_group)

        path_row = QtWidgets.QHBoxLayout()
        self.record_path_edit = QtWidgets.QLineEdit()
        self.record_path_edit.setMaximumWidth(220)
        self.record_path_edit.setPlaceholderText("Select output file...")
        path_row.addWidget(self.record_path_edit, 1)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._on_browse_record_path)
        path_row.addWidget(browse_btn)
        record_layout.addLayout(path_row)

        self.record_autoinc = QtWidgets.QCheckBox("Auto-increment filename if exists")
        self.record_autoinc.setChecked(True)
        record_layout.addWidget(self.record_autoinc)

        self.record_toggle_btn = QtWidgets.QPushButton("Start Recording")
        self.record_toggle_btn.setCheckable(True)
        self._apply_record_button_style(False)
        self.record_toggle_btn.clicked.connect(self._toggle_recording)
        record_layout.addWidget(self.record_toggle_btn)

        record_layout.addStretch(1)
        side_layout.addWidget(self.record_group)

        self.trigger_group = QtWidgets.QGroupBox("Trigger")
        trigger_layout = QtWidgets.QGridLayout(self.trigger_group)

        row = 0
        trigger_layout.addWidget(self._label("Channel"), row, 0)
        self.trigger_channel_combo = QtWidgets.QComboBox()
        self.trigger_channel_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.trigger_channel_combo.setMaximumWidth(200)
        trigger_layout.addWidget(self.trigger_channel_combo, row, 1)
        row += 1

        trigger_layout.addWidget(self._label("Mode"), row, 0)
        mode_layout = QtWidgets.QVBoxLayout()
        self.trigger_mode_continuous = QtWidgets.QRadioButton("No Trigger (Stream)")
        self.trigger_mode_single = QtWidgets.QRadioButton("Single")
        self.trigger_mode_single.setEnabled(False)
        self.trigger_mode_repeating = QtWidgets.QRadioButton("Continuous Trigger")
        self.trigger_mode_repeating.setEnabled(False)
        self.trigger_mode_continuous.setChecked(True)
        mode_layout.addWidget(self.trigger_mode_continuous)
        single_row = QtWidgets.QHBoxLayout()
        single_row.setSpacing(4)
        self.trigger_single_button = QtWidgets.QPushButton("Trigger Once")
        self.trigger_single_button.setEnabled(False)
        single_row.addWidget(self.trigger_mode_single)
        single_row.addWidget(self.trigger_single_button)
        single_row.addStretch(1)
        mode_layout.addLayout(single_row)
        mode_layout.addWidget(self.trigger_mode_repeating)
        trigger_layout.addLayout(mode_layout, row, 1)
        row += 1

        threshold_box = QtWidgets.QHBoxLayout()
        threshold_box.addWidget(self._label("Threshold"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(-10.0, 10.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setValue(0.0)
        self.threshold_spin.setMaximumWidth(100)
        threshold_box.addWidget(self.threshold_spin)
        trigger_layout.addLayout(threshold_box, row, 0, 1, 2)
        row += 1

        pretrig_box = QtWidgets.QHBoxLayout()
        pretrig_box.addWidget(self._label("Pre-trigger (s)"))
        self.pretrigger_combo = QtWidgets.QComboBox()
        self.pretrigger_combo.setMaximumWidth(110)
        for value in (0.0, 0.01, 0.02, 0.05):
            self.pretrigger_combo.addItem(f"{value:.2f}", value)
        self.pretrigger_combo.setCurrentIndex(0)
        pretrig_box.addWidget(self.pretrigger_combo)
        trigger_layout.addLayout(pretrig_box, row, 0, 1, 2)
        row += 1

        window_box = QtWidgets.QHBoxLayout()
        window_box.addWidget(self._label("Window Width (s)"))
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.setMaximumWidth(110)
        for value in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
            self.window_combo.addItem(f"{value:.1f}", value)
        default_index = 3 if self.window_combo.count() > 3 else self.window_combo.count() - 1
        self.window_combo.setCurrentIndex(max(0, default_index))
        window_box.addWidget(self.window_combo)
        trigger_layout.addLayout(window_box, row, 0, 1, 2)

        side_layout.addWidget(self.trigger_group)

        self.channel_opts_group = QtWidgets.QGroupBox("Channel Options")
        channel_opts_layout = QtWidgets.QVBoxLayout(self.channel_opts_group)
        channel_opts_layout.setContentsMargins(8, 12, 8, 12)
        channel_opts_layout.setSpacing(6)

        self.channel_opts_stack = QtWidgets.QStackedWidget()
        placeholder_widget = QtWidgets.QWidget()
        placeholder_layout = QtWidgets.QVBoxLayout(placeholder_widget)
        placeholder_layout.addStretch(1)
        placeholder_label = QtWidgets.QLabel("Select an active channel to configure.")
        placeholder_label.setAlignment(QtCore.Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: rgb(60,60,60); font-style: italic;")
        placeholder_layout.addWidget(placeholder_label)
        placeholder_layout.addStretch(1)
        self.channel_opts_stack.addWidget(placeholder_widget)
        channel_opts_layout.addWidget(self.channel_opts_stack)

        side_layout.addWidget(self.channel_opts_group, 1)
        grid.addWidget(side_panel, 0, 1, 3, 1)

        # Bottom row (spanning full width): device / channel controls.
        bottom_panel = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        self.device_group = QtWidgets.QGroupBox("Device")
        device_layout = QtWidgets.QGridLayout(self.device_group)
        device_layout.setColumnStretch(1, 1)
        self.device_group.setMaximumWidth(320)

        device_layout.addWidget(self._label("Source"), 0, 0)
        self.device_combo = QtWidgets.QComboBox()
        device_layout.addWidget(self.device_combo, 0, 1)

        device_layout.addWidget(self._label("Sample Rate (Hz)"), 1, 0)
        self.sample_rate_spin = QtWidgets.QDoubleSpinBox()
        self.sample_rate_spin.setRange(100.0, 1_000_000.0)
        self.sample_rate_spin.setDecimals(0)
        self.sample_rate_spin.setSingleStep(1000.0)
        self.sample_rate_spin.setValue(20_000.0)
        self.sample_rate_spin.setStyleSheet("color: rgb(255,255,255); background-color: rgb(0,0,0);")
        device_layout.addWidget(self.sample_rate_spin, 1, 1)

        self.device_toggle_btn = QtWidgets.QPushButton("Connect")
        self.device_toggle_btn.setCheckable(True)
        self.device_toggle_btn.clicked.connect(self._on_device_button_clicked)
        device_layout.addWidget(self.device_toggle_btn, 2, 0, 1, 2)

        listen_row = QtWidgets.QHBoxLayout()
        listen_row.setSpacing(6)
        listen_row.addWidget(self._label("Listen Output:"))
        self.listen_device_combo = QtWidgets.QComboBox()
        self.listen_device_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        listen_row.addWidget(self.listen_device_combo, 1)
        device_layout.addLayout(listen_row, 3, 0, 1, 2)

        self.channels_group = QtWidgets.QGroupBox("Channels")
        channels_layout = QtWidgets.QVBoxLayout(self.channels_group)
        channels_layout.setContentsMargins(8, 12, 8, 12)
        channels_layout.setSpacing(6)

        active_label = self._label("Active")
        channels_layout.addWidget(active_label)
        self.active_list = QtWidgets.QListWidget()
        self.active_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.active_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.active_list.setMinimumHeight(50)
        self.active_list.setMaximumHeight(60)
        channels_layout.addWidget(self.active_list)

        available_row = QtWidgets.QHBoxLayout()
        available_row.setContentsMargins(0, 0, 0, 0)
        available_row.setSpacing(6)
        available_label = self._label("Available")
        available_row.addWidget(available_label)
        available_row.addStretch(1)
        self.add_channel_btn = QtWidgets.QPushButton("↑ Add")
        self.add_channel_btn.clicked.connect(self._on_add_channel)
        available_row.addWidget(self.add_channel_btn)
        self.remove_channel_btn = QtWidgets.QPushButton("↓ Remove")
        self.remove_channel_btn.clicked.connect(self._on_remove_channel)
        available_row.addWidget(self.remove_channel_btn)
        channels_layout.addLayout(available_row)
        self.available_combo = QtWidgets.QComboBox()
        channels_layout.addWidget(self.available_combo)

        bottom_layout.addWidget(self.device_group, 1)
        bottom_layout.addWidget(self.channels_group, 2)

        grid.addWidget(bottom_panel, 2, 0)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 0)
        grid.setRowStretch(2, 0)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 3)

        return scope

    def _wire_placeholders(self) -> None:
        """Connect stub widgets to stub slots for future wiring."""
        self.trigger_channel_combo.currentTextChanged.connect(self._emit_trigger_config)
        self.trigger_mode_continuous.toggled.connect(self._emit_trigger_config)
        self.trigger_mode_single.toggled.connect(self._emit_trigger_config)
        self.trigger_mode_repeating.toggled.connect(self._emit_trigger_config)
        self.trigger_mode_continuous.toggled.connect(self._on_trigger_mode_changed)
        self.trigger_mode_single.toggled.connect(self._on_trigger_mode_changed)
        self.trigger_mode_repeating.toggled.connect(self._on_trigger_mode_changed)
        self.threshold_spin.valueChanged.connect(self._emit_trigger_config)
        self.pretrigger_combo.currentIndexChanged.connect(self._emit_trigger_config)
        self.window_combo.currentIndexChanged.connect(self._on_window_changed)
        self.threshold_line.sigPositionChanged.connect(self._on_threshold_line_changed)
        self.active_list.currentItemChanged.connect(self._on_active_channel_selected)
        self.trigger_single_button.clicked.connect(self._on_trigger_single_clicked)
        if hasattr(self, "listen_device_combo"):
            self.listen_device_combo.currentIndexChanged.connect(self._on_listen_device_changed)

    def _populate_audio_outputs(self) -> None:
        if not hasattr(self, "listen_device_combo"):
            return
        options = [{"id": None, "label": "System Default"}]
        try:
            devices = list_output_devices()
        except Exception:
            devices = []
        for entry in devices:
            label = entry.get("label") or entry.get("name") or str(entry.get("id"))
            options.append({"id": entry.get("id"), "label": str(label)})
        seen = set()
        final: List[Dict[str, object]] = []
        for opt in options:
            key = opt.get("id")
            if key in seen:
                continue
            seen.add(key)
            final.append(opt)
        self._audio_devices = final
        combo = self.listen_device_combo
        combo.blockSignals(True)
        combo.clear()
        for opt in final:
            combo.addItem(opt["label"], opt)
        if final:
            combo.setCurrentIndex(0)
        combo.setEnabled(len(final) > 0)
        combo.blockSignals(False)

    def attach_controller(self, controller: Optional[PipelineController]) -> None:
        if controller is self._controller:
            return

        if self._controller is not None:
            if self._dispatcher_signals is not None:
                try:
                    self._dispatcher_signals.tick.disconnect(self._on_dispatcher_tick)
                except (TypeError, RuntimeError):
                    pass
                self._dispatcher_signals = None
            try:
                self.startRecording.disconnect(self._controller.start_recording)
            except (TypeError, RuntimeError):
                pass
            try:
                self.stopRecording.disconnect(self._controller.stop_recording)
            except (TypeError, RuntimeError):
                pass
            try:
                self.triggerConfigChanged.disconnect(self._controller.update_trigger_config)
            except (TypeError, RuntimeError):
                pass
            signals = self._controller.dispatcher_signals()
            if signals is not None:
                try:
                    signals.tick.disconnect(self._on_dispatcher_tick)
                except (TypeError, RuntimeError):
                    pass
            self._stop_visualization_drain()

        self._controller = controller

        if controller is None:
            return

        self.startRecording.connect(controller.start_recording)
        self.stopRecording.connect(controller.stop_recording)
        self.triggerConfigChanged.connect(controller.update_trigger_config)

        self._bind_dispatcher_signals()
        self._ensure_audio_router()
        self._update_status(0)

    def _bind_dispatcher_signals(self) -> None:
        if self._controller is None:
            self._dispatcher_signals = None
            return
        signals = self._controller.dispatcher_signals()
        if signals is None:
            if self._dispatcher_signals is not None:
                try:
                    self._dispatcher_signals.tick.disconnect(self._on_dispatcher_tick)
                except (TypeError, RuntimeError):
                    pass
            self._dispatcher_signals = None
            return
        if self._dispatcher_signals is signals:
            return
        if self._dispatcher_signals is not None:
            try:
                self._dispatcher_signals.tick.disconnect(self._on_dispatcher_tick)
            except (TypeError, RuntimeError):
                pass
        try:
            signals.tick.connect(self._on_dispatcher_tick)
        except (TypeError, RuntimeError):
            return
        self._dispatcher_signals = signals

    def _apply_palette(self) -> None:
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(200, 200, 200))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(223, 223, 223))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(200, 200, 200))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 220))
        palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(223, 223, 223))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(30, 144, 255))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            QMainWindow { background-color: rgb(200,200,200); }
            QWidget { color: rgb(0,0,0); }
            QGroupBox {
                background-color: rgb(223,223,223);
                border: 1px solid rgb(120, 120, 120);
                border-radius: 4px;
                margin-top: 12px;
                padding: 6px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0px 4px 0px 4px;
                color: rgb(128, 0, 0);
            }
            QLabel { color: rgb(0,0,0); }
            QPushButton {
                background-color: rgb(223,223,223);
                color: rgb(0,0,0);
                border: 1px solid rgb(120,120,120);
                padding: 4px 8px;
            }
            QPushButton:checked {
                background-color: rgb(200,200,200);
            }
            QLineEdit,
            QPlainTextEdit,
            QTextEdit,
            QAbstractSpinBox,
            QComboBox {
                color: rgb(0,0,0);
                background-color: rgb(245,245,245);
                selection-background-color: rgb(30,144,255);
                selection-color: rgb(255,255,255);
                border: 1px solid rgb(120,120,120);
                padding: 2px 4px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid rgb(120,120,120);
                background-color: rgb(223,223,223);
            }
            QComboBox QAbstractItemView {
                color: rgb(0,0,0);
                background-color: rgb(245,245,245);
                selection-background-color: rgb(30,144,255);
                selection-color: rgb(255,255,255);
            }
            QListView,
            QListWidget {
                color: rgb(0,0,0);
                background-color: rgb(245,245,245);
                selection-background-color: rgb(30,144,255);
                selection-color: rgb(255,255,255);
                border: 1px solid rgb(120,120,120);
            }
            QCheckBox,
            QRadioButton {
                color: rgb(0,0,0);
            }
            QSlider::groove:horizontal {
                border: 1px solid rgb(120,120,120);
                height: 6px;
                background: rgb(200,200,200);
                margin: 0px;
            }
            QSlider::handle:horizontal {
                background: rgb(30,144,255);
                border: 1px solid rgb(0,0,0);
                width: 14px;
                margin: -4px 0;
            }
            QStatusBar { background-color: rgb(192,192,192); }
            """
        )

    def _style_plot(self) -> None:
        pg.setConfigOption("foreground", (0, 0, 139))
        pg.setConfigOptions(antialias=False)

    def _label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        return label

    def _emit_trigger_config(self, *_) -> None:
        data = self.trigger_channel_combo.currentData()
        channel_id = int(data) if data is not None else None
        ui_mode = self._current_trigger_mode()
        self._trigger_channel_id = channel_id
        self._trigger_mode = ui_mode
        self._trigger_threshold = float(self.threshold_spin.value())
        pre_value = self.pretrigger_combo.currentData()
        self._trigger_pre_seconds = float(pre_value if pre_value is not None else 0.0)
        self._reset_trigger_state()

        idx = channel_id if channel_id is not None else -1
        visual_config = {
            "channel_index": idx,
            "mode": ui_mode,
            "threshold": self.threshold_spin.value(),
            "hysteresis": 0.0,
            "pretrigger_frac": self._trigger_pre_seconds,
            "window_sec": float(self.window_combo.currentData() or 0.0),
        }
        self._update_trigger_visuals(visual_config)

        self.triggerConfigChanged.emit(dict(visual_config))

    def _current_trigger_mode(self) -> str:
        if self.trigger_mode_repeating.isChecked():
            return "continuous"
        if self.trigger_mode_single.isChecked():
            return "single"
        return "stream"

    def _on_device_button_clicked(self) -> None:
        if not hasattr(self, "_device_manager") or self._device_manager is None:
            return
        if self._device_connected:
            self._device_manager.disconnect_device()
            return
        key = self.device_combo.currentData()
        if not key:
            QtWidgets.QMessageBox.information(self, "Device", "Please select a device to connect.")
            self.device_toggle_btn.setChecked(False)
            return
        entry = self._device_map.get(key)
        if entry is not None and not entry.get("device_id"):
            message = entry.get("error") or "No hardware devices detected for this driver."
            QtWidgets.QMessageBox.information(self, "Device", message)
            self.device_toggle_btn.setChecked(False)
            return
        try:
            self._device_manager.connect_device(key, sample_rate=self.sample_rate_spin.value())
        except Exception as exc:  # pragma: no cover - GUI feedback only
            QtWidgets.QMessageBox.critical(self, "Device", f"Failed to connect: {exc}")
            self.device_toggle_btn.blockSignals(True)
            self.device_toggle_btn.setChecked(False)
            self.device_toggle_btn.blockSignals(False)

    def _on_devices_changed(self, entries: List[dict]) -> None:
        self._device_map = {entry["key"]: entry for entry in entries}
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for entry in entries:
            key = entry.get("key")
            name = entry.get("name") or str(key)
            self.device_combo.addItem(name, key)
            idx = self.device_combo.count() - 1
            tooltip_lines = []
            driver_name = entry.get("driver_name")
            device_name = entry.get("device_name")
            if driver_name and device_name:
                tooltip_lines.append(f"{driver_name} device: {device_name}")
            elif driver_name:
                tooltip_lines.append(driver_name)
            vendor = entry.get("device_vendor")
            if vendor:
                tooltip_lines.append(f"Vendor: {vendor}")
            details = entry.get("device_details")
            if isinstance(details, dict):
                for d_key, d_val in details.items():
                    tooltip_lines.append(f"{d_key}: {d_val}")
            error_text = entry.get("error")
            if error_text:
                tooltip_lines.append(str(error_text))
            if tooltip_lines:
                tooltip = "\n".join(str(line) for line in tooltip_lines)
                self.device_combo.setItemData(idx, tooltip, QtCore.Qt.ToolTipRole)
        self.device_combo.blockSignals(False)
        if self._device_connected:
            active_key = self._device_manager.active_key()
            if active_key is not None:
                idx = self.device_combo.findData(active_key)
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)
        self._apply_device_state(self._device_connected and bool(entries))
        self._update_channel_buttons()

    def _on_device_connected(self, key: str) -> None:
        self._device_connected = True
        self._apply_device_state(True)
        idx = self.device_combo.findData(key)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        if self._controller is not None:
            driver = self._device_manager.current_driver()
            if driver is not None:
                channels = self._device_manager.get_available_channels()
                self._controller.attach_source(driver, self.sample_rate_spin.value(), channels)
                self._bind_dispatcher_signals()
        self._update_channel_buttons()

    def _on_device_disconnected(self) -> None:
        self._device_connected = False
        self._apply_device_state(False)
        if self._controller is not None:
            self._controller.detach_device()
            self._controller.clear_active_channels()
        if self._dispatcher_signals is not None:
            try:
                self._dispatcher_signals.tick.disconnect(self._on_dispatcher_tick)
            except (TypeError, RuntimeError):
                pass
            self._dispatcher_signals = None
        self._clear_listen_channel()
        self._reset_trigger_state()
        self.available_combo.clear()
        self.active_list.clear()
        self._clear_channel_panels()
        self.set_trigger_channels([])
        self._update_channel_buttons()
        self._publish_active_channels()

    def _on_available_channels(self, channels: Sequence[object]) -> None:
        self.available_combo.clear()
        self.active_list.clear()
        self._clear_channel_panels()
        for info in channels:
            name = getattr(info, "name", str(info))
            self.available_combo.addItem(name, info)
        if self.available_combo.count():
            self.available_combo.setCurrentIndex(0)
        self.set_trigger_channels(channels)
        self._update_channel_buttons()
        self._publish_active_channels()

    def _on_add_channel(self) -> None:
        idx = self.available_combo.currentIndex()
        if idx < 0:
            return
        info = self.available_combo.itemData(idx)
        item = QtWidgets.QListWidgetItem(self.available_combo.currentText())
        item.setData(QtCore.Qt.UserRole, info)
        self.active_list.addItem(item)
        self.available_combo.removeItem(idx)
        if self.available_combo.count():
            self.available_combo.setCurrentIndex(min(idx, self.available_combo.count() - 1))
        self.active_list.setCurrentItem(item)
        self._update_channel_buttons()
        self._publish_active_channels()
        self._emit_trigger_config()

    def _on_remove_channel(self) -> None:
        current = self.active_list.currentItem()
        if current is None:
            return
        info = current.data(QtCore.Qt.UserRole)
        item = QtWidgets.QListWidgetItem(current.text())
        item.setData(QtCore.Qt.UserRole, info)
        self.available_combo.addItem(item.text(), info)
        row = self.active_list.row(current)
        self.active_list.takeItem(row)
        if self.active_list.count() > 0:
            self.active_list.setCurrentRow(0)
        else:
            self._show_channel_panel(None)
        if self.available_combo.count():
            self.available_combo.setCurrentIndex(self.available_combo.count() - 1)
        self._update_channel_buttons()
        self._publish_active_channels()
        self._emit_trigger_config()

    def _publish_active_channels(self) -> None:
        infos = []
        for index in range(self.active_list.count()):
            item = self.active_list.item(index)
            info = item.data(QtCore.Qt.UserRole)
            if info is not None:
                infos.append(info)
        self._active_channel_infos = infos
        self._update_channel_buttons()
        ids = [getattr(info, "id", None) for info in infos]
        ids = [cid for cid in ids if cid is not None]
        names = [getattr(info, "name", str(info)) for info in infos]
        if list(ids) != self._channel_ids_current:
            self._reset_trigger_state()
        self._sync_channel_panels(ids, names)
        self._reset_scope_for_channels(ids, names)
        self._sync_filter_settings()
        self._ensure_active_channel_focus()
        self._channel_last_samples = {cid: self._channel_last_samples[cid] for cid in ids if cid in self._channel_last_samples}
        self._channel_display_buffers = {cid: self._channel_display_buffers[cid] for cid in ids if cid in self._channel_display_buffers}
        if self._listen_channel_id is not None and self._listen_channel_id not in ids:
            self._clear_listen_channel()
        self.trigger_mode_continuous.setChecked(True)
        if infos:
            self.set_trigger_channels(infos)
        else:
            self.trigger_channel_combo.blockSignals(True)
            self.trigger_channel_combo.clear()
            self.trigger_channel_combo.blockSignals(False)
            self._emit_trigger_config()
        if self._controller is not None:
            if ids:
                self._controller.set_active_channels(ids)
            else:
                self._controller.clear_active_channels()

    def _next_channel_color(self) -> QtGui.QColor:
        color = self._channel_color_cycle[self._next_color_index % len(self._channel_color_cycle)]
        self._next_color_index += 1
        return QtGui.QColor(color)

    def _ensure_channel_config(self, channel_id: int, channel_name: str) -> ChannelConfig:
        config = self._channel_configs.get(channel_id)
        if config is None:
            config = ChannelConfig(color=self._next_channel_color(), channel_name=channel_name)
            self._channel_configs[channel_id] = config
        else:
            config.channel_name = channel_name
        return config

    def _sync_channel_panels(self, channel_ids: Sequence[int], channel_names: Sequence[str]) -> None:
        desired = {cid: name for cid, name in zip(channel_ids, channel_names)}
        for cid, panel in list(self._channel_panels.items()):
            if cid not in desired:
                config = self._channel_configs.get(cid)
                if config is not None and hasattr(self, "_analysis_dock"):
                    channel_name = config.channel_name or f"Channel {cid}"
                    self._analysis_dock.close_tab(channel_name)
                self.channel_opts_stack.removeWidget(panel)
                panel.deleteLater()
                del self._channel_panels[cid]
                self._channel_configs.pop(cid, None)
        if not channel_ids:
            self._show_channel_panel(None)
            return
        for cid, name in desired.items():
            config = self._ensure_channel_config(cid, name)
            panel = self._channel_panels.get(cid)
            if panel is None:
                panel = ChannelOptionsPanel(cid, name, self.channel_opts_stack)
                panel.configChanged.connect(lambda cfg, cid=cid: self._on_channel_config_changed(cid, cfg))
                panel.analysisRequested.connect(lambda cid=cid: self._open_analysis_for_channel(cid))
                self.channel_opts_stack.addWidget(panel)
                self._channel_panels[cid] = panel
            panel.set_channel_name(name)
            panel.set_config(config)
        current = self.active_list.currentItem()
        if current is not None:
            info = current.data(QtCore.Qt.UserRole)
            cid = getattr(info, "id", None)
            self._show_channel_panel(cid)
        else:
            self._show_channel_panel(channel_ids[0] if channel_ids else None)

    def _clear_channel_panels(self) -> None:
        for panel in list(self._channel_panels.values()):
            self.channel_opts_stack.removeWidget(panel)
            panel.deleteLater()
        self._channel_panels.clear()
        self._channel_configs.clear()
        self._channel_last_samples.clear()
        self._channel_display_buffers.clear()
        self._show_channel_panel(None)

    def _show_channel_panel(self, channel_id: Optional[int]) -> None:
        if channel_id is None:
            self.channel_opts_stack.setCurrentIndex(0)
            return
        panel = self._channel_panels.get(channel_id)
        if panel is None:
            self.channel_opts_stack.setCurrentIndex(0)
            return
        index = self.channel_opts_stack.indexOf(panel)
        if index >= 0:
            self.channel_opts_stack.setCurrentIndex(index)

    def _on_active_channel_selected(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        previous: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        _ = previous
        if current is None:
            self._show_channel_panel(None)
            return
        info = current.data(QtCore.Qt.UserRole)
        channel_id = getattr(info, "id", None)
        name = getattr(info, "name", str(info))
        if channel_id is None:
            self._show_channel_panel(None)
            return
        config = self._ensure_channel_config(channel_id, name)
        panel = self._channel_panels.get(channel_id)
        if panel is None:
            panel = ChannelOptionsPanel(channel_id, name, self.channel_opts_stack)
            panel.configChanged.connect(lambda cfg, cid=channel_id: self._on_channel_config_changed(cid, cfg))
            panel.analysisRequested.connect(lambda cid=channel_id: self._open_analysis_for_channel(cid))
            self.channel_opts_stack.addWidget(panel)
            self._channel_panels[channel_id] = panel
        panel.set_channel_name(name)
        panel.set_config(config)
        self._set_active_channel_focus(channel_id)
        self._show_channel_panel(channel_id)

    def _select_active_channel_by_id(self, channel_id: int) -> None:
        target_row = None
        for idx in range(self.active_list.count()):
            item = self.active_list.item(idx)
            info = item.data(QtCore.Qt.UserRole)
            if getattr(info, "id", None) == channel_id:
                target_row = idx
                break
        if target_row is None:
            return
        if self.active_list.currentRow() != target_row:
            self.active_list.setCurrentRow(target_row)
        else:
            self._set_active_channel_focus(channel_id)
            self._show_channel_panel(channel_id)

    def _nearest_channel_at_y(self, y: float) -> Optional[int]:
        """Return the channel whose configured offset is closest to the given y view coordinate."""
        candidates: list[tuple[float, int]] = []
        for cid in self._channel_ids_current:
            config = self._channel_configs.get(cid)
            if config is None:
                continue
            center = config.offset_v
            span = max(config.range_v, 0.1)
            if abs(y - center) <= span:
                candidates.append((abs(y - center), cid))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _on_plot_channel_clicked(self, y: float, button: QtCore.Qt.MouseButton) -> None:
        if button != QtCore.Qt.MouseButton.LeftButton:
            return
        cid = self._nearest_channel_at_y(y)
        if cid is None:
            self._drag_channel_id = None
            return
        self._drag_channel_id = cid
        self._select_active_channel_by_id(cid)
        self._set_active_channel_focus(cid)

    def _on_plot_channel_dragged(self, y: float) -> None:
        if self._drag_channel_id is None:
            return
        config = self._channel_configs.get(self._drag_channel_id)
        if config is None:
            return
        if abs(config.offset_v - y) < 1e-6:
            return
        config.offset_v = y
        panel = self._channel_panels.get(self._drag_channel_id)
        if panel is not None:
            panel.set_config(config)
        self._update_channel_display(self._drag_channel_id)
        self._update_plot_y_range()

    def _on_plot_drag_finished(self) -> None:
        self._drag_channel_id = None

    def _on_channel_config_changed(self, channel_id: int, config: ChannelConfig) -> None:
        existing = self._channel_configs.get(channel_id)
        if existing is not None:
            config.channel_name = existing.channel_name or config.channel_name
        display_changed = existing is not None and existing.display_enabled != config.display_enabled
        panel = self._channel_panels.get(channel_id)
        if panel is not None:
            panel.set_config(config)
        self._channel_configs[channel_id] = config
        if display_changed:
            if not config.display_enabled:
                curve = self._curve_map.get(channel_id)
                if curve is not None:
                    curve.clear()
                self._channel_last_samples.pop(channel_id, None)
                self._channel_display_buffers.pop(channel_id, None)
            else:
                self._channel_last_samples.clear()
                self._channel_display_buffers.clear()
                self._last_times = np.zeros(0, dtype=np.float32)
        self._update_channel_display(channel_id)
        self._refresh_channel_layout()
        self._sync_filter_settings()
        if self._active_channel_id == channel_id:
            self._update_axis_label()
        self._handle_listen_change(channel_id, config.listen_enabled)
        if display_changed and config.display_enabled and self._channel_ids_current and self._channel_names:
            self._reset_scope_for_channels(self._channel_ids_current, self._channel_names)

    def _update_channel_display(self, channel_id: int) -> None:
        """Re-render a single channel's curve using the last raw samples and current offset/range."""
        if channel_id not in self._channel_configs:
            return
        config = self._channel_configs[channel_id]
        curve = self._curve_map.get(channel_id)
        raw = self._channel_last_samples.get(channel_id)
        if curve is None:
            return
        if not config.display_enabled:
            curve.clear()
            return
        if raw is None or raw.size == 0 or self._last_times.size == 0:
            return
        buf = self._channel_display_buffers.get(channel_id)
        if buf is None or buf.shape != raw.shape:
            buf = np.empty_like(raw)
            self._channel_display_buffers[channel_id] = buf
        np.add(raw, config.offset_v, out=buf)
        curve.setData(self._last_times, buf, skipFiniteCheck=True)
        self._apply_active_channel_style()

    def _apply_active_channel_style(self) -> None:
        for cid, curve in self._curve_map.items():
            config = self._channel_configs.get(cid)
            if config is None or curve is None:
                continue
            is_active = cid == self._active_channel_id
            pen = pg.mkPen(config.color, width=3.0 if is_active else 1.6)
            curve.setPen(pen)
            curve.setZValue(1.0 if is_active else 0.0)
            try:
                curve.setOpacity(1.0 if is_active else 0.6)
            except AttributeError:
                pass
        self._update_axis_label()

    def _handle_listen_change(self, channel_id: int, enabled: bool) -> None:
        """Apply listen toggle semantics (single selection) and spin up/tear down audio plumbing."""
        if enabled:
            self._set_listen_channel(channel_id)
        else:
            if self._listen_channel_id == channel_id:
                self._clear_listen_channel(channel_id)

    def _selected_output_device(self) -> Optional[object]:
        combo = getattr(self, "listen_device_combo", None)
        if combo is None:
            return None
        data = combo.currentData()
        if isinstance(data, dict):
            return data.get("id")
        return None

    def _on_listen_device_changed(self) -> None:
        current = self._listen_channel_id
        if current is None:
            return
        self._stop_audio_player()
        if current not in self._channel_ids_current:
            return
        if not self._ensure_audio_player(show_error=False):
            self._clear_listen_channel(current)
        else:
            self._flush_audio_player_queue()

    def _open_analysis_for_channel(self, channel_id: int) -> None:
        dock = getattr(self, "_analysis_dock", None)
        if dock is None:
            return
        config = self._channel_configs.get(channel_id)
        if config is None:
            return
        channel_name = config.channel_name or f"Channel {channel_id}"
        dock.open_analysis(channel_name)

    def _set_listen_channel(self, channel_id: int) -> None:
        """Activate audio monitoring for the requested channel, disabling other listen toggles."""
        cfg = self._channel_configs.get(channel_id)
        if cfg is None:
            return
        if self._current_sample_rate <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "Audio Output",
                "Audio output becomes available after streaming starts.",
            )
            cfg.listen_enabled = False
            panel = self._channel_panels.get(channel_id)
            if panel is not None:
                panel.set_config(cfg)
            return

        for cid, other_cfg in self._channel_configs.items():
            if cid == channel_id:
                continue
            if other_cfg.listen_enabled:
                other_cfg.listen_enabled = False
                panel = self._channel_panels.get(cid)
                if panel is not None:
                    panel.set_config(other_cfg)

        cfg.listen_enabled = True
        panel = self._channel_panels.get(channel_id)
        if panel is not None:
            panel.set_config(cfg)

        with self._audio_lock:
            self._listen_channel_id = channel_id

        self._ensure_audio_router()
        if not self._ensure_audio_player(show_error=True):
            cfg.listen_enabled = False
            if panel is not None:
                panel.set_config(cfg)
            with self._audio_lock:
                self._listen_channel_id = None
            return
        self._flush_audio_player_queue()

    def _clear_listen_channel(self, channel_id: Optional[int] = None) -> None:
        """Disable audio monitoring, optionally for a specific channel."""
        target = channel_id if channel_id is not None else self._listen_channel_id
        if target is None:
            return
        cfg = self._channel_configs.get(target)
        if cfg is not None and cfg.listen_enabled:
            cfg.listen_enabled = False
            panel = self._channel_panels.get(target)
            if panel is not None:
                panel.set_config(cfg)
        with self._audio_lock:
            if self._listen_channel_id == target:
                self._listen_channel_id = None
                stop_audio = True
            else:
                stop_audio = False
        if stop_audio:
            self._stop_audio_player()

    def _ensure_audio_router(self) -> None:
        """Start the background thread that forwards dispatcher audio chunks when needed."""
        if self._audio_router_thread is not None and self._audio_router_thread.is_alive():
            return
        self._audio_router_stop.clear()
        thread = threading.Thread(target=self._audio_router_loop, name="AudioRouter", daemon=True)
        thread.start()
        self._audio_router_thread = thread

    def _stop_audio_router(self) -> None:
        """Stop the audio routing thread if it is currently running."""
        if self._audio_router_thread is None:
            return
        self._audio_router_stop.set()
        self._audio_router_thread.join(timeout=1.0)
        self._audio_router_thread = None
        self._audio_router_stop = threading.Event()

    def _audio_router_loop(self) -> None:
        """Pass chunks from the dispatcher audio queue into the AudioPlayer following listen selection."""
        while not self._audio_router_stop.is_set():
            try:
                controller = self._controller
                if controller is None:
                    self._audio_router_stop.wait(0.1)
                    continue
                try:
                    item = controller.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is EndOfStream or not isinstance(item, Chunk):
                    continue
                samples = np.asarray(item.samples, dtype=np.float32)
                if samples.ndim != 2 or samples.size == 0:
                    continue
                with self._audio_lock:
                    listen_id = self._listen_channel_id
                if listen_id is None or listen_id not in self._channel_ids_current:
                    continue
                if not self._ensure_audio_player():
                    continue
                try:
                    idx = self._channel_ids_current.index(listen_id)
                except ValueError:
                    continue
                if idx >= samples.shape[0]:
                    continue
                mono = samples[idx].astype(np.float32, copy=False)
                if mono.ndim == 1:
                    payload = mono[:, None]
                else:
                    payload = mono.T
                with self._audio_lock:
                    player_queue = self._audio_player_queue
                if player_queue is None:
                    continue
                try:
                    player_queue.put_nowait(payload)
                except queue.Full:
                    pass
            except Exception:
                continue

    def _ensure_audio_player(self, *, show_error: bool = False) -> bool:
        """Create or reconfigure the AudioPlayer to match the current listen stream."""
        sample_rate = int(round(self._current_sample_rate))
        if sample_rate <= 0:
            return False
        device_id = self._selected_output_device()
        with self._audio_lock:
            if (
                self._audio_player is not None
                and abs(self._audio_input_samplerate - sample_rate) < 1e-6
                and self._audio_current_device == device_id
            ):
                return True
            player_to_stop = self._audio_player
            self._audio_player = None
            self._audio_player_queue = None
            self._audio_input_samplerate = 0.0
            self._audio_current_device = None
        if player_to_stop is not None:
            try:
                player_to_stop.stop()
                player_to_stop.join(timeout=1.0)
            except Exception:
                pass
        queue_obj: "queue.Queue" = queue.Queue(maxsize=128)
        config = AudioConfig(
            out_samplerate=44_100,
            out_channels=1,
            device=device_id,
            gain=0.7,
            blocksize=512,
            ring_seconds=0.5,
        )
        try:
            player = AudioPlayer(
                audio_queue=queue_obj,
                input_samplerate=sample_rate,
                config=config,
                selected_channel=0,
            )
        except Exception as exc:
            if show_error:
                QtWidgets.QMessageBox.warning(self, "Audio Output", f"Unable to start audio output: {exc}")
            return False
        with self._audio_lock:
            self._audio_player = player
            self._audio_player_queue = queue_obj
            self._audio_input_samplerate = float(sample_rate)
            self._audio_current_device = device_id
        self._audio_player.start()
        return True

    def _stop_audio_player(self) -> None:
        with self._audio_lock:
            player = self._audio_player
            self._audio_player = None
            self._audio_player_queue = None
            self._audio_input_samplerate = 0.0
            self._audio_current_device = None
        if player is None:
            return
        try:
            player.stop()
            player.join(timeout=1.0)
        except Exception:
            pass

    def _flush_audio_player_queue(self) -> None:
        """Drop any buffered audio frames so a new listen target starts immediately."""
        with self._audio_lock:
            q = self._audio_player_queue
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _ensure_active_channel_focus(self) -> None:
        if self._channel_ids_current:
            if self._active_channel_id not in self._channel_ids_current:
                self._active_channel_id = self._channel_ids_current[0]
        else:
            self._active_channel_id = None
        self._apply_active_channel_style()

    def _set_active_channel_focus(self, channel_id: Optional[int]) -> None:
        if channel_id is not None and channel_id not in self._channel_ids_current:
            return
        if self._active_channel_id == channel_id:
            self._update_axis_label()
            return
        self._active_channel_id = channel_id
        self._apply_active_channel_style()
        self._update_plot_y_range()

    def _refresh_channel_layout(self) -> None:
        """Rebuild plot curves to track active channels and refresh styling."""
        plot_item = self.plot_widget.getPlotItem()
        if not self._channel_ids_current:
            for curve in self._curve_map.values():
                plot_item.removeItem(curve)
            self._curves = []
            self._curve_map.clear()
            self._update_plot_y_range()
            return

        # Remove orphaned curves
        for cid, curve in list(self._curve_map.items()):
            if cid not in self._channel_ids_current:
                plot_item.removeItem(curve)
                del self._curve_map[cid]

        curves: List[pg.PlotCurveItem] = []
        for cid, name in zip(self._channel_ids_current, self._channel_names):
            self._ensure_channel_config(cid, name)
            curve = self._curve_map.get(cid)
            if curve is None:
                curve = pg.PlotCurveItem()
                plot_item.addItem(curve)
                self._curve_map[cid] = curve
            curves.append(curve)
        self._curves = curves
        self._apply_active_channel_style()

        self._update_plot_y_range()
        self._update_axis_label()

    def _register_chunk(self, data: np.ndarray) -> None:
        if data.ndim != 2 or data.size == 0:
            return
        self._chunk_accum_count += 1
        self._chunk_accum_samples += data.shape[1]
        now = time.perf_counter()
        elapsed = now - self._chunk_last_rate_update
        if elapsed >= self._chunk_rate_window:
            self._chunk_rate = self._chunk_accum_count / elapsed if elapsed > 0 else 0.0
            if self._chunk_accum_count > 0:
                self._chunk_mean_samples = self._chunk_accum_samples / self._chunk_accum_count
            else:
                self._chunk_mean_samples = 0.0
            self._chunk_accum_count = 0
            self._chunk_accum_samples = 0
            self._chunk_last_rate_update = now

    def _update_plot_y_range(self) -> None:
        """Apply a symmetric ±range envelope using the selected channel's scale (or first channel)."""
        plot_item = self.plot_widget.getPlotItem()
        span = None
        if self._active_channel_id is not None:
            cfg = self._channel_configs.get(self._active_channel_id)
            if cfg is not None:
                span = max(cfg.range_v, 1e-6)
        if span is None:
            for cid in self._channel_ids_current:
                cfg = self._channel_configs.get(cid)
                if cfg is not None:
                    span = max(cfg.range_v, 1e-6)
                    break
        if span is None:
            plot_item.setYRange(-1.0, 1.0, padding=0)
            return
        min_y = -span
        max_y = span
        plot_item.setYRange(min_y, max_y, padding=0)

    def _update_axis_label(self) -> None:
        axis = self.plot_widget.getPlotItem().getAxis("left")
        if self._active_channel_id is not None:
            config = self._channel_configs.get(self._active_channel_id)
            if config is not None:
                name = config.channel_name or f"Ch {self._active_channel_id}"
                axis_color = QtGui.QColor(config.color)
                rgb = axis_color.getRgb()[:3]
                pen = pg.mkPen(rgb, width=2)
                axis.setPen(pen)
                axis.setTextPen(pen)
                axis.setLabel(text=f"{name} Amplitude (±{config.range_v:.3g} V @ {config.offset_v:.3g} V)", units="V")
                return
        pen = pg.mkPen((0, 0, 139), width=1)
        axis.setPen(pen)
        axis.setTextPen(pen)
        axis.setLabel(text="Amplitude", units="V")

    def _sync_filter_settings(self) -> None:
        controller = self._controller
        if controller is None:
            return
        overrides: Dict[str, ChannelFilterSettings] = {}
        for config in self._channel_configs.values():
            name = config.channel_name
            if not name:
                continue
            if not (config.notch_enabled or config.highpass_enabled or config.lowpass_enabled):
                continue
            overrides[name] = ChannelFilterSettings(
                notch_enabled=config.notch_enabled,
                notch_freq_hz=config.notch_freq,
                lowpass_hz=config.lowpass_freq if config.lowpass_enabled else None,
                highpass_hz=config.highpass_freq if config.highpass_enabled else None,
            )
        base = controller.filter_settings
        settings = FilterSettings(default=base.default, overrides=overrides)
        controller.update_filter_settings(settings)

    def _drain_visualization_queue(self) -> None:
        controller = self._controller
        if controller is None:
            return
        queue_obj = controller.visualization_queue
        while True:
            try:
                item = queue_obj.get_nowait()
            except queue.Empty:
                break
            if item is EndOfStream:
                continue
            try:
                queue_obj.task_done()
            except Exception:
                pass

    def _update_channel_buttons(self) -> None:
        connected = self._device_connected
        self.add_channel_btn.setEnabled(connected and self.available_combo.count() > 0)
        self.remove_channel_btn.setEnabled(connected and self.active_list.count() > 0)
        self._update_trigger_controls()

    def _update_trigger_controls(self) -> None:
        has_active = self.active_list.count() > 0
        for widget in (
            self.trigger_mode_continuous,
            self.trigger_mode_single,
            self.trigger_mode_repeating,
            self.threshold_spin,
            self.pretrigger_combo,
        ):
            widget.setEnabled(has_active)
        self.trigger_single_button.setEnabled(has_active and self.trigger_mode_single.isChecked())

    def _on_browse_record_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select Recording File", "", "HDF5 (*.h5);;All Files (*)")
        if path:
            self.record_path_edit.setText(path)

    def _toggle_recording(self, checked: bool) -> None:
        if checked:
            path = self.record_path_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.information(self, "Recording", "Please choose a file path before recording.")
                self.record_toggle_btn.setChecked(False)
                return
            rollover = self.record_autoinc.isChecked()
            self._apply_record_button_style(True)
            self._set_panels_enabled(False)
            self.startRecording.emit(path, rollover)
        else:
            self._apply_record_button_style(False)
            self._set_panels_enabled(True)
            self.stopRecording.emit()

    def _apply_record_button_style(self, recording: bool) -> None:
        if recording:
            self.record_toggle_btn.setText("Stop Recording (00:00)")
            self.record_toggle_btn.setStyleSheet(
                "background-color: rgb(46,204,113); color: rgb(0,0,0); font-weight: bold;"
            )
        else:
            self.record_toggle_btn.setText("Start Recording")
            self.record_toggle_btn.setStyleSheet(
                "background-color: rgb(220, 20, 60); color: rgb(0,0,0); font-weight: bold;"
            )

    def _set_panels_enabled(self, enabled: bool) -> None:
        for panel in (self.trigger_group, self.device_group, self.channels_group, self.channel_opts_group):
            panel.setEnabled(enabled)
        self.record_path_edit.setEnabled(enabled)
        self.record_autoinc.setEnabled(enabled)
        self.record_toggle_btn.setEnabled(True)
        self.threshold_line.setMovable(enabled)
        self.available_combo.setEnabled(enabled and self._device_connected)
        self.active_list.setEnabled(enabled and self._device_connected)

    def _apply_device_state(self, connected: bool) -> None:
        has_entries = bool(self._device_map)
        has_connectable = any(entry.get("device_id") for entry in self._device_map.values())
        self.device_combo.setEnabled(not connected and has_entries)
        self.sample_rate_spin.setEnabled(not connected)
        self.device_toggle_btn.blockSignals(True)
        self.device_toggle_btn.setChecked(connected)
        self.device_toggle_btn.setText("Disconnect" if connected else "Connect")
        self.device_toggle_btn.setEnabled(connected or has_connectable)
        self.device_toggle_btn.blockSignals(False)
        self.available_combo.setEnabled(connected)
        self.active_list.setEnabled(connected)
        self._update_channel_buttons()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if hasattr(self, "_device_manager") and self._device_manager is not None:
            try:
                self._device_manager.disconnect_device()
            except Exception:
                pass
        self._clear_listen_channel()
        self._stop_audio_router()
        self._stop_audio_player()
        super().closeEvent(event)

    def _quit_application(self) -> None:
        QtWidgets.QApplication.instance().exit()

    def _on_window_changed(self) -> None:
        value = float(self.window_combo.currentData() or 0.0)
        self._current_window_sec = max(value, 1e-3)
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, self._current_window_sec, padding=0.0)
        if self._controller is not None:
            self._controller.update_window_span(self._current_window_sec)
        self._update_status(viz_depth=0)
        if self._trigger_last_sample_rate > 0:
            self._update_trigger_sample_parameters(self._trigger_last_sample_rate)

    def _on_threshold_line_changed(self) -> None:
        value = float(self.threshold_line.value())
        if abs(self.threshold_spin.value() - value) > 1e-6:
            self.threshold_spin.blockSignals(True)
            self.threshold_spin.setValue(value)
            self.threshold_spin.blockSignals(False)
        self._emit_trigger_config()

    def _on_trigger_mode_changed(self) -> None:
        self._update_trigger_controls()

    def _on_trigger_single_clicked(self) -> None:
        if self._trigger_mode != "single":
            self.trigger_mode_single.setChecked(True)
            return
        self._trigger_single_armed = True
        self._trigger_display = None
        self._trigger_display_times = None
        self._trigger_capture_start_abs = None
        self._trigger_capture_end_abs = None
        self._trigger_hold_until = 0.0

    def _reset_trigger_state(self) -> None:
        self._trigger_history.clear()
        self._trigger_history_length = 0
        self._trigger_history_total = 0
        self._trigger_max_chunk = 0
        self._trigger_last_sample_rate = 0.0
        self._trigger_prev_value = 0.0
        self._trigger_capture_start_abs = None
        self._trigger_capture_end_abs = None
        self._trigger_display = None
        self._trigger_display_times = None
        self._trigger_hold_until = 0.0
        self._trigger_display_pre_samples = 0
        if self._trigger_mode != "single":
            self._trigger_single_armed = False
        self.pretrigger_line.setVisible(False)

    def _update_trigger_visuals(self, config: dict) -> None:
        mode = config.get("mode", "stream")
        channel_valid = config.get("channel_index", -1) != -1
        if mode == "stream" or not channel_valid:
            self.threshold_line.setVisible(False)
            self.pretrigger_line.setVisible(False)
            return
        self.threshold_line.setVisible(True)
        pen = pg.mkPen((0, 0, 0), width=3)
        self.threshold_line.setPen(pen)
        try:
            self.threshold_line.setZValue(100)
        except AttributeError:
            pass
        value = float(config.get("threshold", 0.0))
        self.threshold_line.setValue(value)
        pre_value = float(config.get("pretrigger_frac", 0.0) or 0.0)
        if pre_value > 0.0:
            self.pretrigger_line.setVisible(True)
            self.pretrigger_line.setValue(0.0)
        else:
            self.pretrigger_line.setVisible(False)

    def _update_status(self, viz_depth: int) -> None:
        controller = self._controller
        if controller is None:
            stats = {}
            queue_depths: Dict[str, dict] = {}
        else:
            stats = controller.dispatcher_stats()
            queue_depths = controller.queue_depths()

        sr = getattr(self, "_current_sample_rate", 0.0)

        drops = stats.get("dropped", {}) if isinstance(stats, dict) else {}
        evicted = stats.get("evicted", {}) if isinstance(stats, dict) else {}

        now = time.perf_counter()
        if self._chunk_accum_count == 0 and (now - self._chunk_last_rate_update) > (self._chunk_rate_window * 2.0):
            self._chunk_rate = 0.0
            self._chunk_mean_samples = 0.0

        if sr > 0 and self._chunk_mean_samples > 0:
            avg_ms = (self._chunk_mean_samples / sr) * 1_000.0
            chunk_suffix = f"Avg {avg_ms:5.1f} ms"
        elif self._chunk_mean_samples > 0:
            chunk_suffix = f"Avg {self._chunk_mean_samples:5.0f} smp"
        else:
            chunk_suffix = ""

        def _format_queue(info: object, label: str) -> str:
            if not isinstance(info, dict):
                return f"{label}:0/0"
            size = int(info.get("size", 0))
            maxsize = int(info.get("max", 0))
            util = float(info.get("utilization", 0.0)) * 100.0
            if maxsize <= 0:
                return f"{label}:{size}/∞ (0%)"
            return f"{label}:{size}/{maxsize} ({util:3.0f}%)"

        self._status_labels["sr"].setText(
            f"SR: {sr:,.0f} Hz\n"
            f"Drops V:{drops.get('visualization', 0)} "
            f"L:{drops.get('logging', 0)} Evict:{evicted.get('visualization', 0)}"
        )

        chunk_line = f"Chunks/s: {self._chunk_rate:5.1f}"
        if chunk_suffix:
            chunk_line = f"{chunk_line}\n{chunk_suffix}"
        else:
            chunk_line = f"{chunk_line}\n"
        self._status_labels["chunk"].setText(chunk_line)

        viz_queue_text = _format_queue(queue_depths.get("visualization"), "V")

        ring_info = queue_depths.get("viz_buffer", {})
        if isinstance(ring_info, dict):
            ring_seconds = float(ring_info.get("seconds", 0.0))
            ring_capacity_seconds = float(ring_info.get("capacity_seconds", 0.0))
            ring_ms = ring_seconds * 1_000.0
            ring_cap_ms = ring_capacity_seconds * 1_000.0
            if ring_capacity_seconds > 0:
                ring_text = f"History {ring_ms:5.0f}/{ring_cap_ms:5.0f} ms"
            else:
                ring_text = "History 0 ms"
        else:
            ring_text = "History 0 ms"

        analysis_text = _format_queue(queue_depths.get("analysis"), "A")
        audio_text = _format_queue(queue_depths.get("audio"), "Au")
        logging_text = _format_queue(queue_depths.get("logging"), "L")

        self._status_labels["queues"].setText(
            f"Queues {viz_queue_text} {analysis_text} {audio_text} {logging_text}\n{ring_text}"
        )

        self._status_labels["drops"].setText("")

    # ------------------------------------------------------------------
    # Placeholder API (to be implemented later)
    # ------------------------------------------------------------------

    def populate_devices(self, devices: Sequence[str]) -> None:
        """Populate the device panel once selection widgets exist."""
        _ = devices

    def set_active_channels(self, channels: Sequence[str]) -> None:
        """Display the channels currently routed to the plot."""
        names = [getattr(ch, "name", str(ch)) for ch in channels]
        self._ensure_curves(names)
        self.set_trigger_channels(names)

    def set_trigger_channels(self, channels: Sequence[object], *, current: Optional[int] = None) -> None:
        """Update trigger channel choices presented to the user."""
        self.trigger_channel_combo.blockSignals(True)
        self.trigger_channel_combo.clear()
        for entry in channels:
            name = getattr(entry, "name", str(entry))
            cid = getattr(entry, "id", None)
            self.trigger_channel_combo.addItem(name, cid)
        if current is not None:
            idx = self.trigger_channel_combo.findData(current)
            if idx >= 0:
                self.trigger_channel_combo.setCurrentIndex(idx)
        elif self.trigger_channel_combo.count() > 0:
            self.trigger_channel_combo.setCurrentIndex(0)
        self.trigger_channel_combo.blockSignals(False)
        self._emit_trigger_config()

    def _reset_scope_for_channels(self, channel_ids: Sequence[int], channel_names: Sequence[str]) -> None:
        self._ensure_curves_for_ids(channel_ids, channel_names)
        plot_item = self.plot_widget.getPlotItem()
        target_window = float(self.window_combo.currentData() or 1.0)
        self._current_window_sec = max(target_window, 1e-3)
        plot_item.setXRange(0.0, self._current_window_sec, padding=0.0)
        for curve in self._curves:
            curve.clear()
        self._update_plot_y_range()
        self._ensure_active_channel_focus()
        self._current_sample_rate = 0.0
        self._chunk_rate = 0.0
        self._chunk_mean_samples = 0.0
        self._chunk_accum_count = 0
        self._chunk_accum_samples = 0
        self._chunk_last_rate_update = time.perf_counter()
        self._update_status(viz_depth=0)
        if self._controller is not None:
            self._controller.update_window_span(self._current_window_sec)

    def _ensure_curves_for_ids(self, channel_ids: Sequence[int], channel_names: Sequence[str]) -> None:
        """Synchronize the pyqtgraph PlotCurveItems with the current active channel list."""
        plot_item = self.plot_widget.getPlotItem()
        # Remove curves that are no longer needed
        for cid, curve in list(self._curve_map.items()):
            if cid not in channel_ids:
                plot_item.removeItem(curve)
                del self._curve_map[cid]
                self._channel_display_buffers.pop(cid, None)

        self._channel_names = list(channel_names)
        self._channel_ids_current = list(channel_ids)

        self._curves = []
        for cid, name in zip(self._channel_ids_current, self._channel_names):
            config = self._ensure_channel_config(cid, name)
            curve = self._curve_map.get(cid)
            if curve is None:
                curve = pg.PlotCurveItem(pen=pg.mkPen(config.color, width=2.0))
                plot_item.addItem(curve)
                self._curve_map[cid] = curve
            else:
                curve.setPen(pg.mkPen(config.color, width=2.0))
            self._curves.append(curve)

        if not self._channel_ids_current:
            self._update_plot_y_range()
            return

        self._refresh_channel_layout()

    def _process_streaming(
        self,
        data: np.ndarray,
        times_arr: np.ndarray,
        sample_rate: float,
        window_sec: float,
        channel_ids: List[int],
        now: float,
    ) -> None:
        should_redraw = (now - self._last_plot_refresh) >= self._plot_interval
        self.pretrigger_line.setVisible(False)

        if data.ndim != 2 or data.size == 0:
            for curve in self._curves:
                curve.clear()
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            self._chunk_rate = 0.0
            self._chunk_mean_samples = 0.0
            self._chunk_accum_count = 0
            self._chunk_accum_samples = 0
            self._chunk_last_rate_update = now
            self._update_status(viz_depth=0)
            return

        active_samples: Dict[int, np.ndarray] = {}

        for idx, curve in enumerate(self._curves):
            if idx < data.shape[0] and idx < len(channel_ids):
                cid = channel_ids[idx]
                config = self._channel_configs.get(cid)
                if config is None:
                    if should_redraw:
                        curve.clear()
                    continue
                raw = np.asarray(data[idx], dtype=np.float32)
                active_samples[cid] = raw
                if not should_redraw:
                    continue
                if not config.display_enabled:
                    curve.clear()
                    self._channel_display_buffers.pop(cid, None)
                    continue
                buf = self._channel_display_buffers.get(cid)
                if buf is None or buf.shape != raw.shape:
                    buf = np.empty_like(raw)
                    self._channel_display_buffers[cid] = buf
                np.add(raw, config.offset_v, out=buf)
                curve.setData(times_arr, buf, skipFiniteCheck=True)
            else:
                if should_redraw:
                    curve.clear()

        self._channel_last_samples = active_samples
        if should_redraw:
            self._apply_active_channel_style()
            if window_sec > 0:
                self.plot_widget.getPlotItem().setXRange(0, window_sec, padding=0)
            self._update_plot_y_range()
            self._last_plot_refresh = now

        self._current_sample_rate = sample_rate
        self._current_window_sec = window_sec
        self._update_status(viz_depth=0)
        if self._listen_channel_id is not None:
            self._ensure_audio_player()

    def _update_trigger_sample_parameters(self, sample_rate: float) -> None:
        self._trigger_last_sample_rate = sample_rate
        self._trigger_window_samples = max(1, int(round(self._current_window_sec * sample_rate)))
        pre = min(self._trigger_pre_seconds, self._current_window_sec)
        self._trigger_pre_samples = int(round(pre * sample_rate))
        if self._trigger_pre_samples >= self._trigger_window_samples:
            self._trigger_pre_samples = max(0, self._trigger_window_samples - 1)

    def _append_trigger_history(self, chunk_samples: np.ndarray) -> None:
        if chunk_samples.size == 0:
            return
        if not self._trigger_history:
            self._trigger_history = deque()
        self._trigger_history.append(chunk_samples)
        self._trigger_history_length += chunk_samples.shape[0]
        self._trigger_history_total += chunk_samples.shape[0]
        self._trigger_max_chunk = max(self._trigger_max_chunk, chunk_samples.shape[0])
        max_keep = max(self._trigger_window_samples + self._trigger_max_chunk, self._trigger_window_samples + chunk_samples.shape[0])
        while self._trigger_history_length > max_keep and self._trigger_history:
            left = self._trigger_history.popleft()
            self._trigger_history_length -= left.shape[0]

    def _detect_trigger_crossing(self, samples: np.ndarray) -> Optional[int]:
        threshold = float(self.threshold_spin.value())
        prev = self._trigger_prev_value
        for idx, sample in enumerate(samples):
            if prev < threshold <= sample:
                self._trigger_prev_value = float(samples[-1])
                return idx
            prev = sample
        self._trigger_prev_value = float(samples[-1])
        return None

    def _should_arm_trigger(self, now: float) -> bool:
        if self._trigger_display is not None and now < self._trigger_hold_until:
            return False
        if self._trigger_capture_start_abs is not None:
            return False
        if self._trigger_mode == "continuous":
            return True
        if self._trigger_mode == "single":
            return self._trigger_single_armed
        return False

    def _start_trigger_capture(self, chunk_start_abs: int, trigger_idx: int) -> None:
        window = self._trigger_window_samples
        if window <= 0:
            return
        pre = self._trigger_pre_samples
        start_abs = max(chunk_start_abs + trigger_idx - pre, self._trigger_history_total - self._trigger_history_length)
        self._trigger_capture_start_abs = start_abs
        self._trigger_capture_end_abs = start_abs + window
        if self._trigger_mode == "single":
            self._trigger_single_armed = False

    def _finalize_trigger_capture(self) -> None:
        if self._trigger_capture_start_abs is None or self._trigger_capture_end_abs is None:
            return
        if self._trigger_history_total < self._trigger_capture_end_abs:
            return
        if not self._trigger_history:
            return
        earliest_abs = self._trigger_history_total - self._trigger_history_length
        start_abs = max(self._trigger_capture_start_abs, earliest_abs)
        start_idx = start_abs - earliest_abs
        end_idx = start_idx + self._trigger_window_samples
        data = np.concatenate(list(self._trigger_history), axis=0)
        if end_idx > data.shape[0]:
            end_idx = data.shape[0]
        snippet = data[start_idx:end_idx]
        if snippet.shape[0] < self._trigger_window_samples:
            pad = self._trigger_window_samples - snippet.shape[0]
            if snippet.shape[0] > 0:
                last_row = snippet[-1:]
            else:
                last_row = np.zeros((1, data.shape[1]), dtype=np.float32)
            snippet = np.vstack((snippet, np.repeat(last_row, pad, axis=0)))
        if snippet.shape[0] == 0:
            snippet = np.zeros((self._trigger_window_samples, data.shape[1]), dtype=np.float32)
        self._trigger_display = snippet
        self._trigger_display_times = None
        self._trigger_display_pre_samples = min(self._trigger_pre_samples, max(snippet.shape[0] - 1, 0))
        if self._trigger_last_sample_rate > 0:
            duration = self._trigger_window_samples / self._trigger_last_sample_rate
        else:
            duration = self._current_window_sec
        self._trigger_hold_until = time.perf_counter() + max(duration, 1e-3)
        self._trigger_capture_start_abs = None
        self._trigger_capture_end_abs = None

    def _render_trigger_display(self, channel_ids: List[int], window_sec: float) -> None:
        if self._trigger_display is None:
            return
        window = max(window_sec, 1e-6)
        data = self._trigger_display
        n = data.shape[0]
        sr = self._trigger_last_sample_rate if self._trigger_last_sample_rate > 0 else self._current_sample_rate
        if sr <= 0 and window > 0 and n > 0:
            sr = n / window
        if sr <= 0:
            sr = max(n / max(window, 1e-6), 1.0)
        dt = 1.0 / sr
        time_axis = np.arange(n, dtype=np.float32) * float(dt)
        self._trigger_display_times = np.asarray(time_axis, dtype=np.float32)
        time_axis = self._trigger_display_times

        plot_item = self.plot_widget.getPlotItem()
        span = max(window, max(n - 1, 0) * dt)
        plot_item.setXRange(0.0, span, padding=0)

        pre_samples = min(self._trigger_display_pre_samples, n - 1 if n > 0 else 0)
        pre_time = pre_samples * dt
        if pre_time > 0.0:
            self.pretrigger_line.setVisible(True)
            self.pretrigger_line.setValue(pre_time)
        else:
            self.pretrigger_line.setVisible(False)

        for idx, curve in enumerate(self._curves):
            if idx < data.shape[1]:
                cid = channel_ids[idx]
                config = self._channel_configs.get(cid)
                if config is not None and not config.display_enabled:
                    curve.clear()
                    continue
                offset = config.offset_v if config else 0.0
                curve.setData(time_axis, data[:, idx] + offset, skipFiniteCheck=True)
            else:
                curve.clear()
        self._channel_last_samples = {cid: data[:, i].astype(np.float32) for i, cid in enumerate(channel_ids) if i < data.shape[1]}
        self._last_times = time_axis
        self._last_plot_refresh = time.perf_counter()

    def _process_trigger_mode(
        self,
        data: np.ndarray,
        times_arr: np.ndarray,
        sample_rate: float,
        window_sec: float,
        channel_ids: List[int],
        now: float,
    ) -> None:
        if data.ndim != 2 or data.size == 0:
            self._process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)
            return
        if sample_rate > 0 and abs(sample_rate - self._trigger_last_sample_rate) > 1e-6:
            self._update_trigger_sample_parameters(sample_rate)

        chunk_samples = data.T  # shape (samples, channels)
        if chunk_samples.ndim != 2 or chunk_samples.size == 0:
            self._process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)
            return
        self._append_trigger_history(chunk_samples)

        monitor_idx = None
        if self._trigger_channel_id is not None and self._trigger_channel_id in channel_ids:
            monitor_idx = channel_ids.index(self._trigger_channel_id)

        if (
            self._trigger_display is not None
            and self._trigger_mode != "single"
            and now >= self._trigger_hold_until
        ):
            self._trigger_display = None
            self._trigger_display_times = None
            self.pretrigger_line.setVisible(False)
            self._trigger_hold_until = 0.0

        if monitor_idx is not None and self._should_arm_trigger(now) and self._trigger_display is None:
            cross_idx = self._detect_trigger_crossing(chunk_samples[:, monitor_idx])
            if cross_idx is not None:
                chunk_start_abs = self._trigger_history_total - chunk_samples.shape[0]
                self._start_trigger_capture(chunk_start_abs, cross_idx)
        elif monitor_idx is not None and self._trigger_display is None:
            # Maintain previous value even if not armed
            self._detect_trigger_crossing(chunk_samples[:, monitor_idx])

        self._finalize_trigger_capture()

        if self._trigger_display is not None:
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            self._render_trigger_display(channel_ids, window_sec)
            self._update_status(viz_depth=0)
            if self._listen_channel_id is not None:
                self._ensure_audio_player()
            return

        if self._trigger_mode == "single":
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            self._update_status(viz_depth=0)
            if self._listen_channel_id is not None:
                self._ensure_audio_player()
            return

        self._process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)


    @QtCore.Slot(dict)
    def _on_dispatcher_tick(self, payload: dict) -> None:
        samples = payload.get("samples")
        times = payload.get("times")
        status = payload.get("status", {})
        sample_rate = float(status.get("sample_rate", 0.0))
        window_sec = float(status.get("window_sec", 0.0))
        channel_ids = list(payload.get("channel_ids", []))
        channel_names = list(payload.get("channel_names", []))

        data = np.asarray(samples) if samples is not None else np.zeros((0, 0), dtype=np.float32)
        times_arr = np.asarray(times) if times is not None else np.zeros(0, dtype=np.float32)
        self._last_times = times_arr
        now = time.perf_counter()

        self._register_chunk(data)
        self._drain_visualization_queue()

        if channel_ids != self._channel_ids_current:
            self._ensure_curves_for_ids(channel_ids, channel_names)

        if not self._curves:
            self.plot_widget.getPlotItem().setXRange(0, max(window_sec, 0.001), padding=0)
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            self._chunk_rate = 0.0
            self._chunk_mean_samples = 0.0
            self._chunk_accum_count = 0
            self._chunk_accum_samples = 0
            self._chunk_last_rate_update = time.perf_counter()
            self._update_status(viz_depth=0)
            return
        mode = self._trigger_mode or "stream"
        if mode == "stream":
            self._process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)
        else:
            self._process_trigger_mode(data, times_arr, sample_rate, window_sec, channel_ids, now)

    @QtCore.Slot(bool)
    def on_record_toggled(self, enabled: bool) -> None:
        """Placeholder slot that will route a record toggle to the controller."""
        self.recordToggled.emit(enabled)
