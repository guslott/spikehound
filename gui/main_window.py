from __future__ import annotations

import queue
import threading

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from core import DeviceManager, PipelineController
from core.conditioning import ChannelFilterSettings, FilterSettings
from core.models import Chunk, EndOfStream
from audio.player import AudioPlayer, AudioConfig


@dataclass
class ChannelConfig:
    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(0, 0, 139))
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

    def __init__(self, channel_id: int, channel_name: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._channel_id = channel_id
        self._config = ChannelConfig(channel_name=channel_name)
        self._block_updates = False

        self._build_ui()
        self.set_channel_name(channel_name)
        self.set_config(self._config)

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
        color_row.addStretch(1)
        layout.addLayout(color_row)

        # Plot scaling
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("Vertical Range (±V)"))
        self.range_combo = QtWidgets.QComboBox()
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
        lp_row.addWidget(self.lowpass_spin)
        layout.addLayout(lp_row)

        layout.addStretch(1)

        # Downstream feature toggles
        toggle_row = QtWidgets.QHBoxLayout()
        self.listen_btn = QtWidgets.QPushButton("Listen")
        self.listen_btn.setCheckable(True)
        toggle_row.addWidget(self.listen_btn)
        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.setCheckable(True)
        toggle_row.addWidget(self.analyze_btn)
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
        self.analyze_btn.toggled.connect(self._on_widgets_changed)

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
        self.analyze_btn.setChecked(config.analyze_enabled)
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
        self._config.analyze_enabled = self.analyze_btn.isChecked()
        self._emit_config()

    def _emit_config(self) -> None:
        if self._block_updates:
            return
        self.configChanged.emit(replace(self._config))


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

        self._apply_palette()
        self._style_plot()

        self._build_ui()
        self._wire_placeholders()
        self._device_manager = DeviceManager(self)
        self._device_manager.devicesChanged.connect(self._on_devices_changed)
        self._device_manager.deviceConnected.connect(self._on_device_connected)
        self._device_manager.deviceDisconnected.connect(self._on_device_disconnected)
        self._device_manager.availableChannelsChanged.connect(self._on_available_channels)
        self._apply_device_state(False)
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

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(central)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)
        self.setCentralWidget(central)

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

        self.threshold_line = pg.InfiniteLine(angle=0, pen=pg.mkPen((178, 34, 34), width=2), movable=True)
        self.threshold_line.setVisible(False)
        self.plot_widget.addItem(self.threshold_line)

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
            status_row.addWidget(label)
            self._status_labels[key] = label
        status_row.addStretch(1)
        grid.addLayout(status_row, 1, 0, 1, 2)

        # Upper-right: stacked control boxes (Recording, Trigger, Channel Options).
        side_panel = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        self.record_group = QtWidgets.QGroupBox("Recording")
        record_layout = QtWidgets.QVBoxLayout(self.record_group)

        path_row = QtWidgets.QHBoxLayout()
        self.record_path_edit = QtWidgets.QLineEdit()
        self.record_path_edit.setPlaceholderText("Select output file...")
        path_row.addWidget(self.record_path_edit, 1)
        browse_btn = QtWidgets.QPushButton("Browse…")
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
        trigger_layout.addWidget(self.trigger_channel_combo, row, 1)
        row += 1

        trigger_layout.addWidget(self._label("Mode"), row, 0)
        mode_layout = QtWidgets.QVBoxLayout()
        self.trigger_mode_continuous = QtWidgets.QRadioButton("No Trigger (Continuous)")
        self.trigger_mode_single = QtWidgets.QRadioButton("Single")
        self.trigger_mode_single.setEnabled(False)
        self.trigger_mode_repeating = QtWidgets.QRadioButton("Continuous Trigger")
        self.trigger_mode_repeating.setEnabled(False)
        self.trigger_mode_continuous.setChecked(True)
        mode_layout.addWidget(self.trigger_mode_continuous)
        mode_layout.addWidget(self.trigger_mode_single)
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
        threshold_box.addWidget(self.threshold_spin)
        trigger_layout.addLayout(threshold_box, row, 0, 1, 2)
        row += 1

        pretrig_box = QtWidgets.QHBoxLayout()
        pretrig_box.addWidget(self._label("Pre-trigger (s)"))
        self.pretrigger_combo = QtWidgets.QComboBox()
        for value in (0.01, 0.05, 0.10):
            self.pretrigger_combo.addItem(f"{value:.2f}", value)
        self.pretrigger_combo.setCurrentIndex(1)
        pretrig_box.addWidget(self.pretrigger_combo)
        trigger_layout.addLayout(pretrig_box, row, 0, 1, 2)
        row += 1

        window_box = QtWidgets.QHBoxLayout()
        window_box.addWidget(self._label("Window Width (s)"))
        self.window_combo = QtWidgets.QComboBox()
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

        help_label = QtWidgets.QLabel(
            "Select a device from the menu above, then click Connect to get started."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: rgb(40,40,40); font-size: 11px;")
        device_layout.addWidget(help_label, 3, 0, 1, 2)

        bottom_layout.addWidget(self.device_group)

        self.channels_group = QtWidgets.QGroupBox("Channels")
        channels_layout = QtWidgets.QGridLayout(self.channels_group)
        channels_layout.setVerticalSpacing(6)
        channels_layout.setContentsMargins(8, 12, 8, 12)

        channels_layout.addWidget(self._label("Available"), 0, 0)
        channels_layout.addWidget(self._label("Active"), 0, 2)

        self.available_list = QtWidgets.QListWidget()
        self.available_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.available_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.available_list.setMinimumHeight(120)
        self.available_list.setMaximumHeight(140)
        channels_layout.addWidget(self.available_list, 1, 0)

        buttons_layout = QtWidgets.QVBoxLayout()
        self.add_channel_btn = QtWidgets.QPushButton("Add →")
        self.add_channel_btn.clicked.connect(self._on_add_channel)
        buttons_layout.addWidget(self.add_channel_btn)
        self.remove_channel_btn = QtWidgets.QPushButton("← Remove")
        self.remove_channel_btn.clicked.connect(self._on_remove_channel)
        buttons_layout.addWidget(self.remove_channel_btn)
        buttons_layout.addStretch(1)
        channels_layout.addLayout(buttons_layout, 1, 1)

        self.active_list = QtWidgets.QListWidget()
        self.active_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.active_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.active_list.setMinimumHeight(120)
        self.active_list.setMaximumHeight(140)
        channels_layout.addWidget(self.active_list, 1, 2)

        channels_layout.setRowStretch(1, 1)

        bottom_layout.addWidget(self.channels_group, stretch=2)

        bottom_layout.addWidget(self.device_group, 1)
        bottom_layout.addWidget(self.channels_group, 2)

        grid.addWidget(bottom_panel, 2, 0)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 0)
        grid.setRowStretch(2, 0)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 3)

    def _wire_placeholders(self) -> None:
        """Connect stub widgets to stub slots for future wiring."""
        self.trigger_channel_combo.currentTextChanged.connect(self._emit_trigger_config)
        self.trigger_mode_continuous.toggled.connect(self._emit_trigger_config)
        self.trigger_mode_single.toggled.connect(self._emit_trigger_config)
        self.trigger_mode_repeating.toggled.connect(self._emit_trigger_config)
        self.threshold_spin.valueChanged.connect(self._emit_trigger_config)
        self.window_combo.currentIndexChanged.connect(self._on_window_changed)
        self.threshold_line.sigPositionChanged.connect(self._on_threshold_line_changed)
        self.active_list.currentItemChanged.connect(self._on_active_channel_selected)

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
            signals = self._controller.dispatcher_signals()
            if signals is not None:
                try:
                    signals.tick.disconnect(self._on_dispatcher_tick)
                except (TypeError, RuntimeError):
                    pass

        self._controller = controller

        if controller is None:
            return

        self.startRecording.connect(controller.start_recording)
        self.stopRecording.connect(controller.stop_recording)

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

    def _label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        return label

    def _emit_trigger_config(self, *_) -> None:
        data = self.trigger_channel_combo.currentData()
        idx = int(data) if data is not None else -1
        config = {
            "channel_index": idx,
            "mode": self._current_trigger_mode(),
            "threshold": self.threshold_spin.value(),
            "hysteresis": 0.0,
            "pretrigger_frac": 0.0,
            "window_sec": float(self.window_combo.currentData() or 0.0),
        }
        self._update_trigger_visuals(config)

    def _current_trigger_mode(self) -> str:
        return "continuous"

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
        self.available_list.clear()
        self.active_list.clear()
        self._clear_channel_panels()
        self.set_trigger_channels([])
        self._update_channel_buttons()
        self._publish_active_channels()

    def _on_available_channels(self, channels: Sequence[object]) -> None:
        self.available_list.clear()
        self.active_list.clear()
        self._clear_channel_panels()
        for info in channels:
            name = getattr(info, "name", str(info))
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.UserRole, info)
            self.available_list.addItem(item)
        self.set_trigger_channels(channels)
        self._update_channel_buttons()
        self._publish_active_channels()

    def _on_add_channel(self) -> None:
        current = self.available_list.currentItem()
        if current is None:
            return
        info = current.data(QtCore.Qt.UserRole)
        item = QtWidgets.QListWidgetItem(current.text())
        item.setData(QtCore.Qt.UserRole, info)
        self.active_list.addItem(item)
        row = self.available_list.row(current)
        self.available_list.takeItem(row)
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
        self.available_list.addItem(item)
        row = self.active_list.row(current)
        self.active_list.takeItem(row)
        if self.active_list.count() > 0:
            self.active_list.setCurrentRow(0)
        else:
            self._show_channel_panel(None)
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
        self._sync_channel_panels(ids, names)
        self._reset_scope_for_channels(ids, names)
        self._sync_filter_settings()
        self._ensure_active_channel_focus()
        self._channel_last_samples = {cid: self._channel_last_samples[cid] for cid in ids if cid in self._channel_last_samples}
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
        span = max(config.range_v, 1e-6)
        y_clamped = max(min(y, span), -span)
        if abs(config.offset_v - y_clamped) < 1e-6:
            return
        config.offset_v = y_clamped
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
        span = max(config.range_v, 1e-6)
        config.offset_v = max(min(config.offset_v, span), -span)
        panel = self._channel_panels.get(channel_id)
        if panel is not None:
            panel.set_config(config)
        self._channel_configs[channel_id] = config
        self._update_channel_display(channel_id)
        self._refresh_channel_layout()
        self._sync_filter_settings()
        if self._active_channel_id == channel_id:
            self._update_axis_label()
        self._handle_listen_change(channel_id, config.listen_enabled)

    def _update_channel_display(self, channel_id: int) -> None:
        """Re-render a single channel's curve using the last raw samples and current offset/range."""
        if channel_id not in self._channel_configs:
            return
        config = self._channel_configs[channel_id]
        curve = self._curve_map.get(channel_id)
        raw = self._channel_last_samples.get(channel_id)
        if curve is None or raw is None or raw.size == 0 or self._last_times.size == 0:
            return
        display = raw + config.offset_v
        curve.setData(self._last_times, display, skipFiniteCheck=True)
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
                    player_queue.put(payload, timeout=0.05)
                except queue.Full:
                    pass
            except Exception:
                continue

    def _ensure_audio_player(self, *, show_error: bool = False) -> bool:
        """Create or reconfigure the AudioPlayer to match the current listen stream."""
        sample_rate = int(round(self._current_sample_rate))
        if sample_rate <= 0:
            return False
        with self._audio_lock:
            if self._audio_player is not None and abs(self._audio_input_samplerate - sample_rate) < 1e-6:
                return True
            player_to_stop = self._audio_player
            self._audio_player = None
            self._audio_player_queue = None
            self._audio_input_samplerate = 0.0
        if player_to_stop is not None:
            try:
                player_to_stop.stop()
                player_to_stop.join(timeout=1.0)
            except Exception:
                pass
        queue_obj: "queue.Queue" = queue.Queue(maxsize=32)
        config = AudioConfig(out_samplerate=44_100, out_channels=1, gain=0.7, blocksize=512, ring_seconds=0.5)
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
        self._audio_player.start()
        return True

    def _stop_audio_player(self) -> None:
        with self._audio_lock:
            player = self._audio_player
            self._audio_player = None
            self._audio_player_queue = None
            self._audio_input_samplerate = 0.0
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

    def _update_channel_buttons(self) -> None:
        connected = self._device_connected
        self.add_channel_btn.setEnabled(connected and self.available_list.count() > 0)
        self.remove_channel_btn.setEnabled(connected and self.active_list.count() > 0)

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
        self.available_list.setEnabled(enabled and self._device_connected)
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
        self.available_list.setEnabled(connected)
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

    def _on_threshold_line_changed(self) -> None:
        value = float(self.threshold_line.value())
        if abs(self.threshold_spin.value() - value) > 1e-6:
            self.threshold_spin.blockSignals(True)
            self.threshold_spin.setValue(value)
            self.threshold_spin.blockSignals(False)
        self._emit_trigger_config()

    def _update_trigger_visuals(self, config: dict) -> None:
        _ = config
        self.threshold_line.setVisible(False)
        self.pretrigger_line.setVisible(False)

    def _update_status(self, viz_depth: int) -> None:
        controller = self._controller
        if controller is None:
            stats = {}
            queue_depths: Dict[str, tuple[int, int]] = {}
        else:
            stats = controller.dispatcher_stats()
            queue_depths = controller.queue_depths()

        sr = getattr(self, "_current_sample_rate", 0.0)

        drops = stats.get("dropped", {}) if isinstance(stats, dict) else {}
        evicted = stats.get("evicted", {}) if isinstance(stats, dict) else {}

        self._status_labels["sr"].setText(f"SR: {sr:,.0f} Hz")
        self._status_labels["chunk"].setText(f"Chunks/s: {self._chunk_rate:5.1f}")

        viz_size, viz_max = queue_depths.get("visualization", (viz_depth, 0))
        analysis_size, analysis_max = queue_depths.get("analysis", (0, 0))
        audio_size, audio_max = queue_depths.get("audio", (0, 0))
        viz_max_text = "∞" if viz_max == 0 else str(viz_max)
        analysis_max_text = "∞" if analysis_max == 0 else str(analysis_max)
        audio_max_text = "∞" if audio_max == 0 else str(audio_max)
        self._status_labels["queues"].setText(
            f"Queues V:{viz_size}/{viz_max_text} A:{analysis_size}/{analysis_max_text} Au:{audio_size}/{audio_max_text}"
        )

        viz_drops = drops.get("visualization", 0)
        log_drops = drops.get("logging", 0)
        viz_evicted = evicted.get("visualization", 0)
        self._status_labels["drops"].setText(f"Drops V:{viz_drops} L:{log_drops} Evict:{viz_evicted}")

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

        if channel_ids != self._channel_ids_current:
            self._ensure_curves_for_ids(channel_ids, channel_names)

        if not self._curves:
            self.plot_widget.getPlotItem().setXRange(0, max(window_sec, 0.001), padding=0)
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            self._chunk_rate = 0.0
            self._update_status(viz_depth=0)
            return

        if data.ndim != 2 or data.size == 0:
            for curve in self._curves:
                curve.clear()
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            self._chunk_rate = 0.0
            self._update_status(viz_depth=0)
            return

        active_samples: Dict[int, np.ndarray] = {}

        for idx, curve in enumerate(self._curves):
            if idx < data.shape[0] and idx < len(channel_ids):
                cid = channel_ids[idx]
                config = self._channel_configs.get(cid)
                if config is None:
                    curve.clear()
                    continue
                raw = np.asarray(data[idx], dtype=np.float32)
                display = raw + config.offset_v
                curve.setData(times_arr, display, skipFiniteCheck=True)
                active_samples[cid] = raw
            else:
                curve.clear()

        self._channel_last_samples = active_samples
        self._apply_active_channel_style()

        if window_sec > 0:
            self.plot_widget.getPlotItem().setXRange(0, window_sec, padding=0)
        self._update_plot_y_range()

        self._current_sample_rate = sample_rate
        self._current_window_sec = window_sec
        self._chunk_rate = sample_rate
        self._update_status(viz_depth=0)
        if self._listen_channel_id is not None:
            self._ensure_audio_player()

    @QtCore.Slot(bool)
    def on_record_toggled(self, enabled: bool) -> None:
        """Placeholder slot that will route a record toggle to the controller."""
        self.recordToggled.emit(enabled)
