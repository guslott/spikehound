from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from core import PipelineController
from core.runtime import SpikeHoundRuntime
from shared.app_settings import AppSettings
from core.conditioning import ChannelFilterSettings, FilterSettings
from shared.models import ChunkPointer, EndOfStream
from shared.ring_buffer import SharedRingBuffer
from .analysis_dock import AnalysisDock
from .settings_tab import SettingsTab
from .scope_widget import ScopeWidget, ChannelConfig as ScopeChannelConfig
from .device_control_widget import DeviceControlWidget


@dataclass
class ChannelConfig:
    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(0, 0, 139))
    display_enabled: bool = True
    vertical_span_v: float = 1.0
    screen_offset: float = 0.5
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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._title_label = QtWidgets.QLabel("")
        self._title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._title_label)

        # Visual identity controls
        color_row = QtWidgets.QHBoxLayout()
        color_row.setSpacing(4)
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
        range_row.setSpacing(4)
        range_row.addWidget(QtWidgets.QLabel("Vertical Range (±V)"))
        self.range_combo = QtWidgets.QComboBox()
        self.range_combo.setMaximumWidth(120)
        for value in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
            self.range_combo.addItem(f"{value:.1f}", value)
        range_row.addWidget(self.range_combo, 1)
        layout.addLayout(range_row)

        # Vertical offset adjustment (normalized 0-1)
        offset_row = QtWidgets.QHBoxLayout()
        offset_row.setSpacing(4)
        offset_row.addWidget(QtWidgets.QLabel("Vertical Offset"))
        self.offset_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.offset_slider.setRange(0, 100)
        self.offset_slider.setSingleStep(1)
        self.offset_slider.setPageStep(5)
        self.offset_slider.setValue(50)
        self.offset_slider.setFixedWidth(140)
        offset_row.addWidget(self.offset_slider, 1)
        layout.addLayout(offset_row)

        # Filtering controls
        filters_layout = QtWidgets.QVBoxLayout()
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setSpacing(0)
        layout.addLayout(filters_layout)

        notch_row = QtWidgets.QHBoxLayout()
        notch_row.setContentsMargins(0, 0, 0, 0)
        notch_row.setSpacing(4)
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
        filters_layout.addLayout(notch_row)

        hp_row = QtWidgets.QHBoxLayout()
        hp_row.setContentsMargins(0, 0, 0, 0)
        hp_row.setSpacing(4)
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
        filters_layout.addLayout(hp_row)

        lp_row = QtWidgets.QHBoxLayout()
        lp_row.setContentsMargins(0, 0, 0, 0)
        lp_row.setSpacing(4)
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
        filters_layout.addLayout(lp_row)

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
        self.offset_slider.valueChanged.connect(self._on_widgets_changed)
        self.listen_btn.toggled.connect(self._on_widgets_changed)
        self.display_check.toggled.connect(self._on_display_toggled)
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)

    def set_channel_name(self, name: str) -> None:
        self._config.channel_name = name
        self._title_label.setText(f"Channel: {name}")

    def set_config(self, config: ChannelConfig) -> None:
        # Backwards compatibility for older configs
        span = getattr(config, "vertical_span_v", getattr(config, "range_v", 1.0))
        offset = getattr(config, "screen_offset", getattr(config, "offset_v", 0.5))
        config.vertical_span_v = span
        config.screen_offset = offset
        self._config = replace(config)
        self._block_updates = True
        self._apply_color(config.color)
        idx = self.range_combo.findData(config.vertical_span_v)
        if idx >= 0:
            self.range_combo.setCurrentIndex(idx)
        else:
            self.range_combo.addItem(f"{config.vertical_span_v:.3f}", config.vertical_span_v)
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
        self.offset_slider.blockSignals(True)
        self.offset_slider.setValue(int(round(config.screen_offset * 100.0)))
        self.offset_slider.blockSignals(False)
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
        self._config.vertical_span_v = float(self.range_combo.currentData())
        slider_val = float(self.offset_slider.value())
        if abs(slider_val - 50.0) <= 5.0:
            slider_val = 50.0
            self.offset_slider.blockSignals(True)
            self.offset_slider.setValue(int(slider_val))
            self.offset_slider.blockSignals(False)
        self._config.screen_offset = slider_val / 100.0
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
        self._logger = logging.getLogger(__name__)
        self.runtime = SpikeHoundRuntime(
            app_settings_store=getattr(controller, "app_settings_store", None),
            logger=self._logger,
            pipeline=controller,
        )
        self._controller: Optional[PipelineController] = None
        self._app_settings_unsub: Optional[Callable[[], None]] = None
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
        self._analysis_sample_rate: float = 0.0
        self._current_window_sec: float = 1.0
        self._dispatcher_signals: Optional[QtCore.QObject] = None
        self._drag_channel_id: Optional[int] = None
        self._active_channel_id: Optional[int] = None
        # Track which channel is being monitored (for UI state only, AudioManager handles actual routing)
        self._listen_channel_id: Optional[int] = None
        self._listen_device_key: Optional[str] = None
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
        self._downsample_supported = None
        self._window_combo_user_set = False
        self._window_combo_suppress = False
        self._splash_pixmap: Optional[QtGui.QPixmap] = None
        self._splash_label: Optional[QtWidgets.QLabel] = None
        self._splash_aspect_ratio: float = 1.0

        self._settings_tab: Optional[SettingsTab] = None
        self._apply_palette()
        self._style_plot()

        scope_widget = self._create_scope_widget()
        central_placeholder = QtWidgets.QWidget(self)
        central_placeholder.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        )
        central_placeholder.setMinimumHeight(0)
        central_placeholder.setMaximumHeight(0)
        central_placeholder.setVisible(False)
        self._central_placeholder = central_placeholder
        self.setCentralWidget(central_placeholder)
        self._analysis_dock = AnalysisDock(parent=self, controller=self.runtime)
        self._analysis_dock.set_scope_widget(scope_widget, "Scope")
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self._analysis_dock)
        self._analysis_dock.settingsClosed.connect(self._on_settings_tab_closed)
        self._analysis_dock.select_scope()
        self._wire_placeholders()
        self._update_trigger_controls()
        self.runtime.device_manager.devicesChanged.connect(self._on_devices_changed)
        self.runtime.device_manager.deviceConnected.connect(self._on_device_connected)
        self.runtime.device_manager.deviceDisconnected.connect(self._on_device_disconnected)
        self.runtime.device_manager.availableChannelsChanged.connect(self._on_available_channels)
        self._apply_device_state(False)
        self.runtime.device_manager.refresh_devices()
        self.attach_controller(controller)
        app_settings = controller.app_settings if controller is not None else None
        if app_settings is not None:
            self.set_plot_refresh_hz(float(app_settings.plot_refresh_hz))
            if not self._window_combo_user_set:
                self._set_window_combo_value(float(app_settings.default_window_sec))
            self._apply_listen_output_preference(app_settings.listen_output_key)
        # Standard close shortcut (Cmd+W on macOS, Ctrl+W on Windows/Linux)
        self._close_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Close, self)
        self._close_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self._close_shortcut.activated.connect(self.close)
        self._bind_app_settings_store()
        self._emit_trigger_config()
        QtCore.QTimer.singleShot(0, self._update_splash_pixmap)
        QtCore.QTimer.singleShot(0, self._try_load_default_config)

    def _reset_color_cycle(self) -> None:
        """Reset channel color selection to the initial palette order."""
        self._next_color_index = 0

        # Global shortcuts for quitting/closing
        quit_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Quit), self)
        quit_shortcut.activated.connect(self._quit_application)

    @property
    def chunk_rate(self) -> float:
        return float(getattr(self, "_chunk_rate", 0.0))

    @property
    def plot_refresh_hz(self) -> float:
        return float(getattr(self, "_plot_refresh_hz", 0.0))

    def set_plot_refresh_hz(self, hz: float) -> None:
        hz = max(1.0, float(hz))
        self._plot_refresh_hz = hz
        self._plot_interval = 1.0 / hz
        try:
            self.runtime.update_metrics(plot_refresh_hz=hz)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Runtime delegation
    # ------------------------------------------------------------------

    def open_device(self, driver: object, sample_rate: float, channels: Sequence[object]) -> None:
        """Delegate device attachment to the runtime/pipeline controller."""
        try:
            self.runtime.open_device(driver, sample_rate, channels)
        except Exception:
            return
        self._bind_dispatcher_signals()
        self._bind_app_settings_store()

    def configure_acquisition(
        self,
        *,
        sample_rate: Optional[int] = None,
        channels: Optional[list[int]] = None,
        chunk_size: Optional[int] = None,
        filter_settings: Optional[FilterSettings] = None,
        trigger_cfg: Optional[dict] = None,
    ) -> None:
        cfg_trigger = trigger_cfg
        try:
            self.runtime.configure_acquisition(
                sample_rate=sample_rate,
                channels=channels,
                chunk_size=chunk_size,
                filter_settings=filter_settings,
                trigger_cfg=cfg_trigger,
            )
        except Exception:
            return

    def start_acquisition(self) -> None:
        """Start streaming via the runtime."""
        try:
            self.runtime.start_acquisition()
        except Exception:
            return

    def stop_acquisition(self) -> None:
        """Stop streaming via the runtime."""
        try:
            self.runtime.stop_acquisition()
        except Exception:
            return

    def set_default_window_sec(self, value: float) -> None:
        value = max(0.05, float(value))
        if self._device_connected:
            return
        self._set_window_combo_value(value)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _create_scope_widget(self) -> QtWidgets.QWidget:
        scope = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(scope)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)

        # Upper-left: ScopeWidget for multi-channel visualization
        self.scope = ScopeWidget(self)
        grid.addWidget(self.scope, 0, 0)
        
        # Keep references to threshold and pretrigger lines for compatibility
        self.threshold_line = self.scope.threshold_line
        self.pretrigger_line = self.scope.pretrigger_line
        self.plot_widget = self.scope.plot_widget  # For backwards compatibility
        
        # Connect ScopeWidget signals
        self.scope.channelClicked.connect(self._on_scope_channel_clicked)
        self.scope.channelDragged.connect(self._on_scope_channel_dragged)
        self.scope.channelDragFinished.connect(self._on_scope_drag_finished)
        self.scope.thresholdChanged.connect(self._on_scope_threshold_changed)

        self._status_labels = {}

        # Upper-right: stacked control boxes (Recording, Trigger, Channel Options).
        side_panel = QtWidgets.QWidget()
        side_panel.setMaximumWidth(320)
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        self._splash_label = QtWidgets.QLabel()
        self._splash_label.setObjectName("splashLabel")
        self._splash_label.setAlignment(QtCore.Qt.AlignCenter)
        self._splash_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._splash_label.setContentsMargins(0, 0, 0, 0)
        self._splash_label.setStyleSheet(
            "#splashLabel { border: 2px solid rgb(0,0,0); background-color: rgb(0,0,0); padding: 0px; margin: 0px; }"
        )
        self._splash_pixmap = self._load_splash_pixmap()
        if self._splash_pixmap is None:
            self._splash_label.setText("Manlius Pebble Hill School\\nCornell University")
            self._splash_label.setWordWrap(True)
            self._splash_label.setStyleSheet(
                "#splashLabel { border: 2px solid rgb(0,0,0); background-color: rgb(0,0,0); color: rgb(240,240,240); padding: 6px; font-weight: bold; }"
            )
        else:
            self._update_splash_pixmap()
        side_layout.addWidget(self._splash_label)
        side_layout.addSpacing(10)

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
        trigger_layout.setContentsMargins(8, 8, 8, 8)
        trigger_layout.setVerticalSpacing(4)
        trigger_layout.setHorizontalSpacing(6)

        row = 0
        trigger_layout.addWidget(self._label("Channel"), row, 0)
        self.trigger_channel_combo = QtWidgets.QComboBox()
        self.trigger_channel_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.trigger_channel_combo.setMaximumWidth(200)
        trigger_layout.addWidget(self.trigger_channel_combo, row, 1)
        row += 1

        trigger_layout.addWidget(self._label("Mode"), row, 0)
        mode_layout = QtWidgets.QVBoxLayout()
        mode_layout.setSpacing(2)
        self.trigger_mode_continuous = QtWidgets.QRadioButton("No Trigger (Stream)")
        self.trigger_mode_single = QtWidgets.QRadioButton("Single")
        self.trigger_mode_single.setEnabled(False)
        self.trigger_mode_repeating = QtWidgets.QRadioButton("Continuous Trigger")
        self.trigger_mode_repeating.setEnabled(False)
        self.trigger_mode_continuous.setChecked(True)
        mode_layout.addWidget(self.trigger_mode_continuous)
        single_row = QtWidgets.QHBoxLayout()
        single_row.setSpacing(3)
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
        threshold_box.setSpacing(4)
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
        pretrig_box.setSpacing(4)
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
        window_box.setSpacing(4)
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
        grid.addWidget(side_panel, 0, 1, 2, 1)

        # Bottom row (spanning full width): device / channel controls.
        self.device_control = DeviceControlWidget(self)
        
        # Keep references to individual widgets for backward compatibility
        self.device_group = self.device_control.device_group
        self.device_combo = self.device_control.device_combo
        self.scan_hardware_btn = self.device_control.scan_hardware_btn
        self.device_toggle_btn = self.device_control.device_toggle_btn
        self.sample_rate_combo = self.device_control.sample_rate_combo
        self.settings_toggle_btn = self.device_control.settings_toggle_btn
        self.save_config_btn = self.device_control.save_config_btn
        self.load_config_btn = self.device_control.load_config_btn
        self.channels_group = self.device_control.channels_group
        self.active_list = self.device_control.active_list
        self.add_channel_btn = self.device_control.add_channel_btn
        self.remove_channel_btn = self.device_control.remove_channel_btn
        self.available_combo = self.device_control.available_combo
        
        # Connect DeviceControlWidget signals
        self.device_control.deviceSelected.connect(self._on_device_selected)
        self.device_control.deviceConnectRequested.connect(self._on_device_connect_requested)
        self.device_control.deviceDisconnectRequested.connect(self._on_device_disconnect_requested)
        self.device_control.deviceScanRequested.connect(self._on_scan_hardware)
        self.device_control.channelAddRequested.connect(self._on_channel_add_requested)
        self.device_control.channelRemoveRequested.connect(self._on_channel_remove_requested)
        self.device_control.activeChannelSelected.connect(self._on_active_list_index_changed)
        self.device_control.saveConfigRequested.connect(self._on_save_scope_config)
        self.device_control.loadConfigRequested.connect(self._on_load_scope_config)
        self.device_control.settingsToggled.connect(self._toggle_settings_tab)

        grid.addWidget(self.device_control, 1, 0)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 0)
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
        # NOTE: threshold_line signal now handled by ScopeWidget.thresholdChanged
        self.active_list.currentItemChanged.connect(self._on_active_channel_selected)
        self.trigger_single_button.clicked.connect(self._on_trigger_single_clicked)

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

        if hasattr(self, "runtime") and self.runtime is not None:
            try:
                self.runtime.set_pipeline(controller)
            except Exception:
                pass

        self._controller = controller
        if hasattr(self, "_analysis_dock") and self._analysis_dock is not None:
            self._analysis_dock._controller = self.runtime

        if controller is None:
            return

        self.startRecording.connect(controller.start_recording)
        self.stopRecording.connect(controller.stop_recording)
        self.triggerConfigChanged.connect(controller.update_trigger_config)

        self._bind_dispatcher_signals()
        self._update_status(0)
        self._bind_app_settings_store()

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

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_splash_pixmap()

    def _load_splash_pixmap(self) -> Optional[QtGui.QPixmap]:
        splash_path = Path(__file__).resolve().parent.parent / "media" / "mph_cornell_splash.png"
        if not splash_path.exists():
            return None
        pixmap = QtGui.QPixmap(str(splash_path))
        if pixmap.isNull():
            return None
        if pixmap.height() > 0:
            self._splash_aspect_ratio = pixmap.width() / float(pixmap.height())
        return pixmap

    def _update_splash_pixmap(self) -> None:
        if self._splash_label is None or self._splash_pixmap is None or self._splash_pixmap.isNull():
            return

        label_rect = self._splash_label.contentsRect()
        available_width = label_rect.width()
        if available_width <= 0:
            available_width = self._splash_label.width()
        if available_width <= 0 and self._splash_label.parentWidget() is not None:
            available_width = self._splash_label.parentWidget().width()
        if available_width <= 0:
            available_width = 200

        border_px = 4  # 2 px on each side defined in stylesheet
        available_width = int(max(50, available_width - border_px))

        aspect = self._splash_aspect_ratio if self._splash_aspect_ratio > 0 else self._splash_pixmap.width() / max(1, self._splash_pixmap.height())
        target_height = max(1, int(round(available_width / aspect)))

        scaled = self._splash_pixmap.scaled(
            available_width,
            target_height,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        if scaled.isNull():
            return

        total_height = scaled.height() + border_px
        self._splash_label.setMinimumHeight(total_height)
        self._splash_label.setMaximumHeight(total_height)
        self._splash_label.setPixmap(scaled)
        self._splash_label.updateGeometry()

    def _label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        return label

    def _color_to_tuple(self, color: QtGui.QColor) -> tuple[int, int, int, int]:
        if not isinstance(color, QtGui.QColor):
            return (0, 0, 0, 255)
        return (color.red(), color.green(), color.blue(), color.alpha())

    def _color_from_tuple(self, data: Sequence[int]) -> QtGui.QColor:
        try:
            r, g, b, a = (int(x) for x in data)
            return QtGui.QColor(r, g, b, a)
        except Exception:
            return QtGui.QColor(0, 0, 139)

    def _channel_config_to_dict(self, config: ChannelConfig) -> dict:
        return {
            "color": self._color_to_tuple(config.color),
            "display_enabled": bool(config.display_enabled),
            "vertical_span_v": float(config.vertical_span_v),
            "screen_offset": float(config.screen_offset),
            "notch_enabled": bool(config.notch_enabled),
            "notch_freq": float(config.notch_freq),
            "highpass_enabled": bool(config.highpass_enabled),
            "highpass_freq": float(config.highpass_freq),
            "lowpass_enabled": bool(config.lowpass_enabled),
            "lowpass_freq": float(config.lowpass_freq),
            "listen_enabled": bool(config.listen_enabled),
            "analyze_enabled": bool(config.analyze_enabled),
            "channel_name": config.channel_name,
        }

    def _channel_config_from_dict(self, payload: dict, *, fallback_name: str = "") -> ChannelConfig:
        cfg = ChannelConfig()
        try:
            cfg.color = self._color_from_tuple(payload.get("color", (0, 0, 139, 255)))
            cfg.display_enabled = bool(payload.get("display_enabled", True))
            cfg.vertical_span_v = float(payload.get("vertical_span_v", payload.get("range_v", 1.0)))
            cfg.screen_offset = float(payload.get("screen_offset", payload.get("offset_v", 0.5)))
            cfg.notch_enabled = bool(payload.get("notch_enabled", False))
            cfg.notch_freq = float(payload.get("notch_freq", 60.0))
            cfg.highpass_enabled = bool(payload.get("highpass_enabled", False))
            cfg.highpass_freq = float(payload.get("highpass_freq", 10.0))
            cfg.lowpass_enabled = bool(payload.get("lowpass_enabled", False))
            cfg.lowpass_freq = float(payload.get("lowpass_freq", 1_000.0))
            cfg.listen_enabled = bool(payload.get("listen_enabled", False))
            cfg.analyze_enabled = bool(payload.get("analyze_enabled", False))
            cfg.channel_name = str(payload.get("channel_name") or fallback_name or "")
        except Exception:
            cfg.channel_name = fallback_name
        return cfg

    def _collect_scope_config(self) -> Optional[dict]:
        device_key = self.device_combo.currentData()
        if device_key is None:
            QtWidgets.QMessageBox.information(self, "Save Config", "Select a device before saving a scope config.")
            return None
        window_value = float(self.window_combo.currentData() or self._current_window_sec or 0.0)
        payload = {
            "version": 1,
            "device_key": device_key,
            "sample_rate": float(self._current_sample_rate_value()),
            "window_sec": float(window_value),
            "channels": [],
        }
        channel_names = {cid: name for cid, name in zip(self._channel_ids_current, self._channel_names)}
        for cid in self._channel_ids_current:
            cfg = self._channel_configs.get(cid)
            if cfg is None:
                continue
            payload["channels"].append(
                {
                    "id": cid,
                    "name": channel_names.get(cid) or cfg.channel_name or f"Channel {cid}",
                    "config": self._channel_config_to_dict(cfg),
                }
            )
        return payload

    def _find_available_index_by_id(self, channel_id: int) -> int:
        for idx in range(self.available_combo.count()):
            info = self.available_combo.itemData(idx)
            if getattr(info, "id", None) == channel_id:
                return idx
        return -1

    def _apply_scope_config_data(self, data: dict, source: str = "", *, show_dialogs: bool = True) -> None:
        def _info(title: str, message: str) -> None:
            if show_dialogs:
                QtWidgets.QMessageBox.information(self, title, message)
            else:
                self.statusBar().showMessage(message, 5000)

        def _warning(title: str, message: str) -> None:
            if show_dialogs:
                QtWidgets.QMessageBox.warning(self, title, message)
            else:
                self.statusBar().showMessage(message, 7000)

        def _critical(title: str, message: str) -> None:
            if show_dialogs:
                QtWidgets.QMessageBox.critical(self, title, message)
            else:
                self.statusBar().showMessage(message, 8000)

        version = int(data.get("version", 1) or 1)
        if version != 1:
            _warning("Load Config", f"Unsupported config version: {version}")
            return
        device_key = data.get("device_key")
        sample_rate = float(data.get("sample_rate", self._current_sample_rate_value()))
        window_sec = float(data.get("window_sec", self._current_window_sec))
        channels_payload = data.get("channels") or []

        if device_key is not None:
            idx = self.device_combo.findData(device_key)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
            else:
                _warning(
                    "Load Config",
                    f"Device '{device_key}' is not available; cannot load configuration{f' from {source}' if source else ''}.",
                )
                return

        self._set_sample_rate_value(sample_rate)
        self._set_window_combo_value(window_sec)

        if device_key is None:
            _info("Load Config", "No device specified in the configuration.")
            return

        try:
            self.runtime.connect_device(device_key, sample_rate=self._current_sample_rate_value())
        except Exception as exc:
            _critical("Load Config", f"Failed to connect to device '{device_key}': {exc}")
            return

        # Refresh channel lists with the newly connected device
        try:
            available_channels = self.runtime.device_manager.get_available_channels()
            self._on_available_channels(available_channels)
        except Exception:
            pass

        missing_channels: list[int] = []
        self.active_list.blockSignals(True)
        self.available_combo.blockSignals(True)
        self.active_list.clear()
        self._channel_configs.clear()
        try:
            for entry in channels_payload:
                cid = entry.get("id")
                if cid is None:
                    continue
                idx = self._find_available_index_by_id(int(cid))
                if idx < 0:
                    missing_channels.append(int(cid))
                    continue
                info = self.available_combo.itemData(idx)
                name = entry.get("name") or self.available_combo.itemText(idx)
                item = QtWidgets.QListWidgetItem(name)
                item.setData(QtCore.Qt.UserRole, info)
                self.active_list.addItem(item)
                self.available_combo.removeItem(idx)
                cfg = self._channel_config_from_dict(entry.get("config") or {}, fallback_name=name)
                cfg.channel_name = name
                self._channel_configs[int(cid)] = cfg
            if self.active_list.count():
                self.active_list.setCurrentRow(0)
        finally:
            self.active_list.blockSignals(False)
            self.available_combo.blockSignals(False)

        self._publish_active_channels()
        if missing_channels:
            missing_str = ", ".join(str(cid) for cid in missing_channels)
            _info("Load Config", f"Loaded with missing channels: {missing_str}")
        else:
            msg = f"Scope configuration loaded{f' from {source}' if source else ''}."
            self.statusBar().showMessage(msg, 5000)

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

    def _on_save_scope_config(self) -> None:
        payload = self._collect_scope_config()
        if payload is None:
            return
        default_path = Path.home() / "spikehound_scope.json"
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Scope Configuration",
            str(default_path),
            "JSON Files (*.json)",
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        try:
            path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Save Config",
                f"Failed to save configuration: {exc}",
            )
            return
        self.statusBar().showMessage(f"Saved scope configuration to {path}", 5000)

    def _on_load_scope_config(self) -> None:
        start_dir = str(Path.home())
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Scope Configuration",
            start_dir,
            "JSON Files (*.json)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Config",
                f"Failed to read configuration: {exc}",
            )
            return
        if not isinstance(data, dict):
            QtWidgets.QMessageBox.critical(self, "Load Config", "Configuration file is not valid JSON.")
            return
        self._apply_scope_config_data(data, source=str(path))

    def _try_load_default_config(self) -> None:
        default_path = Path.cwd() / "default_config.json"
        if not default_path.is_file():
            return
        try:
            data = json.loads(default_path.read_text())
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to read default_config.json: {exc}", 7000)
            return
        if not isinstance(data, dict):
            self.statusBar().showMessage("default_config.json is not valid JSON.", 7000)
            return
        self._apply_scope_config_data(data, source=str(default_path), show_dialogs=False)

    def _on_device_button_clicked(self) -> None:
        dm = getattr(self.runtime, "device_manager", None)
        if dm is None:
            return
        if self._device_connected:
            dm.disconnect_device()
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
            self.runtime.connect_device(key, sample_rate=self._current_sample_rate_value())
        except Exception as exc:  # pragma: no cover - GUI feedback only
            QtWidgets.QMessageBox.critical(self, "Device", f"Failed to connect: {exc}")
            self.device_toggle_btn.blockSignals(True)
            self.device_toggle_btn.setChecked(False)
            self.device_toggle_btn.blockSignals(False)

    # DeviceControlWidget signal handlers
    def _on_device_connect_requested(self, device_key: str, sample_rate: float) -> None:
        """Handle connection request from DeviceControlWidget."""
        if self._device_connected:
            return
        
        # Actually connect the device using the runtime
        try:
            self.runtime.connect_device(device_key, sample_rate=sample_rate)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Device", f"Failed to connect: {exc}")
            self.device_control.set_connected(False)

    def _on_device_disconnect_requested(self) -> None:
        """Handle disconnection request from DeviceControlWidget."""
        if self._device_connected:
            dm = getattr(self.runtime, "device_manager", None)
            if dm is not None:
                dm.disconnect_device()

    def _on_channel_add_requested(self, channel_id: int) -> None:
        """Handle add channel request from DeviceControlWidget."""
        self._on_add_channel()

    def _on_channel_remove_requested(self, channel_id: int) -> None:
        """Handle remove channel request from DeviceControlWidget."""
        self._on_remove_channel()

    def _on_active_list_index_changed(self, index: int) -> None:
        """Handle active channel selection change by index."""
        if 0 <= index < self.active_list.count():
            item = self.active_list.item(index)
            if item is not None:
                self._on_active_channel_selected(item, None)


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
            dm = getattr(self.runtime, "device_manager", None)
            active_key = dm.active_key() if dm is not None else None
            if active_key is not None:
                idx = self.device_combo.findData(active_key)
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)
        self._on_device_selected()
        self._apply_device_state(self._device_connected and bool(entries))
        self._update_channel_buttons()

    def _on_device_selected(self) -> None:
        key = self.device_combo.currentData()
        entry = self._device_map.get(key) if key else None
        self._populate_sample_rate_options(entry)
        self._update_sample_rate_enabled()

    def _populate_sample_rate_options(self, entry: Optional[dict]) -> None:
        self.sample_rate_combo.blockSignals(True)
        self.sample_rate_combo.clear()
        rates = []
        if entry is not None:
            caps = entry.get("capabilities")
            if hasattr(caps, "sample_rates"):
                rates = getattr(caps, "sample_rates") or []
            elif isinstance(caps, dict):
                rates = caps.get("sample_rates") or []
        for rate in rates:
            self.sample_rate_combo.addItem(f"{int(rate):,}", float(rate))
        if self.sample_rate_combo.count():
            self.sample_rate_combo.setCurrentIndex(0)
        self.sample_rate_combo.setEnabled(bool(rates) and (not self._device_connected or self.active_list.count() == 0))
        self.sample_rate_combo.blockSignals(False)

    def _set_sample_rate_value(self, sample_rate: float) -> None:
        if self.sample_rate_combo.count() == 0:
            return
        idx = self.sample_rate_combo.findData(float(sample_rate))
        if idx < 0:
            idx = 0
        self.sample_rate_combo.setCurrentIndex(idx)

    def _current_sample_rate_value(self) -> float:
        data = self.sample_rate_combo.currentData()
        try:
            value = float(data)
        except Exception:
            value = 0.0
            if self.sample_rate_combo.count() > 0:
                d = self.sample_rate_combo.itemData(0)
                try:
                    value = float(d)
                except Exception:
                    value = 0.0
        # If still zero, use a reasonable default rather than failing
        if value == 0.0:
            value = 20000.0
        return value

    def _update_sample_rate_enabled(self) -> None:
        connected = self._device_connected
        has_active = self.active_list.count() > 0
        self.sample_rate_combo.setEnabled(self.sample_rate_combo.count() > 0 and ((not connected) or (connected and not has_active)))

    def _on_device_connected(self, key: str) -> None:
        self._device_connected = True
        self._reset_color_cycle()
        self._apply_device_state(True)
        idx = self.device_combo.findData(key)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self._bind_dispatcher_signals()
        self._bind_app_settings_store()
        self._update_channel_buttons()

    def _on_device_disconnected(self) -> None:
        self._device_connected = False
        self._reset_color_cycle()
        self._apply_device_state(False)
        self.stop_acquisition()
        self._drain_visualization_queue()
        self._clear_scope_display()
        # Don't clear sample_rate_combo - preserve rates for reconnection
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
        self._update_sample_rate_enabled()

    def _on_scan_hardware(self) -> None:
        dm = getattr(self.runtime, "device_manager", None)
        if dm is None:
            return
        try:
            dm.refresh_devices()
        except Exception:
            pass
        self._publish_active_channels()
        self._clear_scope_display()
        if hasattr(self, "_analysis_dock") and self._analysis_dock is not None:
            try:
                self._analysis_dock.shutdown()
            except Exception:
                pass
            self._analysis_dock.select_scope()

    def _on_available_channels(self, channels: Sequence[object]) -> None:
        # Update DeviceControlWidget with available channels
        self.device_control.set_available_channels(list(channels))
        
        # Clear active channels
        self.active_list.clear()
        self._clear_channel_panels()
        
        # Set trigger channels and update UI
        if self.available_combo.count():
            self.available_combo.setCurrentIndex(0)
        self.set_trigger_channels(channels)
        self._update_channel_buttons()
        self._publish_active_channels()
        self._update_sample_rate_enabled()

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
        # Activate and focus the newly added channel without extra signal chatter.
        self.active_list.blockSignals(True)
        self.active_list.setCurrentItem(item)
        self.active_list.blockSignals(False)
        self._update_channel_buttons()
        self._publish_active_channels()
        self._set_active_channel_focus(getattr(info, "id", None))
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
        if ids:
            self.configure_acquisition(channels=ids)
            # Ensure acquisition is started if we have active channels and a device
            if self._device_connected:
                self.start_acquisition()
        else:
            self.configure_acquisition(channels=[])
            # Stop acquisition if no channels are active
            self.stop_acquisition()
        self._update_sample_rate_enabled()
        dock = getattr(self, "_analysis_dock", None)
        if dock is not None:
            try:
                dock.update_active_channels(infos)
            except AttributeError:
                pass

    def _next_channel_color(self) -> QtGui.QColor:
        color = self._channel_color_cycle[self._next_color_index % len(self._channel_color_cycle)]
        self._next_color_index += 1
        return QtGui.QColor(color)

    def _initial_screen_offset(self) -> float:
        """Pick a starting offset; default all new channels to center."""
        return 0.5

    def _ensure_channel_config(self, channel_id: int, channel_name: str) -> ChannelConfig:
        config = self._channel_configs.get(channel_id)
        if config is None:
            config = ChannelConfig(
                color=self._next_channel_color(),
                channel_name=channel_name,
                screen_offset=self._initial_screen_offset(),
            )
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
            if channel_ids:
                self._set_active_channel_focus(channel_ids[0])

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
            center = config.screen_offset
            candidates.append((abs(float(y) - center), cid))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        distance, cid = candidates[0]
        if distance > 0.05:
            return None
        return cid

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
        self._show_channel_panel(cid)
        self._set_active_channel_focus(cid)

    def _on_plot_channel_dragged(self, y: float) -> None:
        if self._drag_channel_id is None:
            return
        config = self._channel_configs.get(self._drag_channel_id)
        if config is None:
            return
        y_clamped = max(0.0, min(1.0, float(y)))
        # Snap to center if within 5% of mid
        if abs(y_clamped - 0.5) <= 0.05:
            y_clamped = 0.5
        if abs(config.screen_offset - y_clamped) < 1e-6:
            return
        config.screen_offset = y_clamped
        panel = self._channel_panels.get(self._drag_channel_id)
        if panel is not None:
            panel.set_config(config)
        self._update_channel_display(self._drag_channel_id)
        self._update_plot_y_range()
        self._update_axis_label()

    def _on_plot_drag_finished(self) -> None:
        self._drag_channel_id = None

    def _on_channel_config_changed(self, channel_id: int, config: ChannelConfig) -> None:
        # Normalize legacy fields
        if not hasattr(config, "vertical_span_v"):
            try:
                config.vertical_span_v = float(getattr(config, "range_v", 1.0))
            except Exception:
                config.vertical_span_v = 1.0
        if not hasattr(config, "screen_offset"):
            try:
                config.screen_offset = float(getattr(config, "offset_v", 0.5))
            except Exception:
                config.screen_offset = 0.5
        existing = self._channel_configs.get(channel_id)
        if existing is not None:
            config.channel_name = existing.channel_name or config.channel_name
        display_changed = existing is not None and existing.display_enabled != config.display_enabled
        filters_changed = self._filters_changed(existing, config)
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
        if filters_changed:
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
        buf[:] = self._transform_to_screen(raw, config.vertical_span_v, config.screen_offset)
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

    def _open_analysis_for_channel(self, channel_id: int) -> None:
        dock = getattr(self, "_analysis_dock", None)
        if dock is None:
            return
        config = self._channel_configs.get(channel_id)
        if config is None:
            return
        channel_name = config.channel_name or f"Channel {channel_id}"
        sample_rate = self._current_sample_rate
        if sample_rate <= 0:
            sample_rate = float(self._current_sample_rate_value())
        if sample_rate <= 0:
            sample_rate = 1.0
        dock.open_analysis(channel_name, sample_rate)

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

        # Track for UI state
        self._listen_channel_id = channel_id

        #Use controller API to start audio monitoring
        if self._controller:
            self._controller.set_audio_monitoring(channel_id)
            # Update active channels so AudioManager knows about them
            if hasattr(self._controller, '_audio_manager') and self._controller._audio_manager:
                self._controller._audio_manager.update_active_channels(self._channel_ids_current)

    def _clear_listen_channel(self, channel_id: Optional[int] = None) -> None:
        """Disable audio monitoring, optionally for a specific channel."""
        target = channel_id if channel_id is not None else self._listen_channel_id
        if target is None:
            return
        
        # Update UI state
        cfg = self._channel_configs.get(target)
        if cfg is not None and cfg.listen_enabled:
            cfg.listen_enabled = False
            panel = self._channel_panels.get(target)
            if panel is not None:
                panel.set_config(cfg)
        
        # Clear UI state tracking
        if self._listen_channel_id == target:
            self._listen_channel_id = None
        
        # Use controller API to stop audio monitoring
        if self._controller:
            self._controller.set_audio_monitoring(None)

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
                try:
                    curve.setDownsampling(ds=True, auto=True, method="peak")
                except Exception:
                    pass
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
        try:
            self.runtime.update_metrics(chunk_rate=self._chunk_rate, sample_rate=self._current_sample_rate)
        except Exception:
            pass

    def _transform_to_screen(self, raw_data: np.ndarray, span_v: float, offset_pct: float) -> np.ndarray:
        span = max(float(span_v), 1e-9)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.asarray(raw_data, dtype=np.float32) / span + float(offset_pct)
        return np.nan_to_num(result, nan=offset_pct, posinf=offset_pct, neginf=offset_pct)

    def _update_plot_y_range(self) -> None:
        """Fix the normalized viewport to [0.0, 1.0]."""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setYRange(0.0, 1.0, padding=0.0)

    def _update_axis_label(self) -> None:
        axis = self.plot_widget.getPlotItem().getAxis("left")
        if self._active_channel_id is not None:
            config = self._channel_configs.get(self._active_channel_id)
            if config is not None:
                try:
                    axis.set_scaling(config.vertical_span_v, config.screen_offset)
                except AttributeError:
                    pass
                # Fixed ticks based on the channel span; values move with the screen offset.
                try:
                    span = max(float(config.vertical_span_v), 1e-9)
                    offset = float(config.screen_offset)
                    step = span / 10.0
                    vals: list[tuple[float, Optional[str]]] = []
                    start = int(np.floor(((0.0 - offset) * span) / step) - 2)
                    end = int(np.ceil(((1.0 - offset) * span) / step) + 2)
                    for n in range(start, end + 1):
                        v = n * step
                        pos = (v / span) + offset
                        if 0.0 <= pos <= 1.0:
                            vals.append((pos, f"{v:.3g}"))
                    axis.setTicks([vals])
                except Exception:
                    pass
                name = config.channel_name or f"Ch {self._active_channel_id}"
                axis_color = QtGui.QColor(config.color)
                rgb = axis_color.getRgb()[:3]
                pen = pg.mkPen(rgb, width=2)
                axis.setPen(pen)
                axis.setTextPen(pen)
                axis.setLabel(
                    text=f"{name} Amplitude (±{config.vertical_span_v:.3g} V)",
                    units="V",
                )
                return
        pen = pg.mkPen((0, 0, 139), width=1)
        axis.setPen(pen)
        axis.setTextPen(pen)
        axis.setLabel(text="Amplitude", units="V")

    def _filters_changed(self, previous: Optional[ChannelConfig], current: ChannelConfig) -> bool:
        if previous is None:
            return True

        def _diff(a: float, b: float) -> bool:
            return abs(float(a) - float(b)) > 1e-9

        return any(
            [
                previous.notch_enabled != current.notch_enabled,
                _diff(previous.notch_freq, current.notch_freq),
                previous.highpass_enabled != current.highpass_enabled,
                _diff(previous.highpass_freq, current.highpass_freq),
                previous.lowpass_enabled != current.lowpass_enabled,
                _diff(previous.lowpass_freq, current.lowpass_freq),
            ]
        )

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

    def _drain_visualization_queue(self) -> List[ChunkPointer]:
        queue_obj = getattr(self.runtime, "visualization_queue", None)
        if queue_obj is None:
            return []
        pointers: List[ChunkPointer] = []
        while True:
            try:
                item = queue_obj.get_nowait()
            except queue.Empty:
                break
            if item is EndOfStream:
                continue
            if isinstance(item, ChunkPointer):
                pointers.append(item)
            try:
                queue_obj.task_done()
            except Exception:
                pass
        return pointers

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
        self._update_sample_rate_enabled()
        self.device_toggle_btn.blockSignals(True)
        self.device_toggle_btn.setChecked(connected)
        self.device_toggle_btn.setText("Disconnect" if connected else "Connect")
        self.device_toggle_btn.setEnabled(connected or has_connectable)
        self.device_toggle_btn.blockSignals(False)
        self.save_config_btn.setEnabled(has_entries)
        self.load_config_btn.setEnabled(True)
        self.available_combo.setEnabled(connected)
        self.active_list.setEnabled(connected)
        self._update_channel_buttons()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        dm = getattr(self.runtime, "device_manager", None)
        if dm is not None:
            try:
                dm.disconnect_device()
            except Exception:
                pass
        self._clear_listen_channel()
        if hasattr(self, "_analysis_dock") and self._analysis_dock is not None:
            self._analysis_dock.shutdown()
        super().closeEvent(event)

    def _quit_application(self) -> None:
        QtWidgets.QApplication.instance().exit()

    def _set_window_combo_value(self, value: float) -> None:
        if not hasattr(self, "window_combo"):
            return
        target = max(float(value), 0.05)
        idx = -1
        for i in range(self.window_combo.count()):
            data = self.window_combo.itemData(i)
            if data is not None and abs(float(data) - target) < 1e-6:
                idx = i
                break
        if idx < 0:
            self.window_combo.addItem(f"{target:.2f}", target)
            idx = self.window_combo.count() - 1
        self._window_combo_suppress = True
        self.window_combo.setCurrentIndex(idx)
        self._window_combo_suppress = False
        self._apply_window_value(target)

    def _apply_window_value(self, value: float) -> None:
        self._current_window_sec = max(float(value), 1e-3)
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, self._current_window_sec, padding=0.0)
        if self._controller is not None:
            self._controller.update_window_span(self._current_window_sec)
        self._update_status(viz_depth=0)
        if self._trigger_last_sample_rate > 0:
            self._update_trigger_sample_parameters(self._trigger_last_sample_rate)

    def _on_window_changed(self) -> None:
        value = float(self.window_combo.currentData() or 0.0)
        if self._window_combo_suppress:
            self._apply_window_value(value)
            return
        self._window_combo_user_set = True
        self._apply_window_value(value)

    def _on_threshold_line_changed(self) -> None:
        y_norm = float(self.threshold_line.value())
        cfg = self._channel_configs.get(self._trigger_channel_id) or self._channel_configs.get(self._active_channel_id)
        span = cfg.vertical_span_v if cfg is not None else 1.0
        offset = cfg.screen_offset if cfg is not None else 0.0
        value = (y_norm - offset) * span
        if abs(self.threshold_spin.value() - value) > 1e-6:
            self.threshold_spin.blockSignals(True)
            self.threshold_spin.setValue(value)
            self.threshold_spin.blockSignals(False)
        self._emit_trigger_config()

    # ScopeWidget signal handlers
    def _on_scope_channel_clicked(self, channel_id: int) -> None:
        """Handle click on a channel trace in the scope."""
        self._select_channel_by_id(channel_id)

    def _on_scope_channel_dragged(self, channel_id: int, new_offset: float) -> None:
        """Handle dragging a channel trace to reposition it."""
        if channel_id not in self._channel_configs:
            return
        config = self._channel_configs[channel_id]
        config.screen_offset = new_offset
        # Update the scope widget's config
        scope_config = ScopeChannelConfig(
            color=config.color,
            display_enabled=config.display_enabled,
            vertical_span_v=config.vertical_span_v,
            screen_offset=config.screen_offset,
            channel_name=config.channel_name
        )
        self.scope.set_channel_config(channel_id, scope_config)
        # Update the channel panel if exists
        panel = self._channel_panels.get(channel_id)
        if panel is not None:
            panel.set_config(config)

    def _on_scope_drag_finished(self) -> None:
        """Handle end of channel drag operation."""
        pass  # Currently no additional action needed

    def _on_scope_threshold_changed(self, value: float) -> None:
        """Handle threshold line moved by user."""
        # value is in normalized 0-1 screen coordinates, convert to volts
        cfg = self._channel_configs.get(self._trigger_channel_id) or self._channel_configs.get(self._active_channel_id)
        if cfg is not None:
            span = cfg.vertical_span_v
            offset = cfg.screen_offset
            # Convert from normalized screen coords to volts
            voltage = (value - offset) * span
        else:
            # No channel config, can't convert properly
            voltage = 0.0
        
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setValue(voltage)
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
            # Use ScopeWidget API to hide threshold
            self.scope.set_threshold(0.0, visible=False)
            self.pretrigger_line.setVisible(False)
            return
        
        # Calculate threshold position in screen coordinates
        threshold_value = config.get("threshold", 0.0)
        cfg = self._channel_configs.get(self._trigger_channel_id) or self._channel_configs.get(self._active_channel_id)
        if cfg is not None:
            span = cfg.vertical_span_v
            offset = cfg.screen_offset
            # Transform threshold from volts to normalized 0-1 screen coords
            normalized_value = (threshold_value / span) + offset
        else:
            normalized_value = 0.5
        
        # Use ScopeWidget API to show and position threshold
        self.scope.set_threshold(normalized_value, visible=True)
        
        pen = pg.mkPen((0, 0, 0), width=3)
        self.threshold_line.setPen(pen)
        try:
            self.threshold_line.setZValue(100)
        except AttributeError:
            pass
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

        def _set_status_text(key: str, text: str) -> None:
            label = self._status_labels.get(key)
            if label is not None:
                label.setText(text)

    def health_snapshot(self) -> dict:
        return self.runtime.health_snapshot()

        _set_status_text(
            "sr",
            f"SR: {sr:,.0f} Hz\n"
            f"Drops V:{drops.get('visualization', 0)} "
            f"L:{drops.get('logging', 0)} Evict:{evicted.get('visualization', 0)}",
        )

        chunk_line = f"Chunks/s: {self._chunk_rate:5.1f}"
        if chunk_suffix:
            chunk_line = f"{chunk_line}\n{chunk_suffix}"
        else:
            chunk_line = f"{chunk_line}\n"
        _set_status_text("chunk", chunk_line)

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

        audio_text = _format_queue(queue_depths.get("audio"), "Au")
        logging_text = _format_queue(queue_depths.get("logging"), "L")

        _set_status_text("queues", f"Queues {viz_queue_text} {audio_text} {logging_text}\n{ring_text}")
        _set_status_text("drops", "")

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

    def _clear_scope_display(self) -> None:
        plot_item = self.plot_widget.getPlotItem()
        try:
            plot_item.clear()
        except Exception:
            pass
        for curve in list(self._curve_map.values()):
            try:
                plot_item.removeItem(curve)
            except Exception:
                pass
        self._curve_map.clear()
        for curve in self._curves:
            try:
                curve.clear()
            except Exception:
                pass
        self._curves = []
        self._channel_display_buffers.clear()
        self._channel_ids_current = []
        self._channel_names = []
        self._active_channel_id = None
        self._apply_active_channel_style()
        default_window = float(self.window_combo.currentData() or 1.0)
        self._current_window_sec = max(default_window, 1e-3)
        plot_item.setXRange(0.0, self._current_window_sec, padding=0.0)
        self._update_plot_y_range()
        self.threshold_line.setVisible(False)
        self.pretrigger_line.setVisible(False)
        self._current_sample_rate = 0.0
        self._chunk_rate = 0.0
        self._chunk_mean_samples = 0.0
        self._chunk_accum_count = 0
        self._chunk_accum_samples = 0
        self._chunk_last_rate_update = time.perf_counter()
        self._update_status(viz_depth=0)

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
                try:
                    curve.setDownsampling(ds=True, auto=True, method="peak")
                except Exception:
                    pass
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
                buf[:] = self._transform_to_screen(raw, config.vertical_span_v, config.screen_offset)
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
        try:
            self.runtime.update_metrics(sample_rate=sample_rate)
        except Exception:
            pass
        self._update_status(viz_depth=0)
        self._maybe_update_analysis_sample_rate(sample_rate)

    def _maybe_update_analysis_sample_rate(self, sample_rate: float) -> None:
        if sample_rate <= 0:
            return
        if abs(sample_rate - self._analysis_sample_rate) < 1e-3:
            return
        self._analysis_sample_rate = float(sample_rate)
        dock = getattr(self, "_analysis_dock", None)
        if isinstance(dock, AnalysisDock):
            dock.update_sample_rate(sample_rate)

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
        
        # Check for channel count mismatch
        if self._trigger_history:
            last_chunk = self._trigger_history[-1]
            if last_chunk.shape[1] != chunk_samples.shape[1]:
                print(f"Trigger history channel mismatch: had {last_chunk.shape[1]}, got {chunk_samples.shape[1]}. Resetting history.")
                self._trigger_history.clear()
                self._trigger_history_length = 0
                self._trigger_history_total = 0
                self._trigger_max_chunk = 0
                
        self._trigger_history.append(chunk_samples)
        self._trigger_history_length += chunk_samples.shape[0]
        self._trigger_history_total += chunk_samples.shape[0]
        self._trigger_max_chunk = max(self._trigger_max_chunk, chunk_samples.shape[0])
        # Keep 3x the trigger window to prevent evicting PSP tails before capture finalizes
        # OLD: max_keep = max(self._trigger_window_samples + self._trigger_max_chunk, ...)
        # This was too aggressive and evicted PSP tail chunks before finalization!
        max_keep = self._trigger_window_samples * 3
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
            # Waiting for more data...
            return
        if not self._trigger_history:
            return
            
        # Calculate absolute range we need
        earliest_abs = self._trigger_history_total - self._trigger_history_length
        start_abs = max(self._trigger_capture_start_abs, earliest_abs)
        end_abs = start_abs + self._trigger_window_samples
        
        # Efficiently collect only the relevant chunks
        relevant_chunks = []
        current_abs = earliest_abs
        
        for chunk in self._trigger_history:
            chunk_len = chunk.shape[0]
            chunk_end = current_abs + chunk_len
            
            # Check if this chunk overlaps with [start_abs, end_abs]
            if chunk_end > start_abs and current_abs < end_abs:
                relevant_chunks.append(chunk)
                
            current_abs += chunk_len
            
            # Optimization: Stop if we've passed the end of the requested window
            if current_abs >= end_abs:
                break
                
        if not relevant_chunks:
            return

        # Concatenate only the relevant chunks
        data = np.concatenate(relevant_chunks, axis=0)
        
        # Adjust indices relative to the concatenated 'data' buffer
        # The 'data' buffer starts at the 'current_abs' of the first chunk in relevant_chunks
        first_chunk_start_abs = start_abs  # Fallback
        
        # Re-calculate the absolute start of the first chunk we grabbed
        # We need to find the start_abs of relevant_chunks[0]
        # We can do this by backtracking or just being smarter above.
        # Let's just use the offset logic:
        
        # Find the absolute start of the first chunk in the list
        # We know 'earliest_abs' is the start of the *entire* history.
        # We need to find the offset of relevant_chunks[0] from earliest_abs.
        # Actually, let's just re-calculate it properly.
        
        # Simpler approach: Concatenate everything (it's safe now that we only grabbed relevant ones)
        # But we need to know where 'data' starts in absolute terms to slice it correctly.
        
        # Let's find the absolute start of relevant_chunks[0]
        scan_abs = earliest_abs
        data_start_abs = earliest_abs # Default
        found_start = False
        for chunk in self._trigger_history:
            if chunk is relevant_chunks[0]:
                data_start_abs = scan_abs
                found_start = True
                break
            scan_abs += chunk.shape[0]
            
        if not found_start:
            # Should never happen
            return
            
        # Now slice 'data' to get exactly [start_abs, end_abs]
        # relative_start = start_abs - data_start_abs
        start_idx = start_abs - data_start_abs
        end_idx = start_idx + self._trigger_window_samples
        
        # DIAGNOSTIC: Check for truncation
        if end_idx > data.shape[0]:
            missing_samples = end_idx - data.shape[0]
            print(f"!!! TRIGGER TRUNCATION: requested {end_idx} samples, only have {data.shape[0]}, missing {missing_samples} samples")
            print(f"!!! This explains PSP tail truncation in trigger mode!")
            end_idx = data.shape[0]
        snippet = data[start_idx:end_idx]
        if snippet.shape[0] < self._trigger_window_samples:
            pad = self._trigger_window_samples - snippet.shape[0]
            missing_ms = (pad / self._trigger_last_sample_rate) * 1000 if self._trigger_last_sample_rate > 0 else 0
            print(f"!!! TRIGGER PADDING: snippet has {snippet.shape[0]} samples, need {self._trigger_window_samples}, padding {pad} samples ({missing_ms:.1f}ms)")
            print(f"!!! This explains PSP truncation - trigger finalized before PSP tail arrived!")
            print(f"!!! History stats: total={self._trigger_history_total}, length={self._trigger_history_length}, data.shape={data.shape}")
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
                if config is None:
                    curve.setData(time_axis, data[:, idx], skipFiniteCheck=True)
                else:
                    transformed = self._transform_to_screen(data[:, idx], config.vertical_span_v, config.screen_offset)
                    curve.setData(time_axis, transformed, skipFiniteCheck=True)
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
            try:
                self.runtime.update_metrics(sample_rate=sample_rate)
            except Exception:
                pass
            self._render_trigger_display(channel_ids, window_sec)
            self._update_status(viz_depth=0)

            return

        if self._trigger_mode == "single":
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            try:
                self.runtime.update_metrics(sample_rate=sample_rate)
            except Exception:
                pass
            self._update_status(viz_depth=0)

            return

        if self._trigger_mode == "continuous":
            # Only update the plot when we have a trigger display; otherwise hold.
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            try:
                self.runtime.update_metrics(sample_rate=sample_rate)
            except Exception:
                pass
            self._update_status(viz_depth=0)
            return

        self._process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)


    @QtCore.Slot(object)
    def _on_dispatcher_tick(self, payload: object) -> None:
        # Extract pointers and metadata (payload may be dict or ChunkPointer/list).
        status: Dict[str, object] = {}
        channel_ids: List[int] = []
        channel_names: List[str] = []
        samples = None
        times = None
        pointers: List[ChunkPointer] = []

        if isinstance(payload, ChunkPointer):
            pointers = [payload]
        elif isinstance(payload, (list, tuple)) and payload and all(isinstance(p, ChunkPointer) for p in payload):
            pointers = list(payload)
        elif isinstance(payload, dict):
            status = payload.get("status", {}) or {}
            channel_ids = list(payload.get("channel_ids", []))
            channel_names = list(payload.get("channel_names", []))
            samples = payload.get("samples")
            times = payload.get("times")

        # Always drain the queue to keep it from growing.
        queue_pointers = self._drain_visualization_queue()
        has_samples = samples is not None and not (isinstance(samples, np.ndarray) and samples.size == 0)
        
        # Logic to decide source:
        # 1. If in Trigger Mode (single/continuous), we MUST use queue_pointers (chunks) to build history correctly.
        #    We must NEVER use the window payload (samples) because it's a snapshot, not a stream, and will cause
        #    overlaps, duplicates, and channel count mismatches (oscillation).
        # 2. If in Stream Mode, we prefer the payload (window) for smooth display, unless it's missing.
        use_pointers = False
        if self._trigger_mode != "stream":
            # STRICT: Only use pointers. If no pointers, we have no new stream data.
            # Do NOT fall back to samples.
            use_pointers = True
            # CRITICAL: Discard samples to prevent fallback in the 'else' block below
            samples = None
            times = None
        elif not has_samples and queue_pointers:
            use_pointers = True
            
        if use_pointers:
            pointers = queue_pointers

        viz_buffer = self._controller.viz_buffer() if self._controller else None
        sample_rate = float(status.get("sample_rate", self._current_sample_rate))
        window_sec = float(status.get("window_sec", self._current_window_sec))
        if sample_rate > 0:
            self._maybe_update_analysis_sample_rate(sample_rate)

        data: np.ndarray
        times_arr: np.ndarray
        if pointers and viz_buffer is not None:
            blocks: List[np.ndarray] = []
            for ptr in pointers:
                try:
                    block = np.asarray(viz_buffer.read(ptr.start_index, ptr.length), dtype=np.float32)
                except Exception as e:
                    print(f"!!! VIZ_BUFFER READ FAILED: start_index={ptr.start_index}, length={ptr.length}, error={e}")
                    print(f"!!! This explains PSP truncation - chunk was overwritten before GUI could read it!")
                    continue
                if block.size == 0:
                    continue
                blocks.append(block)
            if not blocks:
                return
            data = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
            if sample_rate > 0:
                times_arr = np.arange(data.shape[1], dtype=np.float32) / float(sample_rate)
                window_sec = max(window_sec, float(data.shape[1]) / float(sample_rate))
            else:
                times_arr = np.zeros(data.shape[1], dtype=np.float32)
            if not channel_ids:
                # When using pointers, channel_ids might not be in the payload.
                # Prioritize the cached user selection to preserve non-contiguous IDs.
                channel_ids = self._channel_ids_current if self._channel_ids_current else list(range(data.shape[0]))
            if not channel_names:
                channel_names = [str(cid) for cid in channel_ids]
        else:
            data = np.asarray(samples) if samples is not None else np.zeros((0, 0), dtype=np.float32)
            times_arr = np.asarray(times) if times is not None else np.zeros(0, dtype=np.float32)

        self._last_times = times_arr
        now = time.perf_counter()

        self._register_chunk(data)

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

    def _bind_app_settings_store(self) -> None:
        controller = self._controller
        if controller is None or not hasattr(controller, "app_settings_store"):
            if self._app_settings_unsub:
                self._app_settings_unsub()
                self._app_settings_unsub = None
            return
        store = controller.app_settings_store
        if self._app_settings_unsub:
            self._app_settings_unsub()
        def _apply(settings: AppSettings) -> None:
            try:
                self.set_plot_refresh_hz(float(settings.plot_refresh_hz))
            except Exception:
                pass
            if not self._device_connected and not self._window_combo_user_set:
                self._set_window_combo_value(float(settings.default_window_sec))
            self._apply_listen_output_preference(settings.listen_output_key)
            dm = getattr(self.runtime, "device_manager", None)
            if dm is not None:
                try:
                    dm.set_list_all_audio_devices(settings.list_all_audio_devices, refresh=True)
                except Exception:
                    pass
            if self._controller is not None:
                try:
                    self._controller.set_list_all_audio_devices(settings.list_all_audio_devices)
                except Exception:
                    pass
        self._app_settings_unsub = store.subscribe(_apply)

    def _apply_listen_output_preference(self, key: Optional[str]) -> None:
        prev = getattr(self, "_listen_device_key", None)
        self._listen_device_key = key
        if self._settings_tab is not None:
            self._settings_tab.set_listen_device(key)
        if prev == key:
            return
        if self._listen_channel_id is None:
            return
        self._stop_audio_player()
        current = self._listen_channel_id
        if current is None:
            return
        if not self._ensure_audio_player(show_error=False):
            self._clear_listen_channel(current)
        else:
            self._flush_audio_player_queue()

    def set_listen_output_device(self, device_key: Optional[str]) -> None:
        normalized = None if device_key in (None, "") else str(device_key)
        try:
            self.runtime.set_listen_output_device(normalized)
        except Exception:
            pass
    def _toggle_settings_tab(self, checked: bool) -> None:
        dock = getattr(self, "_analysis_dock", None)
        controller = self._controller
        if dock is None or controller is None:
            self.settings_toggle_btn.blockSignals(True)
            self.settings_toggle_btn.setChecked(False)
            self.settings_toggle_btn.blockSignals(False)
            return
        if checked:
            if self._settings_tab is None:
                self._settings_tab = SettingsTab(self.runtime, self)
                self._settings_tab.set_listen_device(self._listen_device_key)
            dock.open_settings(self._settings_tab)
        else:
            dock.close_settings()

    def _on_settings_tab_closed(self) -> None:
        if hasattr(self, "settings_toggle_btn"):
            self.settings_toggle_btn.blockSignals(True)
            self.settings_toggle_btn.setChecked(False)
            self.settings_toggle_btn.blockSignals(False)
