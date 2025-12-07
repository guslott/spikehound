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
from .channel_controls_widget import ChannelControlsWidget, ChannelDetailPanel
from .device_control_widget import DeviceControlWidget
from .types import ChannelConfig
from .trace_renderer import TraceRenderer
from .trigger_controller import TriggerController
from .plot_manager import PlotManager
from .device_manager import DeviceManager












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
        
        # Create runtime with dependency-injected DeviceManager (keeps core free of Qt imports)
        from core.device_registry import DeviceRegistry
        device_registry = DeviceRegistry()
        device_manager = DeviceManager(device_registry)
        self.runtime = SpikeHoundRuntime(
            app_settings_store=getattr(controller, "app_settings_store", None),
            logger=self._logger,
            pipeline=controller,
            device_registry=device_registry,
            device_manager=device_manager,
        )
        
        self._controller: Optional[PipelineController] = None
        self._app_settings_unsub: Optional[Callable[[], None]] = None
        self.setWindowTitle("SpikeHound - Manlius Pebble Hill School & Cornell University")
        self.resize(1100, 720)
        self.statusBar()

        # Persistent view/model state for the plotting surface and UI panels
        self._renderers: Dict[int, TraceRenderer] = {}
        self._channel_names: List[str] = []
        self._chunk_rate: float = 0.0
        self._device_map: Dict[str, dict] = {}
        self._device_connected = False
        self._active_channel_infos: List[object] = []
        self._channel_ids_current: List[int] = []
        # self._curve_map removed in favor of self._renderers
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
        # TriggerController manages all trigger state and detection logic
        self._trigger_controller = TriggerController(parent=self)
        self._plot_refresh_hz = 40.0
        self._plot_interval = 1.0 / self._plot_refresh_hz
        self._last_plot_refresh = 0.0
        # Actual measured plot refresh rate tracking
        self._actual_plot_refresh_hz = 0.0
        self._plot_refresh_count = 0
        self._plot_refresh_last_calc = time.perf_counter()
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

        # Recording timer state
        self._recording_start_time: Optional[float] = None
        self._recording_timer: Optional[QtCore.QTimer] = None

        self._settings_tab: Optional[SettingsTab] = None
        self._apply_palette()
        self._style_plot()

        scope_widget = self._create_scope_widget()
        
        # Create PlotManager to handle rendering and data processing
        self._plot_manager = PlotManager(
            plot_widget=self.plot_widget,
            trigger_controller=self._trigger_controller,
            parent=self,
        )
        self._plot_manager.sampleRateChanged.connect(self._maybe_update_analysis_sample_rate)
        
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
        
        # Create and add permanent Settings tab
        self._settings_tab = SettingsTab(self.runtime, self)
        self._settings_tab.saveConfigRequested.connect(self._on_save_scope_config)
        self._settings_tab.loadConfigRequested.connect(self._on_load_scope_config)
        self._analysis_dock.set_settings_widget(self._settings_tab, "Settings")
        
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self._analysis_dock)
        self._analysis_dock.select_scope()
        
        # Timer for updating file playback position
        self._playback_timer = QtCore.QTimer(self)
        self._playback_timer.setInterval(100)  # 10 Hz update rate
        self._playback_timer.timeout.connect(self._update_playback_position)
        
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

    # -------------------------------------------------------------------------
    # Trigger state properties (delegate to TriggerController)
    # -------------------------------------------------------------------------
    
    @property
    def _trigger_mode(self) -> str:
        return self._trigger_controller._mode
    
    @_trigger_mode.setter
    def _trigger_mode(self, value: str) -> None:
        self._trigger_controller._mode = value
    
    @property
    def _trigger_channel_id(self) -> Optional[int]:
        return self._trigger_controller._channel_id
    
    @_trigger_channel_id.setter
    def _trigger_channel_id(self, value: Optional[int]) -> None:
        self._trigger_controller._channel_id = value
    
    @property
    def _trigger_threshold(self) -> float:
        return self._trigger_controller._threshold
    
    @_trigger_threshold.setter
    def _trigger_threshold(self, value: float) -> None:
        self._trigger_controller._threshold = value
    
    @property
    def _trigger_pre_seconds(self) -> float:
        return self._trigger_controller._pre_seconds
    
    @_trigger_pre_seconds.setter
    def _trigger_pre_seconds(self, value: float) -> None:
        self._trigger_controller._pre_seconds = value
    
    @property
    def _trigger_pre_samples(self) -> int:
        return self._trigger_controller._pre_samples
    
    @_trigger_pre_samples.setter
    def _trigger_pre_samples(self, value: int) -> None:
        self._trigger_controller._pre_samples = value
    
    @property
    def _trigger_window_samples(self) -> int:
        return self._trigger_controller._window_samples
    
    @_trigger_window_samples.setter
    def _trigger_window_samples(self, value: int) -> None:
        self._trigger_controller._window_samples = value
    
    @property
    def _trigger_last_sample_rate(self) -> float:
        return self._trigger_controller._last_sample_rate
    
    @_trigger_last_sample_rate.setter
    def _trigger_last_sample_rate(self, value: float) -> None:
        self._trigger_controller._last_sample_rate = value
    
    @property
    def _trigger_history(self) -> deque:
        return self._trigger_controller._history
    
    @property
    def _trigger_history_length(self) -> int:
        return self._trigger_controller._history_length
    
    @_trigger_history_length.setter
    def _trigger_history_length(self, value: int) -> None:
        self._trigger_controller._history_length = value
    
    @property
    def _trigger_history_total(self) -> int:
        return self._trigger_controller._history_total
    
    @_trigger_history_total.setter
    def _trigger_history_total(self, value: int) -> None:
        self._trigger_controller._history_total = value
    
    @property
    def _trigger_max_chunk(self) -> int:
        return self._trigger_controller._max_chunk
    
    @_trigger_max_chunk.setter
    def _trigger_max_chunk(self, value: int) -> None:
        self._trigger_controller._max_chunk = value
    
    @property
    def _trigger_prev_value(self) -> float:
        return self._trigger_controller._prev_value
    
    @_trigger_prev_value.setter
    def _trigger_prev_value(self, value: float) -> None:
        self._trigger_controller._prev_value = value
    
    @property
    def _trigger_capture_start_abs(self) -> Optional[int]:
        return self._trigger_controller._capture_start_abs
    
    @_trigger_capture_start_abs.setter
    def _trigger_capture_start_abs(self, value: Optional[int]) -> None:
        self._trigger_controller._capture_start_abs = value
    
    @property
    def _trigger_capture_end_abs(self) -> Optional[int]:
        return self._trigger_controller._capture_end_abs
    
    @_trigger_capture_end_abs.setter
    def _trigger_capture_end_abs(self, value: Optional[int]) -> None:
        self._trigger_controller._capture_end_abs = value
    
    @property
    def _trigger_display(self) -> Optional[np.ndarray]:
        return self._trigger_controller._display
    
    @_trigger_display.setter
    def _trigger_display(self, value: Optional[np.ndarray]) -> None:
        self._trigger_controller._display = value
    
    @property
    def _trigger_display_times(self) -> Optional[np.ndarray]:
        return self._trigger_controller._display_times
    
    @_trigger_display_times.setter
    def _trigger_display_times(self, value: Optional[np.ndarray]) -> None:
        self._trigger_controller._display_times = value
    
    @property
    def _trigger_hold_until(self) -> float:
        return self._trigger_controller._hold_until
    
    @_trigger_hold_until.setter
    def _trigger_hold_until(self, value: float) -> None:
        self._trigger_controller._hold_until = value
    
    @property
    def _trigger_single_armed(self) -> bool:
        return self._trigger_controller._single_armed
    
    @_trigger_single_armed.setter
    def _trigger_single_armed(self, value: bool) -> None:
        self._trigger_controller._single_armed = value
    
    @property
    def _trigger_display_pre_samples(self) -> int:
        return self._trigger_controller._display_pre_samples
    
    @_trigger_display_pre_samples.setter
    def _trigger_display_pre_samples(self, value: int) -> None:
        self._trigger_controller._display_pre_samples = value

    def set_plot_refresh_hz(self, hz: float) -> None:
        hz = max(1.0, float(hz))
        self._plot_refresh_hz = hz
        self._plot_interval = 1.0 / hz
        try:
            self.runtime.update_metrics(plot_refresh_hz=hz)
        except Exception as exc:
            self._logger.debug("Failed to update plot refresh metrics: %s", exc)

    # ------------------------------------------------------------------
    # Runtime delegation
    # ------------------------------------------------------------------

    def open_device(self, driver: object, sample_rate: float, channels: Sequence[object]) -> None:
        """Delegate device attachment to the runtime/pipeline controller."""
        try:
            self.runtime.open_device(driver, sample_rate, channels)
        except Exception as exc:
            self._logger.warning("Failed to open device: %s", exc)
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
        except Exception as exc:
            self._logger.warning("Failed to configure acquisition: %s", exc)
            return

    def start_acquisition(self) -> None:
        """Start streaming via the runtime."""
        try:
            self.runtime.start_acquisition()
        except Exception as exc:
            self._logger.warning("Failed to start acquisition: %s", exc)
            return

    def stop_acquisition(self) -> None:
        """Stop streaming via the runtime."""
        try:
            self.runtime.stop_acquisition()
        except Exception as exc:
            self._logger.warning("Failed to stop acquisition: %s", exc)
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

        # Upper-left: ScopeWidget for multi-channel visualization (spans 2 columns)
        self.scope = ScopeWidget(self)
        grid.addWidget(self.scope, 0, 0, 1, 2)
        
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
        self.record_browse_btn = QtWidgets.QPushButton("Browse…")
        self.record_browse_btn.setFixedWidth(80)
        self.record_browse_btn.clicked.connect(self._on_browse_record_path)
        path_row.addWidget(self.record_browse_btn)
        record_layout.addLayout(path_row)

        self.record_autoinc = QtWidgets.QCheckBox("Auto-increment filename if exists")
        self.record_autoinc.setChecked(True)
        record_layout.addWidget(self.record_autoinc)

        self.record_toggle_btn = QtWidgets.QPushButton("Start Recording")
        self.record_toggle_btn.setCheckable(True)
        self._apply_record_button_style(False)
        self.record_toggle_btn.setEnabled(False)  # Disabled until filename is set
        self.record_toggle_btn.clicked.connect(self._toggle_recording)
        self.record_path_edit.textChanged.connect(self._update_record_button_enabled)
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
        
        # Group radio buttons
        self.trigger_button_group = QtWidgets.QButtonGroup(self)
        self.trigger_button_group.addButton(self.trigger_mode_continuous)
        self.trigger_button_group.addButton(self.trigger_mode_single)
        self.trigger_button_group.addButton(self.trigger_mode_repeating)
        
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
        default_index = 1 if self.window_combo.count() > 1 else self.window_combo.count() - 1
        self.window_combo.setCurrentIndex(max(0, default_index))
        window_box.addWidget(self.window_combo)
        trigger_layout.addLayout(window_box, row, 0, 1, 2)

        side_layout.addWidget(self.trigger_group)

        # Bottom row (spanning full width): device / channel controls.
        self.device_control = DeviceControlWidget(self)
        self.channel_controls = ChannelControlsWidget(self)
        
        # Keep references to individual widgets for backward compatibility
        self.device_group = self.device_control.device_group
        self.device_combo = self.device_control.device_combo
        # self.scan_hardware_btn = self.device_control.scan_hardware_btn
        self.device_toggle_btn = self.device_control.device_toggle_btn
        self.sample_rate_combo = self.device_control.sample_rate_combo
        self.active_combo = self.channel_controls.active_combo
        self.add_channel_btn = self.device_control.add_channel_btn
        self.available_combo = self.device_control.available_combo
        
        # Connect DeviceControlWidget signals
        self.device_control.deviceSelected.connect(self._on_device_selected)
        self.device_control.deviceConnectRequested.connect(self._on_device_connect_requested)
        self.device_control.deviceDisconnectRequested.connect(self._on_device_disconnect_requested)
        self.device_control.channelAddRequested.connect(self._on_channel_add_requested)
        self.channel_controls.activeChannelSelected.connect(self._on_active_list_index_changed)
        
        # Connect playback control signals (for file source)
        self.device_control.playPauseToggled.connect(self._on_playback_toggled)
        self.device_control.seekRequested.connect(self._on_seek_requested)

        grid.addWidget(side_panel, 0, 2)  # Trigger panel top right (col 2)
        grid.addWidget(self.device_control, 1, 0) # Device control bottom left (col 0)
        grid.addWidget(self.channel_controls, 1, 1, 1, 2) # Channel control bottom right (cols 1-2)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

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
        self.threshold_spin.valueChanged.connect(self._on_threshold_spin_changed)
        self.pretrigger_combo.currentIndexChanged.connect(self._emit_trigger_config)
        self.window_combo.currentIndexChanged.connect(self._on_window_changed)
        # NOTE: threshold_line signal now handled by ScopeWidget.thresholdChanged
        # self.active_list.currentItemChanged.connect(self._on_active_channel_selected)
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
            except Exception as exc:
                self._logger.debug("Failed to set pipeline on runtime: %s", exc)

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
        except Exception as exc:
            self._logger.debug("Failed to parse color tuple: %s", exc)
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
        except Exception as exc:
            self._logger.debug("Failed to parse channel config: %s", exc)
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
        except Exception as exc:
            self._logger.debug("Failed to refresh channels after load: %s", exc)

        missing_channels: list[int] = []
        self.active_combo.blockSignals(True)
        self.available_combo.blockSignals(True)
        self.active_combo.clear()
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
                
                self.active_combo.addItem(name, info)
                self.available_combo.removeItem(idx)
                
                cfg = self._channel_config_from_dict(entry.get("config") or {}, fallback_name=name)
                cfg.channel_name = name
                self._channel_configs[int(cid)] = cfg
            if self.active_combo.count():
                self.active_combo.setCurrentIndex(0)
        finally:
            self.active_combo.blockSignals(False)
            self.available_combo.blockSignals(False)

        self._publish_active_channels()
        if missing_channels:
            missing_str = ", ".join(str(cid) for cid in missing_channels)
            _info("Load Config", f"Loaded with missing channels: {missing_str}")
        else:
            msg = f"Scope configuration loaded{f' from {source}' if source else ''}."
            self.statusBar().showMessage(msg, 5000)

    def _emit_trigger_config(self, *args, reset_state: bool = True, update_line: bool = True) -> None:
        data = self.trigger_channel_combo.currentData()
        channel_id = int(data) if data is not None else None
        ui_mode = self._current_trigger_mode()
        self._trigger_channel_id = channel_id
        self._trigger_mode = ui_mode
        self._trigger_threshold = float(self.threshold_spin.value())
        pre_value = self.pretrigger_combo.currentData()
        self._trigger_pre_seconds = float(pre_value if pre_value is not None else 0.0)
        
        # Sync with TriggerController
        window_sec = float(self.window_combo.currentData() or 1.0)
        self._trigger_controller.configure(
            mode=ui_mode,
            channel_id=channel_id,
            threshold=self._trigger_threshold,
            pre_seconds=self._trigger_pre_seconds,
            window_sec=window_sec,
            reset_state=reset_state,
        )
        
        if reset_state:
            self._reset_trigger_state()

        idx = channel_id if channel_id is not None else -1
        visual_config = {
            "channel_index": idx,
            "mode": ui_mode,
            "threshold": self.threshold_spin.value(),
            "hysteresis": 0.0,
            "pretrigger_frac": self._trigger_pre_seconds,
            "window_sec": window_sec,
        }
        self._update_trigger_visuals(visual_config, update_line=update_line)

        self.triggerConfigChanged.emit(dict(visual_config))

    def _on_threshold_spin_changed(self, value: float) -> None:
        """Handle threshold spinbox changes without resetting trigger state."""
        self._emit_trigger_config(reset_state=False)

    def _current_trigger_mode(self) -> str:
        if self.trigger_mode_repeating.isChecked():
            return "continuous"
        if self.trigger_mode_single.isChecked():
            return "single"
        return "stream"

    # ... (skipping unchanged methods) ...

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
        self._emit_trigger_config(reset_state=False, update_line=False)

    # ... (skipping unchanged methods) ...

    def _update_trigger_visuals(self, config: dict, update_line: bool = True) -> None:
        mode = config.get("mode", "stream")
        channel_valid = config.get("channel_index", -1) != -1
        if mode == "stream" or not channel_valid:
            # Use ScopeWidget API to hide threshold
            self.scope.set_threshold(visible=False)
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
        if update_line:
            self.scope.set_threshold(normalized_value, visible=True)
        else:
            self.scope.set_threshold(visible=True)
        
        pen = pg.mkPen((0, 0, 0), width=5)
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
        # Check for launch config preference
        if self._controller and self._controller.app_settings_store:
            settings = self._controller.app_settings_store.get()
            if settings.load_config_on_launch and settings.launch_config_path:
                launch_path = Path(settings.launch_config_path)
                if launch_path.is_file():
                    try:
                        data = json.loads(launch_path.read_text())
                        if isinstance(data, dict):
                            self._apply_scope_config_data(data, source=str(launch_path), show_dialogs=False)
                            self.statusBar().showMessage(f"Loaded launch config: {launch_path.name}", 5000)
                            return
                    except Exception as exc:
                        self.statusBar().showMessage(f"Failed to load launch config: {exc}", 7000)

        # Fallback to default_config.json in CWD
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


    def _on_active_list_index_changed(self, index: int) -> None:
        """Handle active channel selection change by index."""
        if 0 <= index < self.active_combo.count():
            info = self.active_combo.itemData(index)
            self._on_active_channel_selected(info)


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
        
        # Detect if this is a file source device
        is_file_source = False
        if entry is not None:
            driver_name = str(entry.get("driver_name", ""))
            device_name = str(entry.get("name", ""))
            # Check if this is the file source device (check multiple fields)
            if "File" in driver_name or "File" in device_name or (key and "file" in key.lower()):
                is_file_source = True
        
        # Update device control widget mode
        self.device_control.set_file_source_mode(is_file_source)
        
        self._populate_sample_rate_options(entry)
        self._update_sample_rate_enabled()



    def _populate_sample_rate_options(self, entry: Optional[dict]) -> None:
        self.sample_rate_combo.blockSignals(True)
        
        # Determine the target sample rate to restore
        target_rate = 0.0
        if self._device_connected:
            # If connected, the driver config is the absolute source of truth
            driver = getattr(self.runtime, "daq_source", None)
            if driver is not None and getattr(driver, "config", None) is not None:
                target_rate = driver.config.sample_rate
            elif self.runtime.sample_rate > 0:
                target_rate = self.runtime.sample_rate
        else:
            # Otherwise, try to preserve the current UI selection
            target_rate = self._current_sample_rate_value()

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
        
        # Restore selection if possible
        if target_rate > 0:
            self._set_sample_rate_value(target_rate)
        elif self.sample_rate_combo.count():
            self.sample_rate_combo.setCurrentIndex(0)
            
        self.sample_rate_combo.setEnabled(bool(rates) and (not self._device_connected or self.active_combo.count() == 0))
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
        except Exception as exc:
            self._logger.debug("Failed to parse sample rate: %s", exc)
            value = 0.0
            if self.sample_rate_combo.count() > 0:
                d = self.sample_rate_combo.itemData(0)
                try:
                    value = float(d)
                except Exception as exc:
                    self._logger.debug("Failed to parse fallback sample rate: %s", exc)
                    value = 0.0
        # If still zero, use a reasonable default rather than failing
        if value == 0.0:
            value = 20000.0
        return value

    def _update_sample_rate_enabled(self) -> None:
        connected = self._device_connected
        has_active = self.active_combo.count() > 0
        enabled = self.sample_rate_combo.count() > 0 and not connected
        self.sample_rate_combo.setEnabled(enabled)
        if hasattr(self, "device_control") and hasattr(self.device_control, "sample_rate_label"):
            self.device_control.sample_rate_label.setEnabled(enabled)

    def _on_device_connected(self, key: str) -> None:
        self._device_connected = True
        self._apply_device_state(True)
        idx = self.device_combo.findData(key)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self._bind_dispatcher_signals()
        self._bind_app_settings_store()
        self._update_channel_buttons()
        
        # Sync UI with actual negotiated sample rate
        # Prefer driver config (immediate) over runtime metric (delayed)
        driver = getattr(self.runtime, "daq_source", None)
        if driver is not None and getattr(driver, "config", None) is not None:
            self._set_sample_rate_value(driver.config.sample_rate)
        elif self.runtime.sample_rate > 0:
            self._set_sample_rate_value(self.runtime.sample_rate)
        
        # For file source, configure playback controls
        file_source = self._get_file_source()
        if file_source is not None:
            # Ensure file source mode is enabled (triggers playback controls visibility)
            self.device_control.set_file_source_mode(True)
            self.device_control.set_connected(True)  # Re-trigger to show playback controls
            
            # Populate sample rate dropdown with the file's sample rate
            file_rate = file_source._sample_rate
            if file_rate > 0:
                self.device_control.populate_sample_rates([float(file_rate)])
                self._set_sample_rate_value(float(file_rate))
            
            # Start playback timer
            self._start_playback_timer()
            # Set initial playback position
            self._update_playback_position()
            self.device_control.set_playing(True)  # Start in playing state


    def _on_device_disconnected(self) -> None:
        # Stop playback timer
        self._stop_playback_timer()
        self.device_control.reset_playback_controls()
        
        self._device_connected = False
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
        self.active_combo.clear()
        self._clear_channel_panels()
        self.set_trigger_channels([])
        self._update_channel_buttons()
        self._update_sample_rate_enabled()
        
        # Re-apply file source mode based on current device selection
        # This ensures the button says "Browse..." instead of "Click to Connect"
        self._on_device_selected()


    def _on_scan_hardware(self) -> None:
        dm = getattr(self.runtime, "device_manager", None)
        if dm is None:
            return
        try:
            dm.refresh_devices()
        except Exception as exc:
            self._logger.debug("Failed to refresh devices: %s", exc)
        self._publish_active_channels()
        self._clear_scope_display()
        if hasattr(self, "_analysis_dock") and self._analysis_dock is not None:
            try:
                self._analysis_dock.shutdown()
            except Exception as exc:
                self._logger.debug("Failed to shutdown analysis dock: %s", exc)
            self._analysis_dock.select_scope()

    def _on_available_channels(self, channels: Sequence[object]) -> None:
        # Update DeviceControlWidget with available channels
        self.device_control.set_available_channels(list(channels))
        
        # Clear active channels
        self.active_combo.clear()
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
        text = self.available_combo.currentText()
        
        self.active_combo.addItem(text, info)
        self.available_combo.removeItem(idx)
        
        if self.available_combo.count():
            self.available_combo.setCurrentIndex(min(idx, self.available_combo.count() - 1))
            
        # Activate and focus the newly added channel without extra signal chatter.
        self.active_combo.blockSignals(True)
        self.active_combo.setCurrentIndex(self.active_combo.count() - 1)
        self.active_combo.blockSignals(False)
        
        self._update_channel_buttons()
        self._publish_active_channels()
        self._set_active_channel_focus(getattr(info, "id", None))
        self._emit_trigger_config()


    def _publish_active_channels(self) -> None:
        infos = []
        for index in range(self.active_combo.count()):
            info = self.active_combo.itemData(index)
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
            color = self._next_channel_color()
            # print(f"Creating NEW config for ch{channel_id}, color={color}")
            config = ChannelConfig(
                color=color,
                channel_name=channel_name,
                screen_offset=self._initial_screen_offset(),
            )
            self._channel_configs[channel_id] = config
        else:
            # print(f"Reusing config for ch{channel_id}, existing color={config.color}")
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
                self.channel_controls.remove_panel(cid)
                del self._channel_panels[cid]
                self._channel_configs.pop(cid, None)
        if not channel_ids:
            self._show_channel_panel(None)
            return
        for cid, name in desired.items():
            config = self._ensure_channel_config(cid, name)
            panel = self._channel_panels.get(cid)
            if panel is None:
                panel = ChannelDetailPanel(cid, name, self.channel_controls.stack)
                panel.configChanged.connect(lambda cfg, cid=cid: self._on_channel_config_changed(cid, cfg))
                panel.analysisRequested.connect(lambda cid=cid: self._open_analysis_for_channel(cid))
                self.channel_controls.add_panel(cid, panel)
                self._channel_panels[cid] = panel
            panel.set_config(config)
        idx = self.active_combo.currentIndex()
        if idx >= 0:
            info = self.active_combo.itemData(idx)
            cid = getattr(info, "id", None)
            self._show_channel_panel(cid)
        else:
            self._show_channel_panel(channel_ids[0] if channel_ids else None)
            if channel_ids:
                self._set_active_channel_focus(channel_ids[0])

    def _clear_channel_panels(self) -> None:
        self.channel_controls.clear_panels()
        self._channel_panels.clear()
        # Don't clear _channel_configs - configs should persist so channels maintain
        # their color/settings when re-added. Configs are cleared on device disconnect.
        self._channel_last_samples.clear()
        self._channel_display_buffers.clear()
        self._show_channel_panel(None)

    def _show_channel_panel(self, channel_id: Optional[int]) -> None:
        self.channel_controls.show_panel(channel_id)

    def _on_active_channel_selected(self, info: Optional[object]) -> None:
        if info is None:
            self._show_channel_panel(None)
            return
        channel_id = getattr(info, "id", None)
        name = getattr(info, "name", str(info))
        if channel_id is None:
            self._show_channel_panel(None)
            return
        config = self._ensure_channel_config(channel_id, name)
        panel = self._channel_panels.get(channel_id)
        if panel is None:
            panel = ChannelDetailPanel(channel_id, name, self.channel_controls.stack)
            panel.configChanged.connect(lambda cfg, cid=channel_id: self._on_channel_config_changed(cid, cfg))
            panel.analysisRequested.connect(lambda cid=channel_id: self._open_analysis_for_channel(cid))
            self.channel_controls.add_panel(channel_id, panel)
            self._channel_panels[channel_id] = panel
        panel.set_config(config)
        self._set_active_channel_focus(channel_id)
        self._show_channel_panel(channel_id)

    def _select_active_channel_by_id(self, channel_id: int) -> None:
        target_idx = None
        for idx in range(self.active_combo.count()):
            info = self.active_combo.itemData(idx)
            if getattr(info, "id", None) == channel_id:
                target_idx = idx
                break
        if target_idx is None:
            return
        if self.active_combo.currentIndex() != target_idx:
            self.active_combo.setCurrentIndex(target_idx)
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
                renderer = self._renderers.get(channel_id)
                if renderer is not None:
                    renderer.clear()
                self._channel_last_samples.pop(channel_id, None)
                self._channel_display_buffers.pop(channel_id, None)
            else:
                self._channel_last_samples.clear()
                self._channel_display_buffers.clear()
                self._last_times = np.zeros(0, dtype=np.float32)
        
        # If this channel is the current trigger source, update trigger visuals
        # to reflect potential changes in vertical span or offset.
        if self._trigger_channel_id is not None and channel_id == self._trigger_channel_id:
            self._emit_trigger_config(reset_state=False)
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
        renderer = self._renderers.get(channel_id)
        if renderer is None:
            return
            
        # Update renderer config
        renderer.update_config(config)
        
        # Re-push data if we have it
        raw = self._channel_last_samples.get(channel_id)
        if raw is not None and raw.size > 0 and self._last_times.size > 0:
             # We assume times match raw data length roughly or exactly
             # If mismatch, we might skip.
             if raw.shape[-1] == self._last_times.shape[0]:
                 renderer.update_data(raw, self._last_times, downsample=1)
        
        self._apply_active_channel_style()

    def _apply_active_channel_style(self) -> None:
        for cid, renderer in self._renderers.items():
            is_active = cid == self._active_channel_id
            renderer.set_active(is_active)
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
        if not self._channel_ids_current:
            for renderer in self._renderers.values():
                renderer.cleanup()
            self._renderers.clear()
            self._update_plot_y_range()
            return

        # Remove orphaned renderers
        for cid in list(self._renderers.keys()):
            if cid not in self._channel_ids_current:
                self._renderers[cid].cleanup()
                del self._renderers[cid]

        # Ensure all current channels have renderers
        plot_item = self.plot_widget.getPlotItem()
        for cid, name in zip(self._channel_ids_current, self._channel_names):
            self._ensure_channel_config(cid, name)
            if cid not in self._renderers:
                config = self._channel_configs[cid]
                self._renderers[cid] = TraceRenderer(plot_item, config)
        
        self._apply_active_channel_style()

        self._update_plot_y_range()
        self._update_axis_label()

    def _register_chunk(self, data: np.ndarray) -> None:
        self._plot_manager.register_chunk(data)
        # Sync stats back from PlotManager
        self._chunk_rate = self._plot_manager.chunk_rate
        self._chunk_mean_samples = self._plot_manager.chunk_mean_samples
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
                    # Use 2.0 * span because span is the half-range (+/- V)
                    # and the viewport 0.0-1.0 covers the full range (2*span).
                    start = int(np.floor(((0.0 - offset) * 2.0 * span) / step) - 2)
                    end = int(np.ceil(((1.0 - offset) * 2.0 * span) / step) + 2)
                    for n in range(start, end + 1):
                        v = n * step
                        pos = (v / (2.0 * span)) + offset
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
        self._update_trigger_controls()

    def _update_trigger_controls(self) -> None:
        has_active = self.active_combo.count() > 0
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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select Recording File", "", "WAV Files (*.wav);;HDF5 (*.h5);;All Files (*)"
        )
        if path:
            self.record_path_edit.setText(path)

    def _update_record_button_enabled(self, text: str = "") -> None:
        """Enable/disable the record button based on whether a filename is set."""
        has_path = bool(self.record_path_edit.text().strip())
        self.record_toggle_btn.setEnabled(has_path)

    def _toggle_recording(self, checked: bool) -> None:
        if checked:
            path = self.record_path_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.information(self, "Recording", "Please choose a file path before recording.")
                self.record_toggle_btn.setChecked(False)
                return
            rollover = self.record_autoinc.isChecked()
            
            # Auto-increment filename if enabled
            if rollover:
                path = self._get_next_filename(path)
            
            self._start_recording_timer()
            self._apply_record_button_style(True)
            self._set_panels_enabled(False)
            self.startRecording.emit(path, rollover)
        else:
            self._stop_recording_timer()
            self._apply_record_button_style(False)
            self._set_panels_enabled(True)
            self.stopRecording.emit()

    def _get_next_filename(self, base_path: str) -> str:
        """
        Find the next available filename with auto-increment.
        
        Given 'path/to/file.wav', checks for:
        - file.wav (returns this if doesn't exist)
        - file1.wav
        - file2.wav
        - etc.
        """
        import os
        import re
        
        directory = os.path.dirname(base_path) or "."
        basename = os.path.basename(base_path)
        name, ext = os.path.splitext(basename)
        
        # If the base file doesn't exist, use it
        if not os.path.exists(base_path):
            return base_path
        
        # Find all existing files matching the pattern
        pattern = re.compile(rf"^{re.escape(name)}(\d*)\.wav$", re.IGNORECASE)
        max_num = 0
        
        try:
            for filename in os.listdir(directory):
                match = pattern.match(filename)
                if match:
                    num_str = match.group(1)
                    if num_str:
                        max_num = max(max_num, int(num_str))
                    else:
                        # Base file exists, we need at least 1
                        max_num = max(max_num, 0)
        except OSError:
            pass
        
        # Next number is max + 1 (but at least 1 if base file exists)
        next_num = max_num + 1
        new_path = os.path.join(directory, f"{name}{next_num}{ext}")
        return new_path

    def _start_recording_timer(self) -> None:
        """Start the recording duration timer."""
        self._recording_start_time = time.perf_counter()
        if self._recording_timer is None:
            self._recording_timer = QtCore.QTimer(self)
            self._recording_timer.timeout.connect(self._update_recording_duration)
        self._recording_timer.start(1000)  # Update every second

    def _stop_recording_timer(self) -> None:
        """Stop the recording duration timer."""
        if self._recording_timer is not None:
            self._recording_timer.stop()
        self._recording_start_time = None

    def _update_recording_duration(self) -> None:
        """Update the record button with elapsed time based on actual data logged."""
        if self._controller is None:
            return
        
        # Get actual duration from controller (based on frames written)
        elapsed = self._controller.recording_duration_seconds
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.record_toggle_btn.setText(f"Stop Recording ({minutes:02d}:{seconds:02d})")

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
        # Safely enable/disable panels that may or may not exist
        for attr_name in ("trigger_group", "device_group", "channels_group", "channel_opts_group"):
            panel = getattr(self, attr_name, None)
            if panel is not None:
                panel.setEnabled(enabled)
        
        # Handle individual widgets that should exist
        if hasattr(self, "record_path_edit"):
            self.record_path_edit.setEnabled(enabled)
        if hasattr(self, "record_browse_btn"):
            self.record_browse_btn.setEnabled(enabled)
        if hasattr(self, "record_autoinc"):
            self.record_autoinc.setEnabled(enabled)
        if hasattr(self, "device_combo"):
            self.device_combo.setEnabled(not self._device_connected)
        if hasattr(self, "sample_rate_combo"):
            self.sample_rate_combo.setEnabled(not self._device_connected)
        if hasattr(self, "available_combo"):
            self.available_combo.setEnabled(enabled and self._device_connected)
        if hasattr(self, "active_combo"):
            self.active_combo.setEnabled(enabled and self._device_connected)
        if hasattr(self, "add_channel_btn"):
            self.add_channel_btn.setEnabled(enabled and self._device_connected)
        
        # If disabling, ensure we don't leave stale state
        if not enabled and hasattr(self, "_update_channel_buttons"):
            self._update_channel_buttons()

    def _apply_device_state(self, connected: bool) -> None:
        has_entries = bool(self._device_map)
        has_connectable = any(entry.get("device_id") for entry in self._device_map.values())
        self.device_combo.setEnabled(not connected and has_entries)
        self._update_sample_rate_enabled()
        self.device_toggle_btn.blockSignals(True)
        self.device_toggle_btn.setChecked(connected)
        self.device_toggle_btn.setText("Click to Disconnect" if connected else "Click to Connect")
        self.device_toggle_btn.setEnabled(connected or has_connectable)
        self.device_toggle_btn.blockSignals(False)
        self.available_combo.setEnabled(connected)
        self.available_combo.setVisible(connected)
        self.active_combo.setEnabled(connected)
        self.add_channel_btn.setVisible(connected)
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

        # If this channel is the current trigger source, update trigger visuals
        # so the threshold line follows the trace.
        if self._trigger_channel_id is not None and channel_id == self._trigger_channel_id:
            self._emit_trigger_config(reset_state=False)

    def _on_scope_drag_finished(self) -> None:
        """Handle end of channel drag operation."""
        pass  # Currently no additional action needed

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
        """Reset trigger state. Delegates to TriggerController."""
        self._trigger_controller.reset_state()
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
        self._ensure_renderers_for_ids(self._channel_ids_current, names) # Changed from _ensure_curves
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
        self._ensure_renderers_for_ids(channel_ids, channel_names)
        plot_item = self.plot_widget.getPlotItem()
        target_window = float(self.window_combo.currentData() or 1.0)
        self._current_window_sec = max(target_window, 1e-3)
        plot_item.setXRange(0.0, self._current_window_sec, padding=0.0)
        for renderer in self._renderers.values(): # Changed from _curves
            renderer.clear()
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
        
        # Clear renderers
        for renderer in self._renderers.values():
            renderer.cleanup()
        self._renderers.clear()
        
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

    def _ensure_renderers_for_ids(self, channel_ids: Sequence[int], channel_names: Sequence[str]) -> None:
        """Synchronize the TraceRenderers with the current active channel list."""
        # Ensure configs exist for all channels before delegating
        for cid, name in zip(channel_ids, channel_names):
            self._ensure_channel_config(cid, name)
        
        # Delegate to PlotManager
        self._plot_manager.ensure_renderers_for_ids(
            channel_ids, channel_names, self._channel_configs
        )
        
        # Sync state back
        self._channel_ids_current = list(channel_ids)
        self._channel_names = list(channel_names) if channel_names else [f"Ch {i}" for i in channel_ids]
        self._renderers = self._plot_manager.renderers
        
        # Force plot update
        self.plot_widget.getPlotItem().update()
        self._refresh_channel_layout()

    def _maybe_update_analysis_sample_rate(self, sample_rate: float) -> None:
        """Update analysis dock with sample rate changes."""
        if sample_rate <= 0:
            return
        if abs(sample_rate - self._analysis_sample_rate) < 1e-3:
            return
        self._analysis_sample_rate = float(sample_rate)
        dock = getattr(self, "_analysis_dock", None)
        if isinstance(dock, AnalysisDock):
            dock.update_sample_rate(sample_rate)


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

        # RACE CONDITION FIX: Trust the channel metadata from the payload rather than
        # filtering against the GUI's current state. The payload was generated by the  
        # Dispatcher thread at the time the data was captured, so its channel_ids
        # correctly reflect which channels the data belongs to.
        #
        # The GUI should simply render data for whatever channels are present in the
        # incoming payload. If a channel ID in the payload doesn't have a corresponding
        # renderer in the GUI (because the user removed it), Python will handle that
        # gracefully in the rendering code below (the renderer lookup will fail and
        # skip that channel).
        #
        # This approach avoids the previous race condition where:
        # 1. Dispatcher sends a chunk for "Channel A"
        # 2. User rapidly switches to "Channel B"  
        # 3. GUI filters out Channel A data even though it's valid data that should
        #    have been displayed before the switch
        #
        # Note: The old code filtered based on _channel_ids_current which belongs to
        # the GUI thread, but the payload was generated milliseconds ago when the
        # channel selection may have been different.
        #
        # If channel_ids from payload don't match any renderers, the rendering loop
        # will simply skip them (no crash, just no display for removed channels).

        if not self._plot_manager.renderers:
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
            self._plot_manager.process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)
            # Sync state back from PlotManager
            self._current_sample_rate = self._plot_manager.sample_rate
            self._current_window_sec = self._plot_manager.window_sec
            self._chunk_rate = self._plot_manager.chunk_rate
            self._chunk_mean_samples = self._plot_manager.chunk_mean_samples
            self._channel_last_samples = self._plot_manager.channel_last_samples
            self._last_times = self._plot_manager.last_times
            try:
                self.runtime.update_metrics(sample_rate=sample_rate, plot_refresh_hz=self._plot_manager.actual_plot_refresh_hz)
            except Exception:
                pass
        else:
            self._plot_manager.process_trigger_mode(
                data, times_arr, sample_rate, window_sec, channel_ids, now,
                trigger_mode=self._trigger_mode,
                trigger_channel_id=self._trigger_channel_id,
                pretrigger_line=self.pretrigger_line,
            )
            # Sync state back from PlotManager
            self._current_sample_rate = self._plot_manager.sample_rate
            self._current_window_sec = self._plot_manager.window_sec
            self._channel_last_samples = self._plot_manager.channel_last_samples
            self._last_times = self._plot_manager.last_times
            try:
                self.runtime.update_metrics(sample_rate=sample_rate)
            except Exception:
                pass
        self._update_status(viz_depth=0)

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

    # -------------------------------------------------------------------------
    # File Source Playback Controls
    # -------------------------------------------------------------------------

    def _get_file_source(self):
        """Get the current device if it's a FileSource, otherwise None."""
        from daq.file_source import FileSource
        source = getattr(self.runtime, "daq_source", None)
        if isinstance(source, FileSource):
            return source
        return None

    def _on_playback_toggled(self, is_playing: bool) -> None:
        """Handle play/pause toggle from device control widget."""
        source = self._get_file_source()
        if source is None:
            return
        source.set_paused(not is_playing)

    def _on_seek_requested(self, position_secs: float) -> None:
        """Handle seek request from device control widget."""
        source = self._get_file_source()
        if source is None:
            self.device_control.clear_seek_pending()
            return
        source.seek_to_position(position_secs)
        # Immediately update the display to show the new position
        self._update_playback_position()
        # Clear the seek pending flag now that seek is complete
        self.device_control.clear_seek_pending()



    def _update_playback_position(self) -> None:
        """Update playback position display (called by timer)."""
        source = self._get_file_source()
        if source is None:
            return
        pos = source.current_position_seconds
        dur = source.total_duration_seconds
        self.device_control.update_playback_position(pos, dur)
        self.device_control.set_playing(not source.is_paused)

    def _start_playback_timer(self) -> None:
        """Start the playback position update timer."""
        if hasattr(self, "_playback_timer") and self._playback_timer is not None:
            self._playback_timer.start()

    def _stop_playback_timer(self) -> None:
        """Stop the playback position update timer."""
        if hasattr(self, "_playback_timer") and self._playback_timer is not None:
            self._playback_timer.stop()

