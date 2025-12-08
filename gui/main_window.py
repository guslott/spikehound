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
from .recording_control_widget import RecordingControlWidget
from .scope_config_manager import ScopeConfigManager, ScopeConfigProvider
from .trigger_control_widget import TriggerControlWidget
from .channel_manager import ChannelManager
from .audio_listen_manager import AudioListenManager


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
        self._channel_configs: Dict[int, ChannelConfig] = {}
        self._channel_panels: Dict[int, ChannelDetailPanel] = {}
        self._channel_last_samples: Dict[int, np.ndarray] = {}
        self._channel_display_buffers: Dict[int, np.ndarray] = {}
        self._last_times: np.ndarray = np.zeros(0, dtype=np.float32)
        # Color cycle managed by ChannelManager
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
        self._trigger_controller.configChanged.connect(self._on_trigger_config_changed)
        # captureReady signal available for future use if needed
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

        # Config manager for save/load operations
        self._config_manager = ScopeConfigManager(self)

        self._init_ui()
        
        # Timer for updating file playback position
        self._playback_timer = QtCore.QTimer(self)
        self._playback_timer.setInterval(100)  # 10 Hz update rate
        self._playback_timer.timeout.connect(self._update_playback_position)
        
        self.attach_controller(controller)
        self.runtime.device_manager.devicesChanged.connect(self._on_devices_changed)
        self.runtime.device_manager.deviceConnected.connect(self._on_device_connected)
        self.runtime.device_manager.deviceDisconnected.connect(self._on_device_disconnected)
        self.runtime.device_manager.availableChannelsChanged.connect(self._on_available_channels)
        self._apply_device_state(False)
        self.runtime.device_manager.refresh_devices()
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
        pass # _emit_trigger_config removed
        QtCore.QTimer.singleShot(0, self._update_splash_pixmap)
        QtCore.QTimer.singleShot(0, self._try_load_default_config)

    def _init_ui(self) -> None:
        self._settings_tab: Optional[SettingsTab] = None
        self._apply_palette()
        self._style_plot()

        scope_widget = self._create_scope_widget()
        
        # Create PlotManager to handle rendering and data processing
        self._plot_manager = PlotManager(
            plot_widget=self.scope.plot_widget,
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



    def _setup_quit_shortcut(self) -> None:
        """Set up global shortcuts for quitting/closing."""
        quit_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Quit), self)
        quit_shortcut.activated.connect(self._quit_application)

    @property
    def chunk_rate(self) -> float:
        return float(self._chunk_rate)

    @property
    def plot_refresh_hz(self) -> float:
        return float(self._plot_refresh_hz)

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
        

        
        # Connect ScopeWidget signals
        # Connect ScopeWidget signals
        self.scope.viewClicked.connect(self._on_scope_clicked)
        self.scope.viewDragged.connect(self._on_scope_dragged)
        self.scope.viewDragFinished.connect(self._on_scope_drag_finished)
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

        # Recording controls - use extracted widget
        self.record_group = RecordingControlWidget(self)
        self.record_group.recordingStarted.connect(self._on_recording_started)
        self.record_group.recordingStopped.connect(self._on_recording_stopped)
        side_layout.addWidget(self.record_group)


        # Trigger Control Widget (extracted)
        self.trigger_control = TriggerControlWidget(self._trigger_controller)
        side_layout.addWidget(self.trigger_control)



        # Bottom row (spanning full width): device / channel controls.
        self.device_control = DeviceControlWidget(self)
        self.channel_controls = ChannelControlsWidget(self)
        
        # Create ChannelManager to centralize channel state management
        self._channel_manager = ChannelManager(
            channel_controls=self.channel_controls,
            device_control=self.device_control,
            parent=self,
        )
        
        # Create AudioListenManager to centralize audio monitoring
        self._audio_listen_manager = AudioListenManager(parent=self)
        self._audio_listen_manager.set_callbacks(
            get_channel_configs=lambda: self._channel_configs,
            get_channel_panels=lambda: self._channel_panels,
            get_sample_rate=lambda: self._current_sample_rate,
            get_active_channel_ids=lambda: self._channel_ids_current,
            show_message=lambda title, msg: QtWidgets.QMessageBox.information(self, title, msg),
        )
        
        # Wire ChannelManager signals
        self._channel_manager.channelConfigChanged.connect(self._on_channel_manager_config_changed)
        self._channel_manager.activeChannelChanged.connect(self._on_channel_manager_active_changed)
        self._channel_manager.channelsUpdated.connect(self._on_channel_manager_channels_updated)
        self._channel_manager.listenChannelRequested.connect(self._audio_listen_manager.handle_listen_change)
        self._channel_manager.analysisRequested.connect(self._open_analysis_for_channel)
        self._channel_manager.filterSettingsChanged.connect(self._sync_filter_settings)
        
        # Connect DeviceControlWidget signals
        self.device_control.deviceSelected.connect(self._on_device_selected)
        self.device_control.deviceConnectRequested.connect(self._on_device_connect_requested)
        self.device_control.deviceDisconnectRequested.connect(self._on_device_disconnect_requested)
        self.device_control.channelAddRequested.connect(self._on_channel_add_requested)
        # Note: activeChannelSelected is now handled by ChannelManager
        
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

    def attach_controller(self, controller: Optional[PipelineController]) -> None:
        # Check removed to allow re-wiring signals even if controller instance is same


        if self._controller is not None and self._controller is not controller:
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
        if hasattr(self, "_audio_listen_manager"):
            self._audio_listen_manager.set_controller(controller)
        if hasattr(self, "_analysis_dock") and self._analysis_dock is not None:
            self._analysis_dock._controller = self.runtime

        if controller is None:
            return

        self.startRecording.connect(controller.start_recording)
        self.stopRecording.connect(controller.stop_recording)
        self.triggerConfigChanged.connect(controller.update_trigger_config)
        
        # Wire controller to recording widget for duration queries
        if hasattr(self, "record_group") and hasattr(self.record_group, "set_controller"):
            self.record_group.set_controller(controller)

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
        """Delegate to config manager."""
        return self._config_manager.channel_config_to_dict(config)

    def _channel_config_from_dict(self, payload: dict, *, fallback_name: str = "") -> ChannelConfig:
        """Delegate to config manager."""
        return self._config_manager.channel_config_from_dict(payload, fallback_name=fallback_name)

    def _collect_scope_config(self) -> Optional[dict]:
        """Delegate to config manager."""
        return self._config_manager.collect_config()

    def _find_available_index_by_id(self, channel_id: int) -> int:
        """Delegate to config manager."""
        return self._config_manager.find_available_index_by_id(channel_id)

    def _apply_scope_config_data(self, data: dict, source: str = "", *, show_dialogs: bool = True) -> None:
        """Delegate to config manager."""
        self._config_manager.apply_config_data(data, source, show_dialogs=show_dialogs)



    @QtCore.Slot(dict)
    def _on_trigger_config_changed(self, config: dict) -> None:
        """Handle trigger configuration changes from controller."""
        # self._trigger_mode cache removed - use controller or config directly
        
        # Determine if we should clear existing plots
        is_triggered = config.get("mode", "stream") != "stream"
        if not is_triggered:
            # Stream mode
            self.scope.set_pretrigger_position(0.0, visible=False)
            self.scope.set_threshold(visible=False)
        else:
            # Trigger mode
            pre = float(config.get("pre_seconds", 0.01))
            self.scope.set_pretrigger_position(pre, visible=True)
            thresh = float(config.get("threshold", 0.0))
        if self._device_connected:
            self.configure_acquisition(trigger_cfg=config)
        
        self._update_trigger_visuals(config)





    def _on_scope_threshold_changed(self, value: float) -> None:
        """Handle threshold line moved by user."""
        # value is in normalized 0-1 screen coordinates, convert to volts
        cfg = self._channel_configs.get(self._trigger_controller.channel_id) or self._channel_configs.get(self._active_channel_id)
        if cfg is not None:
            span = cfg.vertical_span_v
            offset = cfg.screen_offset
            # Convert from normalized screen coords to volts
            # Must match trace_renderer.py: voltage = (y_norm - offset) * (2 * span)
            voltage = (value - offset) * (2.0 * span)
        else:
            # No channel config, can't convert properly
            voltage = 0.0
        
        self.trigger_control.threshold_spin.blockSignals(True)
        self.trigger_control.threshold_spin.setValue(voltage)
        self.trigger_control.threshold_spin.blockSignals(False)
        # Spinbox change will trigger Controller update via TriggerControlWidget
        # But we need to update line position visually immediately? 
        # Actually setValue triggers valueChanged which TriggerControlWidget handles.
        # It calls configure -> configChanged -> _on_trigger_config_changed -> _update_trigger_visuals.
        # So loop is closed.



    def _update_trigger_visuals(self, config: dict, update_line: bool = True) -> None:
        mode = config.get("mode", "stream")
        channel_valid = config.get("channel_index", -1) != -1
        if mode == "stream" or not channel_valid:
            # Use ScopeWidget API to hide threshold
            self.scope.set_threshold(visible=False)
            self.scope.pretrigger_line.setVisible(False)
            return
        
        # Calculate threshold position in screen coordinates
        threshold_value = config.get("threshold", 0.0)
        cfg = self._channel_configs.get(self._trigger_controller.channel_id) or self._channel_configs.get(self._active_channel_id)
        if cfg is not None:
            span = cfg.vertical_span_v
            offset = cfg.screen_offset
            # Transform threshold from volts to normalized 0-1 screen coords
            # Must match trace_renderer.py: y = (voltage / (2 * span)) + offset
            normalized_value = (threshold_value / (2.0 * span)) + offset
        else:
            normalized_value = 0.5
        
        # Use ScopeWidget API to show and position threshold
        if update_line:
            self.scope.set_threshold(normalized_value, visible=True)
        else:
            self.scope.set_threshold(visible=True)
        
        pen = pg.mkPen((0, 0, 0), width=5)
        self.scope.threshold_line.setPen(pen)
        try:
            self.scope.threshold_line.setZValue(100)
        except AttributeError:
            pass
        pre_value = float(config.get("pretrigger_frac", 0.0) or 0.0)
        if pre_value > 0.0:
            self.scope.pretrigger_line.setVisible(True)
            self.scope.pretrigger_line.setValue(0.0)
        else:
            self.scope.pretrigger_line.setVisible(False)

    def _on_save_scope_config(self) -> None:
        """Delegate to config manager."""
        self._config_manager.save_config()

    def _on_load_scope_config(self) -> None:
        """Delegate to config manager."""
        self._config_manager.load_config()

    def _try_load_default_config(self) -> None:
        """Delegate to config manager."""
        self._config_manager.try_load_default_config()

    def _on_device_button_clicked(self) -> None:
        dm = getattr(self.runtime, "device_manager", None)
        if dm is None:
            return
        if self._device_connected:
            dm.disconnect_device()
            return
        key = self.device_control.device_combo.currentData()
        if not key:
            QtWidgets.QMessageBox.information(self, "Device", "Please select a device to connect.")
            self.device_control.device_toggle_btn.setChecked(False)
            return
        entry = self._device_map.get(key)
        if entry is not None and not entry.get("device_id"):
            message = entry.get("error") or "No hardware devices detected for this driver."
            QtWidgets.QMessageBox.information(self, "Device", message)
            self.device_control.device_toggle_btn.setChecked(False)
            return
        try:
            self.runtime.connect_device(key, sample_rate=self._current_sample_rate_value())
        except Exception as exc:  # pragma: no cover - GUI feedback only
            QtWidgets.QMessageBox.critical(self, "Device", f"Failed to connect: {exc}")
            self.device_control.device_toggle_btn.blockSignals(True)
            self.device_control.device_toggle_btn.setChecked(False)
            self.device_control.device_toggle_btn.blockSignals(False)

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
        added_id = self._channel_manager.on_add_channel()
        if added_id is not None:
            self._update_channel_buttons()
            self._publish_active_channels()
            self._set_active_channel_focus(added_id)

    # -------------------------------------------------------------------------
    # ChannelManager Signal Handlers
    # -------------------------------------------------------------------------

    @QtCore.Slot(int, ChannelConfig)
    def _on_channel_manager_config_changed(self, channel_id: int, config: ChannelConfig) -> None:
        """Handle channel config changes from ChannelManager."""
        # Update local cached state
        self._channel_configs[channel_id] = config
        
        # Update trigger visuals if this is the trigger channel
        if self._trigger_controller.channel_id is not None and channel_id == self._trigger_controller.channel_id:
            cfg = {
                "mode": self._trigger_controller.mode,
                "threshold": self._trigger_controller.threshold,
                "channel_index": self._trigger_controller.channel_id,
                "pretrigger_frac": self._trigger_controller.pre_seconds
            }
            self._update_trigger_visuals(cfg)
        
        # Update channel display
        self._update_channel_display(channel_id)

        
        if self._active_channel_id == channel_id:
            self._update_axis_label()

    @QtCore.Slot(object)
    def _on_channel_manager_active_changed(self, channel_id: object) -> None:
        """Handle active channel changes from ChannelManager."""
        self._active_channel_id = channel_id
        self._apply_active_channel_style()
        self._update_plot_y_range()

    @QtCore.Slot(list, list)
    def _on_channel_manager_channels_updated(self, channel_ids: List[int], channel_names: List[str]) -> None:
        """Handle channel list updates from ChannelManager."""
        # Sync local state
        self._channel_ids_current = list(channel_ids)
        self._channel_names = list(channel_names)
        
        # This is handled by _publish_active_channels for now, 
        # but will eventually be fully managed by ChannelManager


    def _on_devices_changed(self, entries: List[dict]) -> None:
        self._device_map = {entry["key"]: entry for entry in entries}
        self.device_control.device_combo.blockSignals(True)
        self.device_control.device_combo.clear()
        for entry in entries:
            key = entry.get("key")
            name = entry.get("name") or str(key)
            self.device_control.device_combo.addItem(name, key)
            idx = self.device_control.device_combo.count() - 1
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
                self.device_control.device_combo.setItemData(idx, tooltip, QtCore.Qt.ToolTipRole)
        self.device_control.device_combo.blockSignals(False)
        if self._device_connected:
            dm = getattr(self.runtime, "device_manager", None)
            active_key = dm.active_key() if dm is not None else None
            if active_key is not None:
                idx = self.device_control.device_combo.findData(active_key)
                if idx >= 0:
                    self.device_control.device_combo.setCurrentIndex(idx)
        self._on_device_selected()
        self._apply_device_state(self._device_connected and bool(entries))
        self._update_channel_buttons()

    def _on_device_selected(self) -> None:
        key = self.device_control.device_combo.currentData()
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
        self.device_control.sample_rate_combo.blockSignals(True)
        
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

        self.device_control.sample_rate_combo.clear()
        rates = []
        if entry is not None:
            caps = entry.get("capabilities")
            if hasattr(caps, "sample_rates"):
                rates = getattr(caps, "sample_rates") or []
            elif isinstance(caps, dict):
                rates = caps.get("sample_rates") or []
        
        for rate in rates:
            self.device_control.sample_rate_combo.addItem(f"{int(rate):,}", float(rate))
        
        # Restore selection if possible
        if target_rate > 0:
            self._set_sample_rate_value(target_rate)
        elif self.device_control.sample_rate_combo.count():
            self.device_control.sample_rate_combo.setCurrentIndex(0)
            
        self.device_control.sample_rate_combo.setEnabled(bool(rates) and (not self._device_connected or self.channel_controls.active_combo.count() == 0))
        self.device_control.sample_rate_combo.blockSignals(False)

    def _set_sample_rate_value(self, sample_rate: float) -> None:
        if self.device_control.sample_rate_combo.count() == 0:
            return
        idx = self.device_control.sample_rate_combo.findData(float(sample_rate))
        if idx < 0:
            idx = 0
        self.device_control.sample_rate_combo.setCurrentIndex(idx)

    def _current_sample_rate_value(self) -> float:
        data = self.device_control.sample_rate_combo.currentData()
        try:
            value = float(data)
        except Exception as exc:
            self._logger.debug("Failed to parse sample rate: %s", exc)
            value = 0.0
            if self.device_control.sample_rate_combo.count() > 0:
                d = self.device_control.sample_rate_combo.itemData(0)
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
        has_active = self.channel_controls.active_combo.count() > 0
        enabled = self.device_control.sample_rate_combo.count() > 0 and not connected
        self.device_control.sample_rate_combo.setEnabled(enabled)
        if hasattr(self, "device_control") and hasattr(self.device_control, "sample_rate_label"):
            self.device_control.sample_rate_label.setEnabled(enabled)

    def _on_device_connected(self, key: str) -> None:
        self._device_connected = True
        self._apply_device_state(True)
        idx = self.device_control.device_combo.findData(key)
        if idx >= 0:
            self.device_control.device_combo.setCurrentIndex(idx)
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
        # Update widget connected state - this hides playback controls
        self.device_control.set_connected(False)
        
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
        self.device_control.available_combo.clear()
        self.channel_controls.active_combo.clear()
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
        self.channel_controls.active_combo.clear()
        self._clear_channel_panels()
        
        # Set trigger channels and update UI
        if self.device_control.available_combo.count():
            self.device_control.available_combo.setCurrentIndex(0)
        self.set_trigger_channels(channels)
        self._update_channel_buttons()
        self._publish_active_channels()
        self._update_sample_rate_enabled()


    def _publish_active_channels(self) -> None:
        # Delegate channel collection to ChannelManager
        ids, names = self._channel_manager.publish_active_channels()
        
        # Sync local state from ChannelManager
        self._active_channel_infos = self._channel_manager.active_channel_infos
        self._channel_ids_current = list(ids)
        self._channel_names = list(names)
        
        self._update_channel_buttons()
        
        # Check if channels changed for trigger reset
        if list(ids) != self._channel_ids_current:
            self._reset_trigger_state()
        
        # Sync panels via ChannelManager
        self._channel_manager.sync_channel_panels(
            ids, names,
            analysis_dock=getattr(self, "_analysis_dock", None),
        )
        # CRITICAL: Sync configs and panels back to MainWindow BEFORE resetting scope
        # This ensures PlotManager gets the correct configs for creating renderers
        self._channel_panels = self._channel_manager.channel_panels
        self._channel_configs = self._channel_manager.channel_configs
        
        self._reset_scope_for_channels(ids, names)
        self._sync_filter_settings()
        
        # Sync active channel focus via ChannelManager
        self._channel_manager.ensure_active_channel_focus()
        self._active_channel_id = self._channel_manager.active_channel_id
        
        self._channel_last_samples = {cid: self._channel_last_samples[cid] for cid in ids if cid in self._channel_last_samples}
        self._channel_display_buffers = {cid: self._channel_display_buffers[cid] for cid in ids if cid in self._channel_display_buffers}
        
        # Clear listen channel if no longer in active list
        self._audio_listen_manager.clear_if_channel_removed(ids)
        self._listen_channel_id = self._audio_listen_manager.listen_channel_id
        
        infos = self._active_channel_infos
        if infos:
            self.set_trigger_channels(infos)
        else:
            self.set_trigger_channels([])

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
        
        # Ensure AudioManager knows about the current active channels
        if self._controller and hasattr(self._controller, '_audio_manager') and self._controller._audio_manager:
            self._controller._audio_manager.update_active_channels(ids)


    def _ensure_channel_config(self, channel_id: int, channel_name: str) -> ChannelConfig:
        """Delegate to ChannelManager."""
        return self._channel_manager.ensure_channel_config(channel_id, channel_name)

    def _sync_channel_panels(self, channel_ids: Sequence[int], channel_names: Sequence[str]) -> None:
        """Delegate to ChannelManager."""
        self._channel_manager.sync_channel_panels(
            channel_ids, channel_names,
            analysis_dock=getattr(self, "_analysis_dock", None),
        )
        # Sync local state
        self._channel_panels = self._channel_manager.channel_panels
        self._channel_configs = self._channel_manager.channel_configs

    def _clear_channel_panels(self) -> None:
        """Delegate to ChannelManager."""
        self._channel_manager.clear_channel_panels()
        # Clear local display buffers
        self._channel_last_samples.clear()
        self._channel_display_buffers.clear()

    def _show_channel_panel(self, channel_id: Optional[int]) -> None:
        self.channel_controls.show_panel(channel_id)

    def _select_active_channel_by_id(self, channel_id: int) -> None:
        """Delegate to ChannelManager."""
        self._channel_manager.select_active_channel_by_id(channel_id)
        self._active_channel_id = self._channel_manager.active_channel_id

    def _nearest_channel_at_y(self, y: float) -> Optional[int]:
        """Delegate to ChannelManager."""
        return self._channel_manager.get_nearest_channel_at_y(y)

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
        # Handle missing or renamed fields
        if not hasattr(config, "vertical_span_v"):
            try:
                config.vertical_span_v = float(getattr(config, "range_v", 1.0))
            except Exception as e:
                self._logger.debug("Failed to get vertical_span_v: %s", e)
                config.vertical_span_v = 1.0
        if not hasattr(config, "screen_offset"):
            try:
                config.screen_offset = float(getattr(config, "offset_v", 0.5))
            except Exception as e:
                self._logger.debug("Failed to get screen_offset: %s", e)
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
        if self._trigger_controller.channel_id is not None and channel_id == self._trigger_controller.channel_id:
             # Refresh visuals using current controller state
             # We rely on cached _trigger_threshold etc? 
             # Or better, construct a config dict from current simple state?
             # _update_trigger_visuals needs a dict.
             cfg = {
                 "mode": self._trigger_controller.mode,
                 "threshold": self._trigger_controller.threshold,
                 "channel_index": self._trigger_controller.channel_id,
                 "pretrigger_frac": self._trigger_controller.pre_seconds
             }
             self._update_trigger_visuals(cfg)
        self._update_channel_display(channel_id)

        if filters_changed:
            self._sync_filter_settings()
        if self._active_channel_id == channel_id:
            self._update_axis_label()
        self._handle_listen_change(channel_id, config.listen_enabled)
        if display_changed and config.display_enabled and self._channel_ids_current and self._channel_names_current:
            self._reset_scope_for_channels(self._channel_ids_current, self._channel_names_current)

    def _update_channel_display(self, channel_id: int) -> None:
        """Re-render a single channel's curve using the last raw samples and current offset/range."""
        self._plot_manager.update_channel_display(channel_id)

    def _apply_active_channel_style(self) -> None:
        self._plot_manager.set_active_channel(self._active_channel_id)
        self._update_axis_label()

    def _handle_listen_change(self, channel_id: int, enabled: bool) -> None:
        """Delegate to AudioListenManager."""
        self._audio_listen_manager.handle_listen_change(channel_id, enabled)
        # Sync local state
        self._listen_channel_id = self._audio_listen_manager.listen_channel_id

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
        """Delegate to AudioListenManager."""
        self._audio_listen_manager.set_listen_channel(channel_id)
        self._listen_channel_id = self._audio_listen_manager.listen_channel_id

    def _clear_listen_channel(self, channel_id: Optional[int] = None) -> None:
        """Delegate to AudioListenManager."""
        self._audio_listen_manager.clear_listen_channel(channel_id)
        self._listen_channel_id = self._audio_listen_manager.listen_channel_id

    def _ensure_active_channel_focus(self) -> None:
        """Delegate to ChannelManager."""
        self._channel_manager.ensure_active_channel_focus()
        self._active_channel_id = self._channel_manager.active_channel_id
        self._apply_active_channel_style()

    def _set_active_channel_focus(self, channel_id: Optional[int]) -> None:
        """Delegate to ChannelManager."""
        if channel_id is not None and channel_id not in self._channel_manager.channel_ids_current:
            return
        if self._active_channel_id == channel_id:
            self._update_axis_label()
            return
        self._channel_manager.set_active_channel_focus(channel_id)
        self._active_channel_id = self._channel_manager.active_channel_id
        self._apply_active_channel_style()
        self._update_plot_y_range()



    def _register_chunk(self, data: np.ndarray) -> None:
        self._plot_manager.register_chunk(data)
        # Sync stats back from PlotManager
        self._chunk_rate = self._plot_manager.chunk_rate
        self._chunk_mean_samples = self._plot_manager.chunk_mean_samples
        try:
            self.runtime.update_metrics(chunk_rate=self._chunk_rate, sample_rate=self._current_sample_rate)
        except Exception as e:
            self._logger.debug("Failed to update runtime metrics: %s", e)

    def _transform_to_screen(self, raw_data: np.ndarray, span_v: float, offset_pct: float) -> np.ndarray:
        span = max(float(span_v), 1e-9)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.asarray(raw_data, dtype=np.float32) / span + float(offset_pct)
        return np.nan_to_num(result, nan=offset_pct, posinf=offset_pct, neginf=offset_pct)

    def _update_plot_y_range(self) -> None:
        """Fix the normalized viewport to [0.0, 1.0]."""
        plot_item = self.scope.plot_widget.getPlotItem()
        plot_item.setYRange(0.0, 1.0, padding=0.0)

    def _update_axis_label(self) -> None:
        axis = self.scope.plot_widget.getPlotItem().getAxis("left")
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
        self.device_control.add_channel_btn.setEnabled(connected and self.device_control.available_combo.count() > 0)
        # Delegate to trigger control
        if hasattr(self, "trigger_control"):
            self.trigger_control.set_enabled_for_scanning(connected)

    # -------------------------------------------------------------------------
    # Recording signal handlers (logic delegated to RecordingControlWidget)
    # -------------------------------------------------------------------------

    def _on_recording_started(self, path: str, rollover: bool) -> None:
        """Handle recording started signal from RecordingControlWidget."""
        self._set_panels_enabled(False)
        self.startRecording.emit(path, rollover)

    def _on_recording_stopped(self) -> None:
        """Handle recording stopped signal from RecordingControlWidget."""
        self._set_panels_enabled(True)
        self.stopRecording.emit()

    def _set_panels_enabled(self, enabled: bool) -> None:
        # Safely enable/disable panels that may or may not exist
        if hasattr(self, "trigger_control"):
            self.trigger_control.setEnabled(enabled)
            
        for attr_name in ("device_group", "channels_group", "channel_opts_group"):
            panel = getattr(self, attr_name, None)
            if panel is not None:
                panel.setEnabled(enabled)
        
        # Recording widget handles its own enable/disable
        if hasattr(self, "record_group") and hasattr(self.record_group, "set_enabled_for_recording"):
            self.record_group.set_enabled_for_recording(enabled)
        
        if hasattr(self, "device_combo"):
            self.device_control.device_combo.setEnabled(not self._device_connected)
        if hasattr(self, "sample_rate_combo"):
            self.device_control.sample_rate_combo.setEnabled(not self._device_connected)
        if hasattr(self, "available_combo"):
            self.device_control.available_combo.setEnabled(enabled and self._device_connected)
        if hasattr(self, "active_combo"):
            self.channel_controls.active_combo.setEnabled(enabled and self._device_connected)
        if hasattr(self, "add_channel_btn"):
            self.device_control.add_channel_btn.setEnabled(enabled and self._device_connected)
        
        # If disabling, ensure we don't leave stale state
        if not enabled and hasattr(self, "_update_channel_buttons"):
            self._update_channel_buttons()

    def _apply_device_state(self, connected: bool) -> None:
        has_entries = bool(self._device_map)
        has_connectable = any(entry.get("device_id") for entry in self._device_map.values())
        self.device_control.device_combo.setEnabled(not connected and has_entries)
        self._update_sample_rate_enabled()
        self.device_control.device_toggle_btn.blockSignals(True)
        self.device_control.device_toggle_btn.setChecked(connected)
        self.device_control.device_toggle_btn.setText("Click to Disconnect" if connected else "Click to Connect")
        self.device_control.device_toggle_btn.setEnabled(connected or has_connectable)
        self.device_control.device_toggle_btn.blockSignals(False)
        self.device_control.available_combo.setEnabled(connected)
        self.device_control.available_combo.setVisible(connected)
        self.channel_controls.active_combo.setEnabled(connected)
        self.device_control.add_channel_btn.setVisible(connected)
        self._update_channel_buttons()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        dm = getattr(self.runtime, "device_manager", None)
        if dm is not None:
            try:
                dm.disconnect_device()
            except Exception as e:
                self._logger.debug("Exception during device disconnect on close: %s", e)
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
        for i in range(self.trigger_control.window_combo.count()):
            data = self.trigger_control.window_combo.itemData(i)
            if data is not None and abs(float(data) - target) < 1e-6:
                idx = i
                break
        if idx < 0:
            self.trigger_control.window_combo.addItem(f"{target:.2f}", target)
            idx = self.trigger_control.window_combo.count() - 1
        self._window_combo_suppress = True
        self.trigger_control.window_combo.setCurrentIndex(idx)
        self._window_combo_suppress = False
        self._apply_window_value(target)

    def _apply_window_value(self, value: float) -> None:
        self._current_window_sec = max(float(value), 1e-3)
        plot_item = self.scope.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, self._current_window_sec, padding=0.0)
        if self._controller is not None:
            self._controller.update_window_span(self._current_window_sec)
        self._update_status(viz_depth=0)
        if self._trigger_controller.sample_rate > 0:
            self._update_trigger_sample_parameters(self._trigger_controller.sample_rate)

    def _on_window_changed(self) -> None:
        value = float(self.trigger_control.window_combo.currentData() or 0.0)
        if self._window_combo_suppress:
            self._apply_window_value(value)
            return
        self._window_combo_user_set = True
        self._apply_window_value(value)

    def _on_threshold_line_changed(self) -> None:
        y_norm = float(self.scope.threshold_line.value())
        cfg = self._channel_configs.get(self._trigger_controller.channel_id) or self._channel_configs.get(self._active_channel_id)
        span = cfg.vertical_span_v if cfg is not None else 1.0
        offset = cfg.screen_offset if cfg is not None else 0.0
        # Reverse transform: must match trace_renderer.py: voltage = (y_norm - offset) * (2 * span)
        value = (y_norm - offset) * (2.0 * span)
        if abs(self.trigger_control.threshold_spin.value() - value) > 1e-6:
            self.trigger_control.threshold_spin.blockSignals(True)
            self.trigger_control.threshold_spin.setValue(value)
            self.trigger_control.threshold_spin.blockSignals(False)
        pass # _emit_trigger_config removed

    # ScopeWidget signal handlers
    # ScopeWidget signal handlers
    def _on_scope_clicked(self, y: float, button: QtCore.Qt.MouseButton) -> None:
        """Handle click on scope view."""
        if button != QtCore.Qt.MouseButton.LeftButton:
            return
        
        cid = self._plot_manager.get_channel_at_y(y)
        if cid is not None:
            self._select_channel_by_id(cid)
            self._on_scope_dragged(y) # Start drag implicitly

    def _on_scope_dragged(self, y: float) -> None:
        """Handle dragging logic (update active channel offset)."""
        if self._active_channel_id is None:
            return
            
        # We assume active channel is the one being dragged if drag started
        cid = self._active_channel_id
        if cid not in self._channel_configs:
            return
            
        config = self._channel_configs[cid]
        
        y_clamped = max(0.0, min(1.0, float(y)))
        # Snap to center if within 5% of mid
        if abs(y_clamped - 0.5) <= 0.05:
            y_clamped = 0.5
            
        if abs(config.screen_offset - y_clamped) < 1e-6:
            return
            
        config.screen_offset = y_clamped
        # Update PlotManager config
        self._plot_manager.update_channel_configs(self._channel_configs)
        self._plot_manager.update_channel_display(cid)
        
        # Update Channel Detail Panel
        panel = self._channel_panels.get(cid)
        if panel is not None:
             panel.set_config(config)

        # Triggers
        if self._trigger_controller.channel_id is not None and cid == self._trigger_controller.channel_id:
             self._on_trigger_config_changed({
                 "channel_index": cid,
                 "mode": self._trigger_controller.mode,
                 # Trigger widget handles others, but we need to refresh potentially
             })
             # Actually TriggerControlWidget usually updates on its own, but dragging offset
             # doesn't change trigger LEVEL, only visualization of it relative to trace?
             # No, offset changes where the trace is, so threshold might need visual update?
             # ScopeWidget handles threshold line.
             pass

    def _on_scope_drag_finished(self) -> None:
        """Handle end of channel drag operation."""
        # No specific action needed
        pass

    def _reset_trigger_state(self) -> None:
        """Reset trigger state. Delegates to TriggerController."""
        self._trigger_controller.reset_state()
        self.scope.pretrigger_line.setVisible(False)

    def _update_status(self, viz_depth: int) -> None:
        controller = self._controller
        if controller is None:
            stats = {}
            queue_depths: Dict[str, dict] = {}
        else:
            stats = controller.dispatcher_stats()
            queue_depths = controller.queue_depths()

        sr = self._current_sample_rate

        drops = stats.get("dropped", {}) if isinstance(stats, dict) else {}
        evicted = stats.get("evicted", {}) if isinstance(stats, dict) else {}

        # Clear stale chunk rate
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

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def populate_devices(self, devices: Sequence[str]) -> None:
        """Populate the device panel once selection widgets exist."""
        _ = devices

    def set_active_channels(self, channels: Sequence[str]) -> None:
        """Display the channels currently routed to the plot."""
        names = [getattr(ch, "name", str(ch)) for ch in channels]
        self._ensure_renderers_for_ids(self._channel_ids_current, names) # Changed from _ensure_curves
        # Trigger widget handles its own channel logic
        if hasattr(self, "trigger_control"):
            self.trigger_control.update_channels(names)

    def set_trigger_channels(self, channels: Sequence[object], *, current: Optional[int] = None) -> None:
        """Update trigger channel choices presented to the user."""
        if not hasattr(self, "trigger_control"):
            return
            
        # Convert objects to (name, id) list
        formatted_channels = []
        for entry in channels:
            name = getattr(entry, "name", str(entry))
            cid = getattr(entry, "id", None)
            if cid is not None:
                formatted_channels.append((name, cid))
                
        self.trigger_control.update_channels(formatted_channels)
        
        # If specific selection requested
        if current is not None:
            # We need to access the combo directly or add a method.
            # Since TriggerControlWidget exposes combo via alias (implied), we can try:
            idx = self.trigger_control.trigger_channel_combo.findData(current)
            if idx >= 0:
                self.trigger_control.trigger_channel_combo.setCurrentIndex(idx)

    def _reset_scope_for_channels(self, channel_ids: Sequence[int], channel_names: Sequence[str]) -> None:
        target_window = float(self.trigger_control.window_combo.currentData() or 1.0)
        self._plot_manager.reset_scope_for_channels(
            channel_ids, channel_names, self._channel_configs, target_window
        )
        # Sync simple state
        self._current_window_sec = self._plot_manager.window_sec
        self._current_sample_rate = 0.0
        self._chunk_rate = 0.0
        self._chunk_mean_samples = 0.0
        self._update_status(viz_depth=0)
        
        # Ensure local state tracks the new layout
        self._channel_ids_current = list(channel_ids)
        self._channel_names_current = list(channel_names)
        
        if self._controller is not None:
            self._controller.update_window_span(self._current_window_sec)

    def _clear_scope_display(self) -> None:
        self._plot_manager.clear_scope_display()
        self._channel_ids_current = []
        self._channel_names_current = []
        self._active_channel_id = None
        
        # Sync simple state
        default_window = float(self.trigger_control.window_combo.currentData() or 1.0)
        self._current_window_sec = max(default_window, 1e-3)
        self._update_plot_y_range()
        self.scope.threshold_line.setVisible(False)
        self.scope.pretrigger_line.setVisible(False)
        self._current_sample_rate = 0.0
        self._chunk_rate = 0.0
        self._chunk_mean_samples = 0.0
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
        
        # Sync state back partial
        self._channel_ids_current = list(channel_ids)
        self._channel_names_current = list(channel_names) if channel_names else [f"Ch {i}" for i in channel_ids]
    

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
        if self._trigger_controller.mode != "stream":
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
                    self._logger.warning(
                        "VIZ_BUFFER READ FAILED - chunk may have been overwritten before GUI could read it",
                        extra={"start_index": ptr.start_index, "length": ptr.length, "error": str(e)}
                    )
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

        # Register chunk with plot manager
        self._register_chunk(data)

        # Trust channel metadata from payload (Dispatcher source of truth)
        # to avoid race conditions with GUI state.

        if not self._plot_manager.renderers:
            self.scope.plot_widget.getPlotItem().setXRange(0, max(window_sec, 0.001), padding=0)
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            self._chunk_rate = 0.0
            self._chunk_mean_samples = 0.0
            self._chunk_accum_count = 0
            self._chunk_accum_samples = 0
            self._chunk_last_rate_update = time.perf_counter()
            self._update_status(viz_depth=0)
            return
        mode = self._trigger_controller.mode or "stream"
        if mode == "stream":
            self._plot_manager.process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)
            # Sync state back from PlotManager
            self._current_sample_rate = self._plot_manager.sample_rate
            self._current_window_sec = self._plot_manager.window_sec
            self._chunk_rate = self._plot_manager.chunk_rate
            self._chunk_mean_samples = self._plot_manager.chunk_mean_samples
            try:
                self.runtime.update_metrics(sample_rate=sample_rate, plot_refresh_hz=self._plot_manager.actual_plot_refresh_hz)
            except Exception:
                pass
        else:
            self._plot_manager.process_trigger_mode(
                data, times_arr, sample_rate, window_sec, channel_ids, now,
                trigger_mode=self._trigger_controller.mode,
                trigger_channel_id=self._trigger_controller.channel_id,
                pretrigger_line=self.scope.pretrigger_line,
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
        """Slot that will route a record toggle to the controller."""
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
        """Apply audio output device preference."""
        prev = self._audio_listen_manager.listen_device_key if hasattr(self, "_audio_listen_manager") else None
        self._audio_listen_manager.set_listen_device(key)
        self._listen_device_key = self._audio_listen_manager.listen_device_key
        if self._settings_tab is not None:
            self._settings_tab.set_listen_device(key)
        if prev == key:
            return
        # If listen channel is active and device changed, restart monitoring
        if self._audio_listen_manager.listen_channel_id is not None:
            current = self._audio_listen_manager.listen_channel_id
            self._clear_listen_channel(current)
            self._set_listen_channel(current)

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

    # -------------------------------------------------------------------------
    # Properties for ScopeConfigProvider protocol
    # -------------------------------------------------------------------------
    
    @property
    def device_combo(self) -> QtWidgets.QComboBox:
        return self.device_control.device_combo

    @property
    def window_combo(self) -> QtWidgets.QComboBox:
        return self.trigger_control.window_combo

    @property
    def available_combo(self) -> QtWidgets.QComboBox:
        return self.device_control.available_combo

    @property
    def active_combo(self) -> QtWidgets.QComboBox:
        return self.channel_controls.active_combo


