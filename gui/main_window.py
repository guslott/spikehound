from __future__ import annotations

from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:  # pragma: no cover
    from core.controller import PipelineController

from core import DeviceManager


class MainWindow(QtWidgets.QMainWindow):
    """Skeleton window showcasing the future SpikeHound layout."""

    startRequested = QtCore.Signal()
    stopRequested = QtCore.Signal()
    recordToggled = QtCore.Signal(bool)
    startRecording = QtCore.Signal(str, bool)
    stopRecording = QtCore.Signal()
    triggerConfigChanged = QtCore.Signal(dict)

    def __init__(self, controller: Optional["PipelineController"] = None) -> None:
        super().__init__()
        self._controller: Optional["PipelineController"] = None
        self.setWindowTitle("SpikeHound")
        self.resize(1100, 720)
        self.statusBar()

        self._curves: List[pg.PlotCurveItem] = []
        self._channel_names: List[str] = []
        self._chunk_rate: float = 0.0
        self._device_map: Dict[str, dict] = {}
        self._device_connected = False
        self._active_channel_infos: List[object] = []
        self._channel_ids_current: List[int] = []
        self._offset_step = 0.5
        self._channel_offsets: Dict[int, float] = {}
        self._current_sample_rate: float = 0.0
        self._current_window_sec: float = 0.0

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
        self.plot_widget = pg.PlotWidget()
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

        # Upper-right: stacked control boxes (Recording, Trigger).
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
        browse_btn.setStyleSheet("color: rgb(0,0,0);")
        browse_btn.clicked.connect(self._on_browse_record_path)
        path_row.addWidget(browse_btn)
        record_layout.addLayout(path_row)

        self.record_autoinc = QtWidgets.QCheckBox("Auto-increment filename if exists")
        self.record_autoinc.setStyleSheet("color: rgb(0,0,0);")
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
        self.trigger_channel_combo.setStyleSheet("color: rgb(0,0,0);")
        trigger_layout.addWidget(self.trigger_channel_combo, row, 1)
        row += 1

        trigger_layout.addWidget(self._label("Mode"), row, 0)
        mode_layout = QtWidgets.QVBoxLayout()
        self.trigger_mode_continuous = QtWidgets.QRadioButton("No Trigger (Continuous)")
        self.trigger_mode_single = QtWidgets.QRadioButton("Single")
        self.trigger_mode_repeating = QtWidgets.QRadioButton("Continuous Trigger")
        for btn in (self.trigger_mode_continuous, self.trigger_mode_single, self.trigger_mode_repeating):
            btn.setStyleSheet("color: rgb(0,0,0);")
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
        pretrig_box.addWidget(self._label("Pre-trigger (%)"))
        self.pretrigger_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pretrigger_slider.setRange(0, 90)
        self.pretrigger_slider.setValue(10)
        pretrig_box.addWidget(self.pretrigger_slider)
        trigger_layout.addLayout(pretrig_box, row, 0, 1, 2)
        row += 1

        window_box = QtWidgets.QHBoxLayout()
        window_box.addWidget(self._label("Window (s)"))
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.setStyleSheet("color: rgb(0,0,0);")
        for value in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
            self.window_combo.addItem(f"{value:.1f}", value)
        self.window_combo.setCurrentIndex(1)
        window_box.addWidget(self.window_combo)
        trigger_layout.addLayout(window_box, row, 0, 1, 2)

        side_layout.addWidget(self.trigger_group)

        side_layout.addStretch(1)
        grid.addWidget(side_panel, 0, 1)

        # Bottom row (spanning full width): device / channel controls.
        bottom_panel = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        self.device_group = QtWidgets.QGroupBox("Device")
        device_layout = QtWidgets.QGridLayout(self.device_group)
        device_layout.setColumnStretch(1, 1)

        device_layout.addWidget(self._label("Source"), 0, 0)
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setStyleSheet("color: rgb(0,0,0);")
        device_layout.addWidget(self.device_combo, 0, 1)

        device_layout.addWidget(self._label("Sample Rate (Hz)"), 1, 0)
        self.sample_rate_spin = QtWidgets.QDoubleSpinBox()
        self.sample_rate_spin.setRange(100.0, 1_000_000.0)
        self.sample_rate_spin.setDecimals(0)
        self.sample_rate_spin.setSingleStep(1000.0)
        self.sample_rate_spin.setValue(20_000.0)
        self.sample_rate_spin.setStyleSheet("color: rgb(255,255,255); background-color: rgb(0,0,0);")
        device_layout.addWidget(self.sample_rate_spin, 1, 1)

        self.rescan_btn = QtWidgets.QPushButton("Rescan Devices")
        self.rescan_btn.clicked.connect(self._on_rescan_devices)
        device_layout.addWidget(self.rescan_btn, 2, 0, 1, 2)

        self.device_toggle_btn = QtWidgets.QPushButton("Connect")
        self.device_toggle_btn.setCheckable(True)
        self.device_toggle_btn.clicked.connect(self._on_device_button_clicked)
        device_layout.addWidget(self.device_toggle_btn, 3, 0, 1, 2)

        help_label = QtWidgets.QLabel(
            "Drop a *.py driver in daq/ that subclasses BaseSource, then click Rescan to load it."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: rgb(40,40,40); font-size: 11px;")
        device_layout.addWidget(help_label, 4, 0, 1, 2)

        bottom_layout.addWidget(self.device_group, 1)

        self.channels_group = QtWidgets.QGroupBox("Channels")
        channels_layout = QtWidgets.QGridLayout(self.channels_group)

        channels_layout.addWidget(self._label("Available"), 0, 0)
        channels_layout.addWidget(self._label("Active"), 0, 2)

        self.available_list = QtWidgets.QListWidget()
        self.available_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.available_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.available_list.setMaximumHeight(60)
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
        self.active_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.active_list.setMaximumHeight(60)
        channels_layout.addWidget(self.active_list, 1, 2)

        bottom_layout.addWidget(self.channels_group, 1)

        self.channel_opts_group = QtWidgets.QGroupBox("Channel Options")
        channel_opts_layout = QtWidgets.QVBoxLayout(self.channel_opts_group)
        channel_opts_layout.addWidget(self._label("Per-channel options coming soon."))
        channel_opts_layout.addStretch(1)
        bottom_layout.addWidget(self.channel_opts_group, 1)

        grid.addWidget(bottom_panel, 1, 0, 1, 2)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 0)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 3)

    def _wire_placeholders(self) -> None:
        """Connect stub widgets to stub slots for future wiring."""
        self.trigger_channel_combo.currentTextChanged.connect(self._emit_trigger_config)
        self.trigger_mode_continuous.toggled.connect(self._emit_trigger_config)
        self.trigger_mode_single.toggled.connect(self._emit_trigger_config)
        self.trigger_mode_repeating.toggled.connect(self._emit_trigger_config)
        self.threshold_spin.valueChanged.connect(self._emit_trigger_config)
        self.pretrigger_slider.valueChanged.connect(self._emit_trigger_config)
        self.window_combo.currentIndexChanged.connect(self._emit_trigger_config)
        self.threshold_line.sigPositionChanged.connect(self._on_threshold_line_changed)

    def attach_controller(self, controller: Optional["PipelineController"]) -> None:
        if controller is self._controller:
            return

        if self._controller is not None:
            try:
                self.triggerConfigChanged.disconnect(self._controller.update_trigger_config)
            except (TypeError, RuntimeError):
                pass
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

        self.triggerConfigChanged.connect(controller.update_trigger_config)
        self.startRecording.connect(controller.start_recording)
        self.stopRecording.connect(controller.stop_recording)

        signals = controller.dispatcher_signals()
        if signals is not None:
            signals.tick.connect(self._on_dispatcher_tick)

        self._update_status(0)

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
            QPushButton { background-color: rgb(223,223,223); }
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
            "pretrigger_frac": self.pretrigger_slider.value() / 100.0,
            "window_sec": float(self.window_combo.currentData() or 0.0),
        }
        self._update_trigger_visuals(config)
        self.triggerConfigChanged.emit(config)

    def _current_trigger_mode(self) -> str:
        if self.trigger_mode_single.isChecked():
            return "single"
        if self.trigger_mode_repeating.isChecked():
            return "continuous_trigger"
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
        try:
            self._device_manager.connect_device(key, sample_rate=self.sample_rate_spin.value())
        except Exception as exc:  # pragma: no cover - GUI feedback only
            QtWidgets.QMessageBox.critical(self, "Device", f"Failed to connect: {exc}")
            self.device_toggle_btn.blockSignals(True)
            self.device_toggle_btn.setChecked(False)
            self.device_toggle_btn.blockSignals(False)

    def _on_rescan_devices(self) -> None:
        if not hasattr(self, "_device_manager") or self._device_manager is None:
            return
        self._device_manager.refresh_devices()
        if not self._device_connected:
            self.available_list.clear()
            self.active_list.clear()
            self._publish_active_channels()

    def _on_devices_changed(self, entries: List[dict]) -> None:
        self._device_map = {entry["key"]: entry for entry in entries}
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for entry in entries:
            self.device_combo.addItem(entry["name"], entry["key"])
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
        self._update_channel_buttons()

    def _on_device_disconnected(self) -> None:
        self._device_connected = False
        self._apply_device_state(False)
        if self._controller is not None:
            self._controller.detach_device()
            self._controller.clear_active_channels()
        self.available_list.clear()
        self.active_list.clear()
        self.set_trigger_channels([])
        self._update_channel_buttons()
        self._publish_active_channels()

    def _on_available_channels(self, channels: Sequence[object]) -> None:
        self.available_list.clear()
        self.active_list.clear()
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
        if self._controller is not None:
            ids = [getattr(info, "id", None) for info in infos]
            ids = [cid for cid in ids if cid is not None]
            if ids:
                self._controller.set_active_channels(ids)
            else:
                self._controller.clear_active_channels()

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
        has_devices = bool(self._device_map)
        self.device_combo.setEnabled(not connected and has_devices)
        self.sample_rate_spin.setEnabled(not connected)
        self.device_toggle_btn.blockSignals(True)
        self.device_toggle_btn.setChecked(connected)
        self.device_toggle_btn.setText("Disconnect" if connected else "Connect")
        self.device_toggle_btn.setEnabled(connected or has_devices)
        self.device_toggle_btn.blockSignals(False)
        self.rescan_btn.setEnabled(True)
        self.available_list.setEnabled(connected)
        self.active_list.setEnabled(connected)
        self._update_channel_buttons()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if hasattr(self, "_device_manager") and self._device_manager is not None:
            try:
                self._device_manager.disconnect_device()
            except Exception:
                pass
        super().closeEvent(event)

    def _quit_application(self) -> None:
        QtWidgets.QApplication.instance().exit()

    def _on_threshold_line_changed(self) -> None:
        value = float(self.threshold_line.value())
        if abs(self.threshold_spin.value() - value) > 1e-6:
            self.threshold_spin.blockSignals(True)
            self.threshold_spin.setValue(value)
            self.threshold_spin.blockSignals(False)
        self._emit_trigger_config()

    def _update_trigger_visuals(self, config: dict) -> None:
        visible = config["mode"] != "continuous"
        self.threshold_line.setVisible(visible)
        self.pretrigger_line.setVisible(visible)
        if not visible:
            return

        threshold = config["threshold"]
        self.threshold_line.blockSignals(True)
        self.threshold_line.setValue(threshold)
        self.threshold_line.blockSignals(False)

        window_seconds = float(config["window_sec"] or 0.0)
        pretrigger_frac = float(config["pretrigger_frac"])
        x_pos = pretrigger_frac * window_seconds
        self.pretrigger_line.setValue(x_pos)

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

    def _ensure_curves_for_ids(self, channel_ids: Sequence[int], channel_names: Sequence[str]) -> None:
        plot_item = self.plot_widget.getPlotItem()
        for curve in self._curves:
            plot_item.removeItem(curve)
        self._curves = []
        self._channel_names = list(channel_names)
        self._channel_ids_current = list(channel_ids)
        self._channel_offsets = {}
        if not self._channel_ids_current:
            return
        for index, cid in enumerate(self._channel_ids_current):
            pen = pg.mkPen(color=pg.intColor(index, hues=max(len(self._channel_ids_current), 1)), width=1.5)
            curve = pg.PlotCurveItem(pen=pen)
            plot_item.addItem(curve)
            self._curves.append(curve)
            self._channel_offsets[cid] = index * self._offset_step

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

        for idx, curve in enumerate(self._curves):
            if idx < data.shape[0] and idx < len(channel_ids):
                cid = channel_ids[idx]
                offset = self._channel_offsets.get(cid, idx * self._offset_step)
                curve.setData(times_arr, data[idx] + offset, skipFiniteCheck=True)
            else:
                curve.clear()

        if window_sec > 0:
            self.plot_widget.getPlotItem().setXRange(0, window_sec, padding=0)
        if self._channel_offsets:
            max_offset = max(self._channel_offsets.values()) if self._channel_offsets else 0.0
            self.plot_widget.getPlotItem().setYRange(-self._offset_step, max_offset + self._offset_step, padding=0)

        self._current_sample_rate = sample_rate
        self._current_window_sec = window_sec
        self._chunk_rate = sample_rate
        self._update_status(viz_depth=0)

    @QtCore.Slot(bool)
    def on_record_toggled(self, enabled: bool) -> None:
        """Placeholder slot that will route a record toggle to the controller."""
        self.recordToggled.emit(enabled)
