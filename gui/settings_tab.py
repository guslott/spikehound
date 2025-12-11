from __future__ import annotations

import logging
from typing import Dict, Optional
import time
from PySide6 import QtCore, QtWidgets

from shared.app_settings import AppSettings

logger = logging.getLogger(__name__)


class SettingsTab(QtWidgets.QWidget):
    """Settings & Debug panel embedded in the AnalysisDock."""

    saveConfigRequested = QtCore.Signal()
    loadConfigRequested = QtCore.Signal()

    def __init__(self, controller, main_window, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._main_window = main_window
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh_metrics)
        self._init_ui()
        self._load_settings()
        self._timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(self._build_global_box())
        layout.addWidget(self._build_health_box())
        layout.addWidget(self._build_about_box())
        layout.addStretch(1)

    def _build_global_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Global Settings")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(6)

        self.list_audio_check = QtWidgets.QCheckBox("List all audio devices (not just the system default)")
        self.list_audio_check.stateChanged.connect(
            lambda state: self._on_list_audio_toggled(bool(state))
        )
        
        # Rescan button on left, checkbox on right
        audio_devices_row = QtWidgets.QHBoxLayout()
        self.rescan_btn = QtWidgets.QPushButton("Rescan Devices")
        self.rescan_btn.setFixedWidth(130)
        self.rescan_btn.clicked.connect(self._on_rescan_clicked)
        audio_devices_row.addWidget(self.rescan_btn)
        audio_devices_row.addWidget(self.list_audio_check)
        audio_devices_row.addStretch()
        form.addRow(audio_devices_row)

        # Config management and auto-load on same row
        # Save/Load buttons on left, auto-load checkbox on right
        config_row = QtWidgets.QHBoxLayout()
        self.save_config_btn = QtWidgets.QPushButton("Save Config")
        self.save_config_btn.setFixedWidth(90)
        self.save_config_btn.clicked.connect(self.saveConfigRequested.emit)
        self.load_config_btn = QtWidgets.QPushButton("Load Config")
        self.load_config_btn.setFixedWidth(90)
        self.load_config_btn.clicked.connect(self.loadConfigRequested.emit)
        config_row.addWidget(self.save_config_btn)
        config_row.addWidget(self.load_config_btn)
        
        # Launch config checkbox - label will be updated to show path when set
        self._launch_config_path: Optional[str] = None
        self.load_launch_check = QtWidgets.QCheckBox("Load Config on Launch")
        self.load_launch_check.stateChanged.connect(self._on_load_launch_toggled)
        config_row.addWidget(self.load_launch_check)
        config_row.addStretch()
        config_row.addWidget(self.load_launch_check)
        config_row.addStretch()
        form.addRow(config_row)

        # Recording settings (Pro float32 and Auto-increment)
        rec_row = QtWidgets.QHBoxLayout()
        self.rec_float32_check = QtWidgets.QCheckBox("Use 32-bit Pro WAV (float)")
        self.rec_float32_check.setToolTip("Record as 32-bit floating point WAV instead of standard 16-bit PCM. Prevents clipping but creates larger files.")
        self.rec_float32_check.stateChanged.connect(self._on_rec_float32_toggled)
        
        self.rec_autoinc_check = QtWidgets.QCheckBox("Auto-increment filename")
        self.rec_autoinc_check.setToolTip("Automatically append a number if the file already exists (e.g. file1.wav, file2.wav)")
        self.rec_autoinc_check.stateChanged.connect(self._on_rec_autoinc_toggled)

        rec_row.addWidget(self.rec_float32_check)
        rec_row.addWidget(self.rec_autoinc_check)
        rec_row.addStretch()
        form.addRow("Recording Defaults:", rec_row)

        # Audio output device selection
        audio_row = QtWidgets.QHBoxLayout()
        self.listen_combo = QtWidgets.QComboBox()
        self.listen_combo.setMinimumWidth(200)
        self.listen_combo.currentIndexChanged.connect(self._on_listen_device_changed)
        audio_row.addWidget(self.listen_combo, 1)
        self.audio_refresh_btn = QtWidgets.QPushButton("Refresh")
        self.audio_refresh_btn.setFixedWidth(70)
        self.audio_refresh_btn.clicked.connect(self._populate_listen_devices)
        audio_row.addWidget(self.audio_refresh_btn)
        form.addRow("\"Listen\" Output Device:", audio_row)

        return box

    def _build_health_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Health Metrics")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        labels = [
            # Pipeline status
            ("Sample Rate", "sample_rate"),
            ("Uptime", "uptime"),
            ("Chunk rate", "chunk_rate"),
            ("Throughput", "throughput"),
            ("Plot refresh", "plot_refresh"),
            # Queue health
            ("Source queue", "source_queue"),
            ("Viz queue", "viz_queue"),
            ("Audio queue", "audio_queue"),
            ("Logging queue", "logging_queue"),
            # Buffer status
            ("Viz buffer", "viz_buffer"),
            # Data integrity
            ("Xruns", "xruns"),
            ("Drops", "drops"),
            ("Dispatcher", "dispatcher"),
        ]
        self._metric_fields: Dict[str, QtWidgets.QLabel] = {}
        for row, (title, key) in enumerate(labels):
            name_label = QtWidgets.QLabel(title + ":")
            name_label.setStyleSheet("font-weight: bold;")
            grid.addWidget(name_label, row, 0, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            value = QtWidgets.QLabel("–")
            value.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            grid.addWidget(value, row, 1)
            self._metric_fields[key] = value
        
        # Keep columns close together - label column fixed, value column can stretch
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        
        # State for rate calculation
        self._last_frames = 0
        self._last_time = time.perf_counter()
        
        return box

    def _build_about_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("About")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addWidget(QtWidgets.QLabel("SpikeHound 2.0"))
        layout.addWidget(QtWidgets.QLabel("Dr. Gus Lott & Taylor Mangoba"))
        layout.addWidget(QtWidgets.QLabel("License: 0BSD"))
        link = QtWidgets.QLabel(
            '<span>Github Repo: <a href="https://github.com/guslott/spikehound/">github.com/guslott/spikehound/</a></span>'
        )
        link.setOpenExternalLinks(True)
        layout.addWidget(link)
        return box

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        if self._controller is None:
            return
        store = self._controller.app_settings_store
        if store is None:
            return
        settings = store.get()
        self._apply_settings(settings)
        self._populate_listen_devices()

    def _apply_settings(self, settings: AppSettings) -> None:
        self.list_audio_check.blockSignals(True)
        list_all = bool(settings.list_all_audio_devices)
        self.list_audio_check.setChecked(list_all)
        self.list_audio_check.blockSignals(False)
        
        # Sync SoundcardSource flag
        try:
            from daq.soundcard_source import SoundCardSource
            SoundCardSource.set_list_all_devices(list_all)
        except ImportError:
            pass
        
        # Update launch config checkbox state and label
        self._launch_config_path = settings.launch_config_path or None
        self.load_launch_check.blockSignals(True)
        self.load_launch_check.setChecked(bool(settings.load_config_on_launch))
        self.load_launch_check.blockSignals(False)
        self._update_launch_checkbox_label()

        # Update recording checkboxes
        if hasattr(self, "rec_float32_check"):
            self.rec_float32_check.blockSignals(True)
            self.rec_float32_check.setChecked(bool(settings.recording_use_float32))
            self.rec_float32_check.blockSignals(False)

            self.rec_autoinc_check.blockSignals(True)
            self.rec_autoinc_check.setChecked(bool(settings.recording_auto_increment))
            self.rec_autoinc_check.blockSignals(False)

    def _update_launch_checkbox_label(self) -> None:
        """Update the launch config checkbox label to show path if set."""
        if self._launch_config_path:
            self.load_launch_check.setText(f"Load Config on Launch: {self._launch_config_path}")
        else:
            self.load_launch_check.setText("Load Config on Launch")

    def _on_load_launch_toggled(self, state: int) -> None:
        enabled = bool(state)
        if enabled:
            # Opening file picker - show dialog
            if not self._launch_config_path:
                self._on_browse_launch_config()
                # If user cancelled, uncheck the box
                if not self._launch_config_path:
                    self.load_launch_check.blockSignals(True)
                    self.load_launch_check.setChecked(False)
                    self.load_launch_check.blockSignals(False)
                    return
        else:
            # Unchecking - clear the path
            self._launch_config_path = None
            self._update_settings(load_config_on_launch=False, launch_config_path=None)
            self._update_launch_checkbox_label()
            return
        
        self._update_settings(load_config_on_launch=enabled)

    def _on_browse_launch_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Launch Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._launch_config_path = path
            self._update_settings(launch_config_path=path, load_config_on_launch=True)
            self._update_launch_checkbox_label()
            # Ensure checkbox is checked
            if not self.load_launch_check.isChecked():
                self.load_launch_check.blockSignals(True)
                self.load_launch_check.setChecked(True)
                self.load_launch_check.blockSignals(False)

    def _update_settings(self, **kwargs) -> None:
        if self._controller is None:
            return
        store = self._controller.app_settings_store
        if store is None:
            return
        store.update(**kwargs)

    def _on_list_audio_toggled(self, enabled: bool) -> None:
        self._update_settings(list_all_audio_devices=enabled)
        
        # Update SoundcardSource flag
        # We need to import it safely
        try:
            from daq.soundcard_source import SoundCardSource
            SoundCardSource.set_list_all_devices(enabled)
        except ImportError:
            pass
            
        # Refresh output list
        self._populate_listen_devices()
        
        # Refresh input list (via controller scan)
        self._on_rescan_clicked()

    def _on_rec_float32_toggled(self, state: int) -> None:
        self._update_settings(recording_use_float32=bool(state))

    def _on_rec_autoinc_toggled(self, state: int) -> None:
        self._update_settings(recording_auto_increment=bool(state))

    def _on_rescan_clicked(self) -> None:
        """Trigger a device scan via the controller."""
        if self._controller and hasattr(self._controller, "scan_devices"):
            self._controller.scan_devices()

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _populate_listen_devices(self) -> None:
        from audio.player import list_output_devices  # local import to avoid cycles

        list_all = self.list_audio_check.isChecked()
        devices = list_output_devices(list_all=list_all)
        self.listen_combo.blockSignals(True)
        self.listen_combo.clear()
        self.listen_combo.addItem("System Default", None)
        for dev in devices:
            name = dev.get("name") or dev.get("label") or dev.get("id") or "Unknown"
            device_id = dev.get("id")
            self.listen_combo.addItem(str(name), device_id)
        self.listen_combo.blockSignals(False)

        if self._controller is None:
            return
        settings = self._controller.app_settings_store
        if settings is None:
            return
        self.set_listen_device(settings.get().listen_output_key)

    def _on_listen_device_changed(self, index: int) -> None:
        if index < 0:
            return
        key = self.listen_combo.itemData(index)
        value = str(key) if key is not None else None
        if self._controller is not None and hasattr(self._controller, "set_listen_output_device"):
            try:
                self._controller.set_listen_output_device(value)
                return
            except Exception as exc:
                logger.debug("Failed to set listen output device via controller: %s", exc)
        if hasattr(self._main_window, "set_listen_output_device"):
            self._main_window.set_listen_output_device(value)
            return
        if self._controller is None:
            return
        store = self._controller.app_settings_store
        if store is not None:
            store.update(listen_output_key=value)

    def set_listen_device(self, device_key: Optional[str]) -> None:
        if self.listen_combo.count() == 0:
            return
        key = device_key if device_key not in ("", None) else None
        idx = self.listen_combo.findData(key)
        if idx < 0:
            idx = 0
        self.listen_combo.blockSignals(True)
        self.listen_combo.setCurrentIndex(idx)
        self.listen_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _refresh_metrics(self) -> None:
        controller = self._controller
        if controller is None:
            return
        snapshot = {}
        if hasattr(self._main_window, "health_snapshot"):
            try:
                snapshot = self._main_window.health_snapshot()
            except Exception as exc:
                logger.debug("Failed to get health snapshot: %s", exc)
                snapshot = {}
        
        # Sample rate
        sample_rate = snapshot.get("sample_rate", 0.0)
        if sample_rate >= 1000:
            self._metric_fields["sample_rate"].setText(f"{sample_rate / 1000:.1f} kHz")
        else:
            self._metric_fields["sample_rate"].setText(f"{sample_rate:.0f} Hz")
        
        # Uptime (format as HH:MM:SS)
        uptime = snapshot.get("uptime")
        if uptime is not None:
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            self._metric_fields["uptime"].setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self._metric_fields["uptime"].setText("–")
        
        # Chunk rate
        chunk_rate = snapshot.get("chunk_rate", 0.0)
        self._metric_fields["chunk_rate"].setText(f"{chunk_rate:5.2f} chunks/s")
        
        # Throughput (from dispatcher processed_frames)
        stats = snapshot.get("dispatcher", {})
        processed_frames = stats.get("processed_frames", 0) if isinstance(stats, dict) else 0
        now = time.perf_counter()
        dt = now - getattr(self, "_last_time", now)
        if dt > 0.0:
            df = processed_frames - getattr(self, "_last_frames", 0)
            if df < 0:
                df = 0
            rate = df / dt
            self._metric_fields["throughput"].setText(f"{rate:,.1f} samples/s")
            self._last_frames = processed_frames
            self._last_time = now

        # Plot refresh
        plot_hz = snapshot.get("plot_refresh_hz", 0.0)
        self._metric_fields["plot_refresh"].setText(f"{plot_hz:5.1f} Hz")
        
        # Source queue
        source = snapshot.get("source", {})
        src_queue_size = source.get("queue_size", 0)
        src_queue_max = source.get("queue_max", 0)
        if src_queue_max > 0:
            src_util = (src_queue_size / src_queue_max) * 100
            self._metric_fields["source_queue"].setText(f"{src_queue_size}/{src_queue_max} ({src_util:.0f}%)")
        else:
            self._metric_fields["source_queue"].setText("–")
        
        # Queue health
        depths = snapshot.get("queues", {})
        self._metric_fields["viz_queue"].setText(self._format_queue(depths.get("visualization")))
        self._metric_fields["audio_queue"].setText(self._format_queue(depths.get("audio")))
        self._metric_fields["logging_queue"].setText(self._format_queue(depths.get("logging")))
        
        # Viz buffer status
        viz_buf = depths.get("viz_buffer", {})
        if viz_buf:
            seconds_filled = viz_buf.get("seconds", 0.0)
            capacity_sec = viz_buf.get("capacity_seconds", 0.0)
            utilization = viz_buf.get("utilization", 0.0) * 100
            self._metric_fields["viz_buffer"].setText(f"{seconds_filled:.2f} / {capacity_sec:.2f} sec ({utilization:.0f}%)")
        else:
            self._metric_fields["viz_buffer"].setText("–")
        
        # Xruns and Drops
        xruns = source.get("xruns", 0)
        drops = source.get("drops", 0)
        self._metric_fields["xruns"].setText(str(xruns))
        self._metric_fields["drops"].setText(str(drops))
        
        # Dispatcher stats
        dropped = stats.get("dropped", {}) if isinstance(stats, dict) else {}
        forwarded = stats.get("forwarded", {}) if isinstance(stats, dict) else {}
        received = stats.get("received", 0) if isinstance(stats, dict) else 0
        processed = stats.get("processed", 0) if isinstance(stats, dict) else 0
        evicted = stats.get("evicted", {}) if isinstance(stats, dict) else {}
        self._metric_fields["dispatcher"].setText(
            f"recv:{received} proc:{processed} fwd:{forwarded.get('analysis',0)} "
            f"drop:{dropped.get('analysis',0)} evict:{evicted.get('analysis',0)}"
        )

    def _format_queue(self, info: Optional[dict]) -> str:
        if not info:
            return "–"
        size = int(info.get("size", 0))
        maxsize = int(info.get("max", 0))
        util = float(info.get("utilization", 0.0)) * 100.0
        if maxsize <= 0:
            return f"{size}/∞"
        return f"{size}/{maxsize} ({util:3.0f}%)"

