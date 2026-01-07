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
        config_row.addStretch(1)
        config_row.addWidget(self.load_launch_check)
        form.addRow(config_row)

        # Recording settings (Pro float32 and Auto-increment)
        rec_row = QtWidgets.QHBoxLayout()
        self.rec_float32_check = QtWidgets.QCheckBox("Use 32-bit Float Pro WAV Format (instead of 16-bit)")
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

        col1_labels = [
            ("Sample Rate", "sample_rate", "The rate at which samples are acquired by the hardware. Expected: Should match the requested sampling rate (e.g., 44.1 kHz, 100 kHz)."),
            ("Uptime", "uptime", "Time since the tool started running. Expected: Continuously increasing during operation."),
            ("Chunk rate", "chunk_rate", "Frequency of data chunks being delivered by the source. Expected: Usually stable (e.g., ~10-100 Hz) depending on hardware settings."),
            ("Throughput", "throughput", "Number of samples processed per second. Expected: Should match the Sample Rate if the system is keeping up."),
            ("Plot refresh", "plot_refresh", "Frequency of UI plot updates. Expected: Usually stabilizes at ~30-60 Hz for smooth visualization."),
            ("Xruns", "xruns", "Hardware under-runs or over-runs reported by the driver. Expected: Should be 0. Any value > 0 indicates potential data loss or timing glitches."),
            ("Drops", "drops", "Total samples dropped by the source before reaching the dispatcher. Expected: Should be 0. Values > 0 indicate the system cannot pull data fast enough."),
        ]
        
        col2_labels = [
            ("Source queue", "source_queue", "Queue between the DAQ source and the dispatcher. Expected: Low utilization (< 50%). High utilization suggests a processing bottleneck."),
            ("Viz queue", "viz_queue", "Data queued for visualization. Expected: Low occupancy. High values indicate UI thread lag."),
            ("Audio queue", "audio_queue", "Data queued for audio playback. Expected: Low occupancy. High values indicate audio driver lag."),
            ("Logging queue", "logging_queue", "Data queued for disk logging. Expected: Low occupancy. High values indicate slow disk I/O."),
            ("Analysis queue", "analysis_queue", "Data queued for real-time analysis (spikes, filters). Expected: Balanced occupancy. Large backlogs indicate analysis code is too slow."),
            ("Viz buffer", "viz_buffer", "Size of the rolling buffer used for plotting. Expected: Stable size based on the \"Scope Time\" setting."),
            ("Buffer headroom", "buffer_headroom", "Safety margin in the hardware/driver buffer before it wraps. Expected: > 5s (Green). Low values (< 5s, Red) indicate imminent wrap-around."),
        ]

        self._metric_fields: Dict[str, QtWidgets.QLabel] = {}
        
        # Populate Column 1 (0, 1)
        for row, (title, key, tooltip) in enumerate(col1_labels):
            name_label = QtWidgets.QLabel(title + ":")
            name_label.setStyleSheet("font-weight: bold;")
            name_label.setToolTip(tooltip)
            grid.addWidget(name_label, row, 0, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            value = QtWidgets.QLabel("–")
            value.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            grid.addWidget(value, row, 1)
            self._metric_fields[key] = value

        # Populate Column 2 (2, 3)
        for row, (title, key, tooltip) in enumerate(col2_labels):
            name_label = QtWidgets.QLabel(title + ":")
            name_label.setStyleSheet("font-weight: bold;")
            name_label.setToolTip(tooltip)
            grid.addWidget(name_label, row, 2, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            value = QtWidgets.QLabel("–")
            value.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            grid.addWidget(value, row, 3)
            self._metric_fields[key] = value
        
        # Dispatcher stats span the bottom
        row = max(len(col1_labels), len(col2_labels))
        name_label = QtWidgets.QLabel("Dispatcher:")
        name_label.setStyleSheet("font-weight: bold;")
        name_label.setToolTip("Internal statistics of the message dispatcher. Expected: Received and Processed counts should be close; drops should be zero.")
        grid.addWidget(name_label, row, 0, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        
        disp_val = QtWidgets.QLabel("–")
        grid.addWidget(disp_val, row, 1, 1, 3) # Span 3 columns
        self._metric_fields["dispatcher"] = disp_val

        # Add a vertical divider line or just use spacing? Spacing is fine.
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 1)
        
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
        if self._controller:
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
        if self._controller is not None:
            try:
                self._controller.set_listen_output_device(value)
                return
            except Exception as exc:
                logger.debug("Failed to set listen output device via controller: %s", exc)
        
        self._main_window.set_listen_output_device(value)

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
        dt = now - self._last_time
        if dt > 0.0:
            df = processed_frames - self._last_frames
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
        
        # Queue health with drop counts
        depths = snapshot.get("queues", {})
        dropped = stats.get("dropped", {}) if isinstance(stats, dict) else {}
        evicted = stats.get("evicted", {}) if isinstance(stats, dict) else {}
        policies = stats.get("policies", {}) if isinstance(stats, dict) else {}
        
        self._metric_fields["viz_queue"].setText(
            self._format_queue_with_drops(depths.get("visualization"), policies.get("visualization", "?"), dropped.get("visualization", 0))
        )
        self._metric_fields["audio_queue"].setText(
            self._format_queue_with_drops(depths.get("audio"), policies.get("audio", "?"), dropped.get("audio", 0))
        )
        self._metric_fields["logging_queue"].setText(
            self._format_queue_with_drops(depths.get("logging"), policies.get("logging", "?"), evicted.get("logging", 0))
        )
        self._metric_fields["analysis_queue"].setText(
            self._format_queue_with_drops(depths.get("analysis"), policies.get("analysis", "?"), evicted.get("analysis", 0) + dropped.get("analysis", 0))
        )
        
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
        
        # Buffer headroom (wrap risk indicator)
        headroom_sec = source.get("buffer_headroom_sec", 0.0)
        capacity_sec = source.get("buffer_capacity_sec", 0.0)
        headroom_label = self._metric_fields.get("buffer_headroom")
        if headroom_label is not None:
            if capacity_sec > 0:
                # Color-code: green > 10s, yellow 5-10s, red < 5s
                if headroom_sec >= 10.0:
                    color = "green"
                elif headroom_sec >= 5.0:
                    color = "orange"
                else:
                    color = "red"
                headroom_label.setText(f"{headroom_sec:.1f} / {capacity_sec:.0f} sec")
                headroom_label.setStyleSheet(f"color: {color};")
            else:
                headroom_label.setText("–")
                headroom_label.setStyleSheet("")
        
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

    def _format_queue_with_drops(self, info: Optional[dict], policy: str, drops: int) -> str:
        """Format queue status with policy abbreviation and drop count."""
        if not info:
            return "–"
        size = int(info.get("size", 0))
        maxsize = int(info.get("max", 0))
        # Policy abbreviations: L=lossless, DN=drop-newest, DO=drop-oldest
        policy_abbr = {"lossless": "L", "drop-newest": "DN", "drop-oldest": "DO"}.get(policy, "?")
        drop_str = f" ↓{drops}" if drops > 0 else ""
        if maxsize <= 0:
            return f"{size}/∞ [{policy_abbr}]{drop_str}"
        return f"{size}/{maxsize} [{policy_abbr}]{drop_str}"
