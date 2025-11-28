from __future__ import annotations

from typing import Dict, Optional
import time
from PySide6 import QtCore, QtWidgets

from shared.app_settings import AppSettings


class SettingsTab(QtWidgets.QWidget):
    """Settings & Debug panel embedded in the AnalysisDock."""

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
        layout.addWidget(self._build_audio_box())
        layout.addWidget(self._build_health_box())
        layout.addWidget(self._build_about_box())
        layout.addStretch(1)

    def _build_global_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Global Settings")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(6)

        self.list_audio_check = QtWidgets.QCheckBox("List all audio devices")
        self.list_audio_check.stateChanged.connect(
            lambda state: self._update_settings(list_all_audio_devices=bool(state))
        )
        form.addRow(self.list_audio_check)

        return box

    def _build_audio_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Audio Monitoring")
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        self.listen_combo = QtWidgets.QComboBox()
        layout.addWidget(self.listen_combo, 1)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self._populate_listen_devices)
        layout.addWidget(refresh_btn)
        self.listen_combo.currentIndexChanged.connect(self._on_listen_device_changed)
        return box

    def _build_health_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Health Metrics")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        labels = [
            ("Dispatcher", "dispatcher"),
            ("Visualization queue", "viz_queue"),
            ("Audio queue", "audio_queue"),
            ("Logging queue", "logging_queue"),
            ("Chunk rate", "chunk_rate"),
            ("Throughput", "throughput"),
            ("Plot refresh", "plot_refresh"),
        ]
        self._metric_fields: Dict[str, QtWidgets.QLabel] = {}
        for row, (title, key) in enumerate(labels):
            name_label = QtWidgets.QLabel(title + ":")
            name_label.setStyleSheet("font-weight: bold;")
            grid.addWidget(name_label, row, 0, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            value = QtWidgets.QLabel("–")
            value.setMinimumWidth(220)
            value.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            grid.addWidget(value, row, 1)
            self._metric_fields[key] = value
        
        # State for rate calculation
        self._last_frames = 0
        self._last_time = time.perf_counter()
        
        return box

    def _build_about_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("About")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addWidget(QtWidgets.QLabel("SpikeHound"))
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
        self.list_audio_check.setChecked(bool(settings.list_all_audio_devices))
        self.list_audio_check.blockSignals(False)

    def _update_settings(self, **kwargs) -> None:
        if self._controller is None:
            return
        store = self._controller.app_settings_store
        if store is None:
            return
        store.update(**kwargs)

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _populate_listen_devices(self) -> None:
        from audio.player import list_output_devices  # local import to avoid cycles

        devices = list_output_devices()
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
            except Exception:
                pass
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
            except Exception:
                snapshot = {}
        stats = snapshot.get("dispatcher", {})
        dropped = stats.get("dropped", {}) if isinstance(stats, dict) else {}
        forwarded = stats.get("forwarded", {}) if isinstance(stats, dict) else {}
        received = stats.get("received", 0)
        processed = stats.get("processed", 0)
        processed_frames = stats.get("processed_frames", 0)
        evicted = stats.get("evicted", {}) if isinstance(stats, dict) else {}
        self._metric_fields["dispatcher"].setText(
            f"recv:{received} proc:{processed} fwd:{forwarded.get('analysis',0)} "
            f"drop:{dropped.get('analysis',0)} evict:{evicted.get('analysis',0)}"
        )

        depths = snapshot.get("queues", {})
        self._metric_fields["viz_queue"].setText(self._format_queue(depths.get("visualization")))
        self._metric_fields["audio_queue"].setText(self._format_queue(depths.get("audio")))
        self._metric_fields["logging_queue"].setText(self._format_queue(depths.get("logging")))

        chunk_rate = snapshot.get("chunk_rate", 0.0)
        self._metric_fields["chunk_rate"].setText(f"{chunk_rate:5.2f} chunks/s")
        
        # Calculate throughput
        now = time.perf_counter()
        dt = now - getattr(self, "_last_time", now)
        if dt > 0.0:
            df = processed_frames - getattr(self, "_last_frames", 0)
            # Handle counter reset or initial state
            if df < 0:
                df = 0
            rate = df / dt
            self._metric_fields["throughput"].setText(f"{rate:,.1f} samples/s")
            self._last_frames = processed_frames
            self._last_time = now

        plot_hz = snapshot.get("plot_refresh_hz", 0.0)
        self._metric_fields["plot_refresh"].setText(f"{plot_hz:5.1f} Hz")

    def _format_queue(self, info: Optional[dict]) -> str:
        if not info:
            return "0/0 (0%)"
        size = int(info.get("size", 0))
        maxsize = int(info.get("max", 0))
        util = float(info.get("utilization", 0.0)) * 100.0
        if maxsize <= 0:
            return f"{size}/∞"
        return f"{size}/{maxsize} ({util:3.0f}%)"
