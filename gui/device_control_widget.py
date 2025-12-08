"""DeviceControlWidget - Device selection and channel management controls.

Extracted from MainWindow to provide focused device connection and channel
management UI components.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6 import QtCore, QtWidgets


def _format_time(seconds: float) -> str:
    """Format seconds as mm:ss.xx for time display."""
    if seconds < 0:
        seconds = 0
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


class DeviceControlWidget(QtWidgets.QWidget):
    """Widget for device selection, connection, and channel management."""

    # Signals
    deviceSelected = QtCore.Signal(str)  # device_key
    deviceConnectRequested = QtCore.Signal(str, float)  # device_key, sample_rate
    deviceDisconnectRequested = QtCore.Signal()
    # deviceScanRequested = QtCore.Signal()  # Moved to SettingsTab
    channelAddRequested = QtCore.Signal(int)  # channel_id
    sampleRateChanged = QtCore.Signal(float)
    
    # Playback control signals (for file source)
    playPauseToggled = QtCore.Signal(bool)  # is_playing (True = play, False = pause)
    seekRequested = QtCore.Signal(float)  # position in seconds

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        
        # State
        self._device_map: Dict[str, dict] = {}
        self._connected = False
        self._available_channels: List = []
        self._active_channel_infos: List = []
        self._is_file_source_mode = False
        self._total_duration_secs = 0.0
        
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the device and channel control panels."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Device group (left side)
        self.device_group = QtWidgets.QGroupBox("Device")
        device_layout = QtWidgets.QGridLayout(self.device_group)
        device_layout.setContentsMargins(6, 6, 6, 6)
        device_layout.setHorizontalSpacing(6)
        device_layout.setVerticalSpacing(6)
        device_layout.setColumnStretch(1, 1)

        disabled_style = """
            QComboBox:disabled {
                background-color: #e0e0e0;
                color: #808080;
                border: 1px solid #c0c0c0;
            }
            QLabel:disabled {
                color: #808080;
            }
        """
        
        # Row 0: Device combo
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.device_combo.setStyleSheet(disabled_style)
        device_layout.addWidget(self.device_combo, 0, 0, 1, 2)

        # Row 1: Sample rate + connect button
        self.sample_rate_label = self._label("Sample Rate (Hz)")
        self.sample_rate_label.setStyleSheet(disabled_style)
        device_layout.addWidget(self.sample_rate_label, 1, 0)
        
        sample_rate_row = QtWidgets.QHBoxLayout()
        sample_rate_row.setSpacing(6)
        
        self.sample_rate_combo = QtWidgets.QComboBox()
        self.sample_rate_combo.setEditable(False)
        self.sample_rate_combo.setFixedHeight(24)
        self.sample_rate_combo.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.sample_rate_combo.setStyleSheet(disabled_style)
        sample_rate_row.addWidget(self.sample_rate_combo, stretch=1)
        
        self.device_toggle_btn = QtWidgets.QPushButton("Click to Connect")
        self.device_toggle_btn.setCheckable(True)
        self.device_toggle_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.device_toggle_btn.setFixedHeight(24)
        sample_rate_row.addWidget(self.device_toggle_btn, stretch=1)
        
        device_layout.addLayout(sample_rate_row, 1, 1)

        # Row 2: Available Channels Row
        available_row = QtWidgets.QHBoxLayout()
        available_row.setSpacing(6)
        
        self.available_combo = QtWidgets.QComboBox()
        self.available_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        available_row.addWidget(self.available_combo, stretch=1)
        
        self.add_channel_btn = QtWidgets.QPushButton("Add Channel")
        self.add_channel_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        available_row.addWidget(self.add_channel_btn)
        
        # Initially hidden until connected
        self.available_combo.setVisible(False)
        self.add_channel_btn.setVisible(False)
        
        device_layout.addLayout(available_row, 2, 0, 1, 2)

        # Row 3: Playback controls (for file source only)
        # Layout: [Play/Pause button] [Slider] [Time label]
        playback_row = QtWidgets.QHBoxLayout()
        playback_row.setSpacing(6)
        playback_row.setContentsMargins(0, 0, 0, 0)  # No extra margins
        
        # Play/Pause button (square with standard icons)
        self.play_pause_btn = QtWidgets.QPushButton("▶")
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.setFixedSize(28, 28)
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                padding: 0px;
                border: 1px solid #808080;
                background-color: #e0e0e0;
            }
            QPushButton:checked {
                background-color: #c0c0c0;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)
        self.play_pause_btn.setToolTip("Play/Pause")
        playback_row.addWidget(self.play_pause_btn)
        
        # Position slider (fills available space)
        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.position_slider.setRange(0, 1000)  # Will be updated based on file duration
        self.position_slider.setValue(0)
        self.position_slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.position_slider.setTracking(True)  # Emit valueChanged while dragging
        playback_row.addWidget(self.position_slider, stretch=1)
        
        # Time label: "0:00.00/0:00.00"
        self.time_label = QtWidgets.QLabel("0:00.00/0:00.00")
        self.time_label.setFixedWidth(120)
        self.time_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.time_label.setStyleSheet("font-family: 'Menlo', 'Consolas', 'Courier New', monospace; font-size: 11px;")
        playback_row.addWidget(self.time_label)
        
        # Flag to prevent snap-back after seek
        self._seek_pending = False
        
        # Create a widget to contain the playback row so we can show/hide it
        self.playback_widget = QtWidgets.QWidget()
        self.playback_widget.setContentsMargins(0, 0, 0, 0)  # No extra margins
        self.playback_widget.setLayout(playback_row)
        self.playback_widget.setVisible(False)  # Hidden by default
        
        device_layout.addWidget(self.playback_widget, 3, 0, 1, 2)


        layout.addWidget(self.device_group, 1)

        # Connect signals
        self.device_combo.currentIndexChanged.connect(self._on_device_selected)
        self.device_toggle_btn.clicked.connect(self._on_toggle_clicked)
        self.sample_rate_combo.currentIndexChanged.connect(self._on_sample_rate_changed)
        self.add_channel_btn.clicked.connect(self._on_add_channel)
        
        # Playback control signals
        self.play_pause_btn.toggled.connect(self._on_play_pause_toggled)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.valueChanged.connect(self._on_slider_value_changed)

    def _label(self, text: str) -> QtWidgets.QLabel:
        """Create a standard label."""
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("font-weight: bold;")
        return label

    # Public API - Device Management

    def set_devices(self, entries: List[dict]) -> None:
        """Update the device dropdown with available devices."""
        self._device_map = {entry["key"]: entry for entry in entries}
        
        # Preserve current selection if possible
        current_key = self.device_combo.currentData()
        
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        
        for entry in entries:
            key = entry.get("key")
            name = entry.get("name") or str(key)
            self.device_combo.addItem(name, key)
            idx = self.device_combo.count() - 1
            
            # Build tooltip
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
        
        # Restore selection if it still exists
        if current_key is not None:
            idx = self.device_combo.findData(current_key)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)

    def set_connected(self, connected: bool) -> None:
        """Update UI state based on connection status."""
        self._connected = connected
        self.device_toggle_btn.blockSignals(True)
        self.device_toggle_btn.setChecked(connected)
        
        # Update button text based on mode
        if connected:
            self.device_toggle_btn.setText("Click to Disconnect")
        else:
            self.device_toggle_btn.setText("Browse..." if self._is_file_source_mode else "Click to Connect")
        
        self.device_toggle_btn.blockSignals(False)
        
        self.device_combo.setEnabled(not connected)
        self.sample_rate_combo.setEnabled(not connected)
        
        self.available_combo.setVisible(connected)
        self.add_channel_btn.setVisible(connected)
        
        # Show playback controls only when file source is connected
        self.playback_widget.setVisible(connected and self._is_file_source_mode)

    def set_file_source_mode(self, enabled: bool) -> None:
        """
        Toggle file source mode which shows special UI elements.
        
        When enabled:
        - Connect button says "Browse..."
        - Playback controls become visible when connected
        """
        self._is_file_source_mode = enabled
        
        # Update button text if not connected
        if not self._connected:
            self.device_toggle_btn.setText("Browse..." if enabled else "Click to Connect")
        
        # Update playback widget visibility
        self.playback_widget.setVisible(self._connected and enabled)
        
        # Clear sample rate when switching to file source (will be populated after file selection)
        if enabled and not self._connected:
            self.sample_rate_combo.blockSignals(True)
            self.sample_rate_combo.clear()
            self.sample_rate_combo.blockSignals(False)

    def get_selected_device_key(self) -> Optional[str]:
        """Return the currently selected device key."""
        return self.device_combo.currentData()

    def get_selected_sample_rate(self) -> float:
        """Return the currently selected sample rate."""
        return float(self.sample_rate_combo.currentData() or 0.0)

    def populate_sample_rates(self, rates: List[float]) -> None:
        """Populate the sample rate dropdown."""
        current = self.sample_rate_combo.currentData()
        
        self.sample_rate_combo.blockSignals(True)
        self.sample_rate_combo.clear()
        
        for rate in rates:
            if rate >= 1_000:
                label = f"{rate / 1_000:.1f} kHz"
            else:
                label = f"{rate:.0f} Hz"
            self.sample_rate_combo.addItem(label, rate)
        
        self.sample_rate_combo.blockSignals(False)
        
        # Restore selection if possible
        if current is not None:
            idx = self.sample_rate_combo.findData(current)
            if idx >= 0:
                self.sample_rate_combo.setCurrentIndex(idx)

    # Public API - Channel Management

    def set_available_channels(self, channels: List) -> None:
        """Update the available channels dropdown."""
        self._available_channels = list(channels)
        
        self.available_combo.blockSignals(True)
        self.available_combo.clear()
        
        for ch in channels:
            name = getattr(ch, "name", str(ch))
            # Store the full channel object, not just the ID
            self.available_combo.addItem(name, ch)
        
        self.available_combo.blockSignals(False)

    def set_active_channels(self, infos: List) -> None:
        """Update the active channels list."""
        self._active_channel_infos = list(infos)
        
        self.active_combo.blockSignals(True)
        self.active_combo.clear()
        
        for info in infos:
            name = getattr(info, "name", str(info))
            self.active_combo.addItem(name, info)
        
        self.active_combo.blockSignals(False)

    def get_active_channel_count(self) -> int:
        """Return the number of active channels."""
        return self.active_combo.count()

    def get_selected_active_index(self) -> int:
        """Return the index of the selected active channel (-1 if none)."""
        return self.active_combo.currentIndex()

    def set_selected_active_index(self, index: int) -> None:
        """Select an active channel by index."""
        if 0 <= index < self.active_combo.count():
            self.active_combo.setCurrentIndex(index)

    # Public API - Playback Controls

    def update_playback_position(self, position_secs: float, total_secs: float) -> None:
        """Update the playback position display (for file source)."""
        self._total_duration_secs = total_secs
        
        # Skip all updates if slider is being dragged or seek is pending
        if self.position_slider.isSliderDown() or self._seek_pending:
            return
        
        # Update time label
        self.time_label.setText(f"{_format_time(position_secs)}/{_format_time(total_secs)}")
        
        # Update slider position
        if total_secs > 0:
            slider_pos = int((position_secs / total_secs) * 1000)
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(slider_pos)
            self.position_slider.blockSignals(False)


    def set_playing(self, is_playing: bool) -> None:
        """Update the play/pause button state."""
        self.play_pause_btn.blockSignals(True)
        self.play_pause_btn.setChecked(is_playing)
        self.play_pause_btn.setText("⏸" if is_playing else "▶")
        self.play_pause_btn.blockSignals(False)

    def reset_playback_controls(self) -> None:
        """Reset playback controls to initial state."""
        self._total_duration_secs = 0.0
        self.time_label.setText("0:00.00/0:00.00")
        self.position_slider.setValue(0)
        self.set_playing(False)

    # Signal handlers

    def _on_device_selected(self) -> None:
        """Handle device selection change."""
        device_key = self.device_combo.currentData()
        if device_key is not None:
            self.deviceSelected.emit(device_key)

    def _on_toggle_clicked(self, checked: bool) -> None:
        """Handle connect/disconnect button click."""
        if checked:
            device_key = self.get_selected_device_key()
            sample_rate = self.get_selected_sample_rate()
            # For file source, sample rate may be 0 (determined after file selection)
            if device_key:
                if self._is_file_source_mode or sample_rate > 0:
                    self.deviceConnectRequested.emit(device_key, sample_rate)
                else:
                    # No sample rate selected
                    self.device_toggle_btn.setChecked(False)
        else:
            self.deviceDisconnectRequested.emit()

    def _on_sample_rate_changed(self) -> None:
        """Handle sample rate selection change."""
        rate = self.get_selected_sample_rate()
        if rate > 0:
            self.sampleRateChanged.emit(rate)

    def _on_add_channel(self) -> None:
        """Handle add channel button click."""
        ch_info = self.available_combo.currentData()
        if ch_info is not None:
            ch_id = getattr(ch_info, "id", None)
            if ch_id is not None:
                self.channelAddRequested.emit(ch_id)

    def _on_play_pause_toggled(self, checked: bool) -> None:
        """Handle play/pause button toggle."""
        # checked = True means button is pressed (paused state visually), but we invert for playback
        is_playing = checked
        self.play_pause_btn.setText("⏸" if is_playing else "▶")
        self.playPauseToggled.emit(is_playing)

    def _on_slider_released(self) -> None:
        """Handle slider release for seeking."""
        if self._total_duration_secs <= 0:
            return
        slider_pos = self.position_slider.value()
        position_secs = (slider_pos / 1000.0) * self._total_duration_secs
        
        # Set pending flag to prevent snap-back
        self._seek_pending = True
        self.seekRequested.emit(position_secs)

    def clear_seek_pending(self) -> None:
        """Clear the seek pending flag (called after seek completes)."""
        self._seek_pending = False

    def _on_slider_value_changed(self, value: int) -> None:
        """Update time display and seek while slider is being dragged."""
        if not self.position_slider.isSliderDown():
            return  # Only update during drag
        if self._total_duration_secs <= 0:
            return
        position_secs = (value / 1000.0) * self._total_duration_secs
        self.time_label.setText(f"{_format_time(position_secs)}/{_format_time(self._total_duration_secs)}")
        
        # Emit live seek for smooth scrubbing
        self._seek_pending = True
        self.seekRequested.emit(position_secs)


