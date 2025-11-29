"""DeviceControlWidget - Device selection and channel management controls.

Extracted from MainWindow to provide focused device connection and channel
management UI components.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6 import QtCore, QtWidgets


class DeviceControlWidget(QtWidgets.QWidget):
    """Widget for device selection, connection, and channel management."""

    # Signals
    deviceSelected = QtCore.Signal(str)  # device_key
    deviceConnectRequested = QtCore.Signal(str, float)  # device_key, sample_rate
    deviceDisconnectRequested = QtCore.Signal()
    deviceDisconnectRequested = QtCore.Signal()
    # deviceScanRequested = QtCore.Signal()  # Moved to SettingsTab
    channelAddRequested = QtCore.Signal(int)  # channel_id
    channelRemoveRequested = QtCore.Signal(int)  # channel_id
    activeChannelSelected = QtCore.Signal(int)  # index in active list
    sampleRateChanged = QtCore.Signal(float)
    saveConfigRequested = QtCore.Signal()
    loadConfigRequested = QtCore.Signal()
    settingsToggled = QtCore.Signal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        
        # State
        self._device_map: Dict[str, dict] = {}
        self._connected = False
        self._available_channels: List = []
        self._active_channel_infos: List = []
        
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
        self.device_group.setMaximumWidth(320)

        device_layout.addWidget(self._label("Source"), 0, 0)
        self.device_combo = QtWidgets.QComboBox()
        device_layout.addWidget(self.device_combo, 0, 1)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setSpacing(6)
        # Scan button moved to Settings tab
        self.device_toggle_btn = QtWidgets.QPushButton("Connect")
        self.device_toggle_btn.setCheckable(True)
        self.device_toggle_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.device_toggle_btn.setFixedHeight(32)
        controls_row.addWidget(self.device_toggle_btn)
        device_layout.addLayout(controls_row, 1, 0, 1, 2)

        device_layout.addWidget(self._label("Sample Rate (Hz)"), 2, 0)
        self.sample_rate_combo = QtWidgets.QComboBox()
        self.sample_rate_combo.setEditable(False)
        self.sample_rate_combo.setFixedHeight(24)
        self.sample_rate_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        device_layout.addWidget(self.sample_rate_combo, 2, 1)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setSpacing(6)
        self.settings_toggle_btn = QtWidgets.QPushButton("Settings")
        self.settings_toggle_btn.setCheckable(True)
        controls_row.addWidget(self.settings_toggle_btn)
        self.save_config_btn = QtWidgets.QPushButton("Save Config")
        controls_row.addWidget(self.save_config_btn)
        self.load_config_btn = QtWidgets.QPushButton("Load Config")
        controls_row.addWidget(self.load_config_btn)
        controls_row.addStretch(1)
        device_layout.addLayout(controls_row, 3, 0, 1, 2)

        layout.addWidget(self.device_group, 1)

        # Channels group (right side)
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
        available_row.addWidget(self.add_channel_btn)
        self.remove_channel_btn = QtWidgets.QPushButton("↓ Remove")
        available_row.addWidget(self.remove_channel_btn)
        channels_layout.addLayout(available_row)
        self.available_combo = QtWidgets.QComboBox()
        channels_layout.addWidget(self.available_combo)

        layout.addWidget(self.channels_group, 2)

        # Connect signals
        self.device_combo.currentIndexChanged.connect(self._on_device_selected)
        # self.scan_hardware_btn.clicked.connect(self._on_scan_clicked)
        self.device_toggle_btn.clicked.connect(self._on_toggle_clicked)
        self.sample_rate_combo.currentIndexChanged.connect(self._on_sample_rate_changed)
        self.settings_toggle_btn.toggled.connect(self._on_settings_toggled)
        self.save_config_btn.clicked.connect(self._on_save_config)
        self.load_config_btn.clicked.connect(self._on_load_config)
        self.add_channel_btn.clicked.connect(self._on_add_channel)
        self.remove_channel_btn.clicked.connect(self._on_remove_channel)
        self.active_list.currentRowChanged.connect(self._on_active_channel_selected)

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
        self.device_toggle_btn.setText("Disconnect" if connected else "Connect")
        self.device_toggle_btn.blockSignals(False)
        
        self.device_combo.setEnabled(not connected)
        # self.scan_hardware_btn.setEnabled(not connected)
        self.sample_rate_combo.setEnabled(not connected)

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
        
        self.active_list.blockSignals(True)
        self.active_list.clear()
        
        for info in infos:
            name = getattr(info, "name", str(info))
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.UserRole, info)
            self.active_list.addItem(item)
        
        self.active_list.blockSignals(False)

    def get_active_channel_count(self) -> int:
        """Return the number of active channels."""
        return self.active_list.count()

    def get_selected_active_index(self) -> int:
        """Return the index of the selected active channel (-1 if none)."""
        return self.active_list.currentRow()

    def set_selected_active_index(self, index: int) -> None:
        """Select an active channel by index."""
        if 0 <= index < self.active_list.count():
            self.active_list.setCurrentRow(index)

    # Signal handlers

    def _on_device_selected(self) -> None:
        """Handle device selection change."""
        device_key = self.device_combo.currentData()
        if device_key is not None:
            self.deviceSelected.emit(device_key)

    # def _on_scan_clicked(self) -> None:
    #     """Handle scan hardware button click."""
    #     self.deviceScanRequested.emit()

    def _on_toggle_clicked(self, checked: bool) -> None:
        """Handle connect/disconnect button click."""
        if checked:
            device_key = self.get_selected_device_key()
            sample_rate = self.get_selected_sample_rate()
            if device_key and sample_rate > 0:
                self.deviceConnectRequested.emit(device_key, sample_rate)
        else:
            self.deviceDisconnectRequested.emit()

    def _on_sample_rate_changed(self) -> None:
        """Handle sample rate selection change."""
        rate = self.get_selected_sample_rate()
        if rate > 0:
            self.sampleRateChanged.emit(rate)

    def _on_settings_toggled(self, checked: bool) -> None:
        """Handle settings button toggle."""
        self.settingsToggled.emit(checked)

    def _on_save_config(self) -> None:
        """Handle save config button click."""
        self.saveConfigRequested.emit()

    def _on_load_config(self) -> None:
        """Handle load config button click."""
        self.loadConfigRequested.emit()

    def _on_add_channel(self) -> None:
        """Handle add channel button click."""
        ch_info = self.available_combo.currentData()
        if ch_info is not None:
            ch_id = getattr(ch_info, "id", None)
            if ch_id is not None:
                self.channelAddRequested.emit(ch_id)

    def _on_remove_channel(self) -> None:
        """Handle remove channel button click."""
        current_row = self.active_list.currentRow()
        if current_row >= 0:
            item = self.active_list.item(current_row)
            info = item.data(QtCore.Qt.UserRole)
            ch_id = getattr(info, "id", None)
            if ch_id is not None:
                self.channelRemoveRequested.emit(ch_id)

    def _on_active_channel_selected(self, index: int) -> None:
        """Handle active channel selection change."""
        if index >= 0:
            self.activeChannelSelected.emit(index)
