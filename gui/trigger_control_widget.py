"""TriggerControlWidget - UI wrapper for oscilloscope trigger settings.

This widget provides controls for configuring the trigger subsystem:
- Mode selection: Stream (no trigger), Single, or Repeated Trigger
- Channel selection: Which channel to monitor for trigger events
- Threshold: Voltage level that initiates a capture
- Pre-trigger: Amount of data to capture before the trigger event
- Window width: Total capture duration

The widget syncs its state with a TriggerController instance and forwards
all configuration changes to the controller.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6 import QtWidgets

from .trigger_controller import TriggerController

logger = logging.getLogger(__name__)


class TriggerControlWidget(QtWidgets.QWidget):
    """
    Encapsulates the Trigger settings UI (Group Box).
    
    Responsibilities:
    - Create and layout trigger widgets (Mode, Channel, Threshold, Pre-trigger, Window).
    - Sync UI state with TriggerController.
    - Forward UI events to TriggerController.
    """

    def __init__(
        self,
        controller: TriggerController,
        parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        
        self._setup_ui()
        self._connect_signals()

    @staticmethod
    def _format_window_width(value: float) -> str:
        """Format sweep widths while preserving the 0.05s option."""
        if value < 0.1:
            return f"{value:.2f}"
        return f"{value:.1f}"
        
    def _setup_ui(self) -> None:
        """Create the trigger UI layout."""
        self.trigger_group = QtWidgets.QGroupBox("Trigger")
        trigger_layout = QtWidgets.QGridLayout(self.trigger_group)
        trigger_layout.setContentsMargins(8, 8, 8, 8)
        trigger_layout.setVerticalSpacing(4)
        trigger_layout.setHorizontalSpacing(6)
        
        # Main layout for this widget should just contain the group box
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.trigger_group)

        row = 0
        
        # Channel Selection (populated externally via update_channels()).
        trigger_layout.addWidget(QtWidgets.QLabel("Channel"), row, 0)
        self.trigger_channel_combo = QtWidgets.QComboBox()
        self.trigger_channel_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.trigger_channel_combo.setMaximumWidth(200)
        trigger_layout.addWidget(self.trigger_channel_combo, row, 1)
        row += 1

        # Mode Selection
        trigger_layout.addWidget(QtWidgets.QLabel("Mode"), row, 0)
        mode_layout = QtWidgets.QVBoxLayout()
        mode_layout.setSpacing(2)
        
        self.trigger_mode_stream = QtWidgets.QRadioButton("No Trigger (Stream)")
        self.trigger_mode_single = QtWidgets.QRadioButton("Single")
        # Disabled until a valid trigger channel exists (see _set_trigger_modes_enabled).
        self.trigger_mode_single.setEnabled(False)
        
        self.trigger_mode_repeated = QtWidgets.QRadioButton("Repeated Trigger")
        self.trigger_mode_repeated.setEnabled(False)
        
        # Group radio buttons
        self.trigger_button_group = QtWidgets.QButtonGroup(self)
        self.trigger_button_group.addButton(self.trigger_mode_stream)
        self.trigger_button_group.addButton(self.trigger_mode_single)
        self.trigger_button_group.addButton(self.trigger_mode_repeated)
        
        self.trigger_mode_stream.setChecked(True)
        mode_layout.addWidget(self.trigger_mode_stream)
        
        single_row = QtWidgets.QHBoxLayout()
        single_row.setSpacing(3)
        self.trigger_single_button = QtWidgets.QPushButton("Trigger Once")
        self.trigger_single_button.setEnabled(False)
        single_row.addWidget(self.trigger_mode_single)
        single_row.addWidget(self.trigger_single_button)
        single_row.addStretch(1)
        mode_layout.addLayout(single_row)
        
        mode_layout.addWidget(self.trigger_mode_repeated)
        trigger_layout.addLayout(mode_layout, row, 1)
        row += 1

        # Threshold
        threshold_box = QtWidgets.QHBoxLayout()
        threshold_box.setSpacing(4)
        threshold_box.addWidget(QtWidgets.QLabel("Threshold"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(-10.0, 10.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setValue(0.0)
        self.threshold_spin.setMaximumWidth(100)
        threshold_box.addWidget(self.threshold_spin)
        trigger_layout.addLayout(threshold_box, row, 0, 1, 2)
        row += 1

        # Edge selection + hysteresis (noise reject)
        edge_box = QtWidgets.QHBoxLayout()
        edge_box.setSpacing(4)
        edge_box.addWidget(QtWidgets.QLabel("Edge"))
        self.edge_combo = QtWidgets.QComboBox()
        self.edge_combo.setMaximumWidth(90)
        for label, data in (("Rising", "rising"), ("Falling", "falling"), ("Either", "either")):
            self.edge_combo.addItem(label, data)
        edge_box.addWidget(self.edge_combo)
        edge_box.addSpacing(8)
        edge_box.addWidget(QtWidgets.QLabel("Noise reject (V)"))
        self.hysteresis_spin = QtWidgets.QDoubleSpinBox()
        self.hysteresis_spin.setRange(0.0, 10.0)
        self.hysteresis_spin.setSingleStep(0.01)
        self.hysteresis_spin.setDecimals(3)
        self.hysteresis_spin.setValue(0.0)
        self.hysteresis_spin.setMaximumWidth(90)
        self.hysteresis_spin.setToolTip(
            "Hysteresis band: the signal must retreat this far past the "
            "threshold before the trigger re-arms. 0 disables it."
        )
        edge_box.addWidget(self.hysteresis_spin)
        edge_box.addStretch(1)
        trigger_layout.addLayout(edge_box, row, 0, 1, 2)
        row += 1

        # Pre-trigger
        pretrig_box = QtWidgets.QHBoxLayout()
        pretrig_box.setSpacing(4)
        pretrig_box.addWidget(QtWidgets.QLabel("Pre-trigger (s)"))
        self.pretrigger_combo = QtWidgets.QComboBox()
        self.pretrigger_combo.setMaximumWidth(110)
        for value in (0.0, 0.01, 0.02, 0.05):
            self.pretrigger_combo.addItem(f"{value:.2f}", value)
        self.pretrigger_combo.setCurrentIndex(3)
        pretrig_box.addWidget(self.pretrigger_combo)
        trigger_layout.addLayout(pretrig_box, row, 0, 1, 2)
        row += 1

        # Window Width
        window_box = QtWidgets.QHBoxLayout()
        window_box.setSpacing(4)
        window_box.addWidget(QtWidgets.QLabel("Window Width (s)"))
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.setMaximumWidth(110)
        for value in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
            self.window_combo.addItem(self._format_window_width(value), value)
        default_index = 4  # 1.0 s
        self.window_combo.setCurrentIndex(default_index)
        window_box.addWidget(self.window_combo)
        trigger_layout.addLayout(window_box, row, 0, 1, 2)

    def _connect_signals(self) -> None:
        """Connect internal UI validation and forward changes to controller."""
        # Channels: When selection changes, update controller
        self.trigger_channel_combo.currentIndexChanged.connect(self._on_config_changed)
        
        # Mode
        self.trigger_button_group.buttonClicked.connect(self._on_mode_changed)
        self.trigger_single_button.clicked.connect(self._on_trigger_once_clicked)
        
        # Threshold
        self.threshold_spin.valueChanged.connect(self._on_config_changed)

        # Edge + hysteresis
        self.edge_combo.currentIndexChanged.connect(self._on_config_changed)
        self.hysteresis_spin.valueChanged.connect(self._on_config_changed)

        # Pre-trigger & Window
        self.pretrigger_combo.currentIndexChanged.connect(self._on_config_changed)
        self.window_combo.currentIndexChanged.connect(self._on_config_changed)

    def _on_mode_changed(self, button: QtWidgets.QAbstractButton) -> None:
        """Handle radio button changes."""
        # Enable/Disable "Trigger Once" button
        is_single = (button == self.trigger_mode_single)
        # We only enable the button if single mode is active AND enabled (which implies we have channels)
        self.trigger_single_button.setEnabled(is_single and self.trigger_mode_single.isEnabled())
        self._on_config_changed()

    def _on_trigger_once_clicked(self) -> None:
        """Arm single trigger."""
        self._controller.arm_single()

    def _on_config_changed(self) -> None:
        """Collect UI state and configure controller."""
        # Determine mode
        if self.trigger_mode_stream.isChecked():
            mode = "stream"
        elif self.trigger_mode_single.isChecked():
            mode = "single"
        else:
            mode = "repeated"

        # Channel
        chan_data = self.trigger_channel_combo.currentData()
        channel_id = int(chan_data) if chan_data is not None else None

        # A triggered mode requires a valid channel; TriggerConfig rejects
        # channel_index=None for single/repeated and would raise mid-signal,
        # silently aborting the visual update. Fall back to stream instead.
        if mode in ("single", "repeated") and channel_id is None:
            mode = "stream"
            self.trigger_mode_stream.blockSignals(True)
            self.trigger_mode_stream.setChecked(True)
            self.trigger_mode_stream.blockSignals(False)
            self.trigger_single_button.setEnabled(False)

        # Threshold
        threshold = self.threshold_spin.value()
        
        # Pre-trigger
        pre_data = self.pretrigger_combo.currentData()
        pre_seconds = float(pre_data) if pre_data is not None else 0.01

        # Window
        win_data = self.window_combo.currentData()
        window_sec = float(win_data) if win_data is not None else 1.0

        # Edge + hysteresis
        edge_data = self.edge_combo.currentData()
        edge = str(edge_data) if edge_data is not None else "rising"
        hysteresis = self.hysteresis_spin.value()

        # Only a structural change (mode / channel / pre-trigger / window / edge
        # / hysteresis) needs to reset the capture state. A pure threshold
        # *value* change -- a spinbox tick or a threshold-line drag -- must
        # preserve the sample history and pre-trigger buffer so detection keeps
        # running smoothly.
        controller = self._controller
        structural_change = (
            mode != controller.mode
            or channel_id != controller.channel_id
            or abs(pre_seconds - controller.pre_seconds) > 1e-9
            or abs(window_sec - controller.window_sec) > 1e-9
            or edge != controller.edge
            or abs(hysteresis - controller.hysteresis) > 1e-9
        )

        controller.configure(
            mode=mode,
            channel_id=channel_id,
            threshold=threshold,
            pre_seconds=pre_seconds,
            window_sec=window_sec,
            edge=edge,
            hysteresis=hysteresis,
            reset_state=structural_change,
            preserve_display_on_reset=True,
        )

    def set_enabled_for_scanning(self, enabled: bool) -> None:
        """Enable/Disable controls based on device connection."""
        # Even if enabled, specific modes might be disabled if no channels exist
        # But broadly, the group box is enabled
        self.trigger_group.setEnabled(enabled)

    def update_channels(self, channels: list[tuple[str, int]]) -> None:
        """
        Update the channel combobox.
        channels: list of (name, id) tuples.
        """
        current_data = self.trigger_channel_combo.currentData()
        self.trigger_channel_combo.blockSignals(True)
        self.trigger_channel_combo.clear()
        
        if not channels:
            self.trigger_channel_combo.addItem("No Channels", None)
            self.trigger_channel_combo.setEnabled(False)
            self._set_trigger_modes_enabled(False)
        else:
            for name, cid in channels:
                self.trigger_channel_combo.addItem(name, cid)
            self.trigger_channel_combo.setEnabled(True)
            self._set_trigger_modes_enabled(True)
            
            # Restore selection if possible
            idx = self.trigger_channel_combo.findData(current_data)
            if idx >= 0:
                self.trigger_channel_combo.setCurrentIndex(idx)
            else:
                self.trigger_channel_combo.setCurrentIndex(0)
                
        self.trigger_channel_combo.blockSignals(False)
        # Trigger an update since channel might have changed (or defaulted)
        self._on_config_changed()

    def _set_trigger_modes_enabled(self, enabled: bool) -> None:
        """Enable/Disable trigger radio buttons (requires valid channel)."""
        self.trigger_mode_single.setEnabled(enabled)
        self.trigger_mode_repeated.setEnabled(enabled)
        # Stream mode is always enabled
        
        # If we disabled current mode, switch to stream
        if not enabled and not self.trigger_mode_stream.isChecked():
            self.trigger_mode_stream.setChecked(True)
            self.trigger_single_button.setEnabled(False)
            # config change will happen via update_channels -> _on_config_changed
