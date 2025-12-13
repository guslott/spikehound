from dataclasses import replace
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .types import ChannelConfig


class ChannelDetailPanel(QtWidgets.QWidget):
    """Per-channel configuration widget with a 3-column layout."""

    configChanged = QtCore.Signal(ChannelConfig)
    analysisRequested = QtCore.Signal()

    def __init__(self, channel_id: int, channel_name: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._channel_id = channel_id
        self._config = ChannelConfig(channel_name=channel_name)
        self._block_updates = False

        self._build_ui()
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
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # --- Column 1: Visuals ---
        col1 = QtWidgets.QVBoxLayout()
        col1.setSpacing(6)
        
        # Color & Display
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(QtWidgets.QLabel("Color"))
        self.color_btn = QtWidgets.QPushButton()
        self.color_btn.setFixedWidth(48)
        self.color_btn.clicked.connect(self._choose_color)
        color_row.addWidget(self.color_btn)
        self.display_check = QtWidgets.QCheckBox("Display")
        self.display_check.setChecked(True)
        color_row.addWidget(self.display_check)
        color_row.addStretch(1)
        col1.addLayout(color_row)

        # Vertical Range
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("Range (±V)"))
        self.range_combo = QtWidgets.QComboBox()
        self.range_combo.setMinimumWidth(150)
        self.range_combo.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        for value in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
            self.range_combo.addItem(f"{value:.1f}", value)
        range_row.addWidget(self.range_combo)
        range_row.addStretch(1)
        col1.addLayout(range_row)

        # Vertical Offset
        offset_row = QtWidgets.QHBoxLayout()
        offset_row.addWidget(QtWidgets.QLabel("Offset"))
        self.offset_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.offset_slider.setRange(0, 100)
        self.offset_slider.setSingleStep(1)
        self.offset_slider.setPageStep(5)
        self.offset_slider.setValue(50)
        offset_row.addWidget(self.offset_slider)
        col1.addLayout(offset_row)
        
        col1.addStretch(1)
        layout.addLayout(col1, 1)

        # --- Column 2: Filters ---
        col2 = QtWidgets.QVBoxLayout()
        col2.setSpacing(2)

        disabled_spin_style = """
            QDoubleSpinBox:disabled {
                background-color: #e0e0e0;
                color: #808080;
                border: 1px solid #c0c0c0;
            }
        """

        # Notch
        notch_row = QtWidgets.QHBoxLayout()
        self.notch_check = QtWidgets.QCheckBox("Notch")
        notch_row.addWidget(self.notch_check)
        self.notch_spin = QtWidgets.QDoubleSpinBox()
        self.notch_spin.setRange(1.0, 1000.0)
        self.notch_spin.setValue(60.0)
        self.notch_spin.setDecimals(0)
        self.notch_spin.setSuffix(" Hz")
        self.notch_spin.setStyleSheet(disabled_spin_style)
        notch_row.addWidget(self.notch_spin)
        col2.addLayout(notch_row)

        # High-pass
        hp_row = QtWidgets.QHBoxLayout()
        self.highpass_check = QtWidgets.QCheckBox("High-pass")
        hp_row.addWidget(self.highpass_check)
        self.highpass_spin = QtWidgets.QDoubleSpinBox()
        self.highpass_spin.setRange(0.1, 10000.0)
        self.highpass_spin.setValue(10.0)
        self.highpass_spin.setDecimals(0)
        self.highpass_spin.setSuffix(" Hz")
        self.highpass_spin.setStyleSheet(disabled_spin_style)
        hp_row.addWidget(self.highpass_spin)
        col2.addLayout(hp_row)

        # Low-pass
        lp_row = QtWidgets.QHBoxLayout()
        self.lowpass_check = QtWidgets.QCheckBox("Low-pass")
        lp_row.addWidget(self.lowpass_check)
        self.lowpass_spin = QtWidgets.QDoubleSpinBox()
        self.lowpass_spin.setRange(1.0, 50000.0)
        self.lowpass_spin.setValue(1000.0)
        self.lowpass_spin.setDecimals(0)
        self.lowpass_spin.setSuffix(" Hz")
        self.lowpass_spin.setStyleSheet(disabled_spin_style)
        lp_row.addWidget(self.lowpass_spin)
        col2.addLayout(lp_row)

        col2.addStretch(1)
        layout.addLayout(col2, 1)

        # --- Column 3: Tools ---
        col3 = QtWidgets.QVBoxLayout()
        col3.setSpacing(6)

        self.listen_btn = QtWidgets.QPushButton("Listen")
        self._apply_toggle_style(self.listen_btn)
        col3.addWidget(self.listen_btn)

        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        col3.addWidget(self.analyze_btn)

        self.spectrogram_btn = QtWidgets.QPushButton("Spectrogram")
        self._apply_toggle_style(self.spectrogram_btn)
        col3.addWidget(self.spectrogram_btn)

        col3.addStretch(1)
        layout.addLayout(col3, 1)

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
        # Spectrogram button is not connected yet as per requirements

    def set_config(self, config: ChannelConfig) -> None:
        # Backwards compatibility
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
        self.notch_spin.setValue(config.notch_freq_hz)
        self.notch_spin.setEnabled(config.notch_enabled)
        self.highpass_check.setChecked(config.highpass_enabled)
        self.highpass_spin.setValue(config.highpass_hz)
        self.highpass_spin.setEnabled(config.highpass_enabled)
        self.lowpass_check.setChecked(config.lowpass_enabled)
        self.lowpass_spin.setValue(config.lowpass_hz)
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
        self._config.notch_freq_hz = float(self.notch_spin.value())
        self._config.highpass_enabled = self.highpass_check.isChecked()
        self._config.highpass_hz = float(self.highpass_spin.value())
        self._config.lowpass_enabled = self.lowpass_check.isChecked()
        self._config.lowpass_hz = float(self.lowpass_spin.value())
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


class ChannelControlsWidget(QtWidgets.QGroupBox):
    """
    Combined widget for active channel selection and channel configuration.
    
    Layout:
    - Top: Active Channel Dropdown
    - Bottom: Stacked widget containing ChannelDetailPanels
    """
    
    activeChannelSelected = QtCore.Signal(int)  # index
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Channel Controls:", parent)
        self._panels = {}  # channel_id -> ChannelDetailPanel
        self._build_ui()
        
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(6)
        
        # Active Channel Dropdown - Manually positioned in resizeEvent
        self.active_combo = QtWidgets.QComboBox(self)
        self.active_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.active_combo.setFixedHeight(20) # Match typical title bar height
        self.active_combo.currentIndexChanged.connect(self._on_active_index_changed)
        # layout.addWidget(self.active_combo) # Removed from layout
        
        # Stacked Widget for Channel Panels
        self.stack = QtWidgets.QStackedWidget()
        
        # Placeholder
        placeholder = QtWidgets.QWidget()
        ph_layout = QtWidgets.QVBoxLayout(placeholder)
        ph_label = QtWidgets.QLabel("Select an active channel to configure.")
        ph_label.setAlignment(QtCore.Qt.AlignCenter)
        ph_label.setStyleSheet("color: rgb(60,60,60); font-style: italic;")
        ph_layout.addWidget(ph_label)
        self.stack.addWidget(placeholder)
        
        layout.addWidget(self.stack)
        
    def _on_active_index_changed(self, index: int) -> None:
        self.activeChannelSelected.emit(index)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Position the combo box next to the title
        # "Channel Controls:" text width
        fm = self.fontMetrics()
        title_width = fm.horizontalAdvance("Channel Controls:")
        
        # Position: x = title_width + padding, y = 0 (fix clipping)
        x_pos = title_width + 35 # Increased padding to avoid text overlap
        y_pos = 0 
        
        # Width: Reduced fixed width to ensure border shows on the right
        combo_width = 160
        # Ensure it doesn't go beyond the widget width
        max_width = self.width() - x_pos - 10
        final_width = min(combo_width, max_width)
        
        if final_width > 0:
            self.active_combo.setGeometry(x_pos, y_pos, final_width, 22)
            self.active_combo.raise_()
        
    def add_panel(self, channel_id: int, panel: ChannelDetailPanel) -> None:
        self._panels[channel_id] = panel
        self.stack.addWidget(panel)
        
    def remove_panel(self, channel_id: int) -> None:
        if channel_id in self._panels:
            panel = self._panels.pop(channel_id)
            self.stack.removeWidget(panel)
            panel.deleteLater()
            
    def show_panel(self, channel_id: Optional[int]) -> None:
        if channel_id is None or channel_id not in self._panels:
            self.stack.setCurrentIndex(0)  # Placeholder
        else:
            self.stack.setCurrentWidget(self._panels[channel_id])
            
    def clear_panels(self) -> None:
        for cid in list(self._panels.keys()):
            self.remove_panel(cid)
