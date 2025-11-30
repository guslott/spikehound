"""ScopeWidget - Dedicated widget for multi-channel waveform visualization.

Extracted from MainWindow to provide a focused component for plot rendering,
channel display management, and trigger visualization.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class ChannelConfig:
    """Configuration for a single channel's display properties."""
    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(0, 0, 139))
    display_enabled: bool = True
    vertical_span_v: float = 1.0
    screen_offset: float = 0.5
    channel_name: str = ""


class VoltageAxis(pg.AxisItem):
    """Axis item that maps normalized 0-1 coordinates to volts for display."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._span = 1.0
        self._offset = 0.5

    def set_scaling(self, span: float, offset: float) -> None:
        self._span = max(float(span), 1e-9)
        self._offset = float(offset)

    def tickStrings(self, values, scale, spacing):
        try:
            return [f"{(float(v) - self._offset) * self._span:.3g}" for v in values]
        except Exception:
            return super().tickStrings(values, scale, spacing)


class ChannelViewBox(pg.ViewBox):
    """Custom ViewBox to expose click & drag events for channel selection."""

    channelClicked = QtCore.Signal(float, QtCore.Qt.MouseButton)
    channelDragged = QtCore.Signal(float)
    channelDragFinished = QtCore.Signal()

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("enableMenu", False)
        super().__init__(*args, **kwargs)
        self._dragging = False
        self._drag_button: Optional[QtCore.Qt.MouseButton] = None

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


class ScopeWidget(QtWidgets.QWidget):
    """Multi-channel oscilloscope visualization widget."""

    # Signals
    channelClicked = QtCore.Signal(int)  # channel_id
    channelDragged = QtCore.Signal(int, float)  # channel_id, new_offset
    channelDragFinished = QtCore.Signal()
    thresholdChanged = QtCore.Signal(float)
    pretriggerChanged = QtCore.Signal(float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        
        # Channel state
        self._channel_configs: Dict[int, ChannelConfig] = {}
        self._curves: Dict[int, pg.PlotCurveItem] = {}
        self._channel_data: Dict[int, np.ndarray] = {}
        self._active_channel_id: Optional[int] = None
        self._drag_channel_id: Optional[int] = None
        
        # Time axis
        self._last_times: np.ndarray = np.zeros(0, dtype=np.float32)
        
        # Build UI
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the plot widget and controls."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create plot widget with custom viewbox and axis
        self._view_box = ChannelViewBox()
        self._left_axis = VoltageAxis("left")
        self.plot_widget = pg.PlotWidget(
            viewBox=self._view_box,
            enableMenu=False,
            axisItems={"left": self._left_axis}
        )

        # Configure plot appearance
        try:
            self.plot_widget.hideButtons()
        except Exception:
            pass

        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setBackground(QtGui.QColor(211, 230, 204))
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Amplitude", units="V")

        plot_item = self.plot_widget.getPlotItem()
        plot_item.getAxis("left").setPen(pg.mkPen((0, 0, 139)))
        plot_item.getAxis("bottom").setPen(pg.mkPen((0, 0, 139)))
        plot_item.showGrid(x=True, y=True, alpha=0.4)
        plot_item.vb.setBorder(pg.mkPen((0, 0, 139)))

        # Add threshold and pretrigger lines
        self.threshold_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen((178, 34, 34), width=3),
            movable=True
        )
        self.threshold_line.setVisible(False)
        self.plot_widget.addItem(self.threshold_line)
        try:
            self.threshold_line.setZValue(100)
        except AttributeError:
            pass

        self.pretrigger_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen((0, 0, 139), style=QtCore.Qt.DashLine),
            movable=False
        )
        self.pretrigger_line.setVisible(False)
        self.plot_widget.addItem(self.pretrigger_line)

        layout.addWidget(self.plot_widget)

        # Connect signals
        self.threshold_line.sigPositionChanged.connect(self._on_threshold_moved)
        self._view_box.channelClicked.connect(self._on_channel_clicked)
        self._view_box.channelDragged.connect(self._on_channel_dragged)
        self._view_box.channelDragFinished.connect(self._on_drag_finished)

    # Public API for channel management

    def add_channel(self, channel_id: int, config: ChannelConfig) -> None:
        """Add a new channel to the display."""
        self._channel_configs[channel_id] = config
        
        # Create curve if it doesn't exist
        if channel_id not in self._curves:
            curve = pg.PlotCurveItem()
            try:
                curve.setDownsampling(ds=True, auto=True, method="peak")
            except Exception:
                pass
            self.plot_widget.getPlotItem().addItem(curve)
            self._curves[channel_id] = curve
        
        self._update_channel_style(channel_id)

    def remove_channel(self, channel_id: int) -> None:
        """Remove a channel from the display."""
        if channel_id in self._curves:
            plot_item = self.plot_widget.getPlotItem()
            plot_item.removeItem(self._curves[channel_id])
            del self._curves[channel_id]
        
        self._channel_configs.pop(channel_id, None)
        self._channel_data.pop(channel_id, None)

    def set_channel_config(self, channel_id: int, config: ChannelConfig) -> None:
        """Update configuration for an existing channel."""
        if channel_id not in self._channel_configs:
            return
        
        self._channel_configs[channel_id] = config
        self._update_channel_style(channel_id)
        self._update_channel_display(channel_id)

    def set_channel_data(self, channel_id: int, times: np.ndarray, data: np.ndarray) -> None:
        """Update the data for a specific channel."""
        if channel_id not in self._channel_configs:
            return
        
        config = self._channel_configs[channel_id]
        if not config.display_enabled:
            return
        
        # Transform to screen coordinates
        transformed = self._transform_to_screen(data, config.vertical_span_v, config.screen_offset)
        
        # Store and update curve
        self._channel_data[channel_id] = transformed
        self._last_times = times
        
        curve = self._curves.get(channel_id)
        if curve is not None:
            curve.setData(times, transformed, skipFiniteCheck=True)

    def update_all_channels(self, times: np.ndarray, data_dict: Dict[int, np.ndarray]) -> None:
        """Update multiple channels at once with new time/data arrays."""
        self._last_times = times
        
        for channel_id, raw_data in data_dict.items():
            if channel_id not in self._channel_configs:
                continue
            
            config = self._channel_configs[channel_id]
            if not config.display_enabled:
                if channel_id in self._curves:
                    self._curves[channel_id].clear()
                continue
            
            transformed = self._transform_to_screen(raw_data, config.vertical_span_v, config.screen_offset)
            self._channel_data[channel_id] = transformed
            
            curve = self._curves.get(channel_id)
            if curve is not None:
                curve.setData(times, transformed, skipFiniteCheck=True)

    def set_active_channel(self, channel_id: Optional[int]) -> None:
        """Set which channel is visually highlighted as active."""
        self._active_channel_id = channel_id
        self._apply_active_channel_style()
        self._update_axis_label()

    def set_window_duration(self, seconds: float) -> None:
        """Set the time window visible on the X axis."""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setXRange(0.0, max(seconds, 0.001), padding=0.0)

    def set_threshold(self, value: Optional[float] = None, visible: bool = True) -> None:
        """Set threshold line position and visibility."""
        if value is not None:
            self.threshold_line.setValue(value)
        self.threshold_line.setVisible(visible)

    def set_pretrigger_position(self, time_sec: float, visible: bool = True) -> None:
        """Set pretrigger line position and visibility."""
        self.pretrigger_line.setValue(time_sec)
        self.pretrigger_line.setVisible(visible)

    def clear_all_channels(self) -> None:
        """Remove all channels and clear the plot."""
        plot_item = self.plot_widget.getPlotItem()
        for curve in self._curves.values():
            plot_item.removeItem(curve)
        
        self._curves.clear()
        self._channel_configs.clear()
        self._channel_data.clear()
        self._active_channel_id = None
        self._last_times = np.zeros(0, dtype=np.float32)

    # Internal helper methods

    def _transform_to_screen(self, raw_data: np.ndarray, span_v: float, offset_pct: float) -> np.ndarray:
        """Transform raw voltage data to normalized 0-1 screen coordinates."""
        span = max(float(span_v), 1e-9)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.asarray(raw_data, dtype=np.float32) / span + float(offset_pct)
        return np.nan_to_num(result, nan=offset_pct, posinf=offset_pct, neginf=offset_pct)

    def _update_channel_style(self, channel_id: int) -> None:
        """Update the pen style for a channel based on active state."""
        config = self._channel_configs.get(channel_id)
        curve = self._curves.get(channel_id)
        
        if config is None or curve is None:
            return
        
        is_active = (channel_id == self._active_channel_id)
        pen = pg.mkPen(config.color, width=3.0 if is_active else 1.6)
        curve.setPen(pen)
        curve.setZValue(1.0 if is_active else 0.0)
        
        try:
            curve.setOpacity(1.0 if is_active else 0.6)
        except AttributeError:
            pass

    def _apply_active_channel_style(self) -> None:
        """Update all channel styles based on current active channel."""
        for channel_id in self._curves.keys():
            self._update_channel_style(channel_id)

    def _update_channel_display(self, channel_id: int) -> None:
        """Re-render a channel with current config and cached data."""
        config = self._channel_configs.get(channel_id)
        curve = self._curves.get(channel_id)
        raw_data = self._channel_data.get(channel_id)
        
        if curve is None or config is None:
            return
        
        if not config.display_enabled:
            curve.clear()
            return
        
        if raw_data is None or raw_data.size == 0 or self._last_times.size == 0:
            return
        
        # Re-transform and update
        transformed = self._transform_to_screen(raw_data, config.vertical_span_v, config.screen_offset)
        curve.setData(self._last_times, transformed, skipFiniteCheck=True)

    def _update_axis_label(self) -> None:
        """Update the left axis to show the active channel's voltage range."""
        if self._active_channel_id is not None:
            config = self._channel_configs.get(self._active_channel_id)
            if config is not None:
                try:
                    self._left_axis.set_scaling(config.vertical_span_v, config.screen_offset)
                except AttributeError:
                    pass
                
                # Set axis label and color
                name = config.channel_name or f"Ch {self._active_channel_id}"
                axis_color = QtGui.QColor(config.color)
                rgb = axis_color.getRgb()[:3]
                pen = pg.mkPen(rgb, width=2)
                
                axis = self.plot_widget.getPlotItem().getAxis("left")
                axis.setPen(pen)
                axis.setTextPen(pen)
                axis.setLabel(
                    text=f"{name} Amplitude (±{config.vertical_span_v:.3g} V)",
                    units="V",
                )
                return
        
        # Default axis styling
        pen = pg.mkPen((0, 0, 139), width=1)
        axis = self.plot_widget.getPlotItem().getAxis("left")
        axis.setPen(pen)
        axis.setTextPen(pen)
        axis.setLabel(text="Amplitude", units="V")

    def _update_plot_y_range(self) -> None:
        """Fix the normalized viewport to [0.0, 1.0]."""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setYRange(0.0, 1.0, padding=0.0)

    # Signal handlers

    def _on_threshold_moved(self) -> None:
        """Emit signal when user moves the threshold line."""
        value = float(self.threshold_line.value())
        self.thresholdChanged.emit(value)

    def _on_channel_clicked(self, y: float, button: QtCore.Qt.MouseButton) -> None:
        """Handle click on a channel trace."""
        if button != QtCore.Qt.MouseButton.LeftButton:
            return
        
        # Find nearest channel to click position
        channel_id = self._nearest_channel_at_y(y)
        if channel_id is not None:
            self._drag_channel_id = channel_id
            self.channelClicked.emit(channel_id)

    def _on_channel_dragged(self, y: float) -> None:
        """Handle dragging a channel trace."""
        if self._drag_channel_id is None:
            return
        
        y_clamped = max(0.0, min(1.0, float(y)))
        # Snap to center if within 5% of mid
        if abs(y_clamped - 0.5) <= 0.05:
            y_clamped = 0.5
        
        self.channelDragged.emit(self._drag_channel_id, y_clamped)

    def _on_drag_finished(self) -> None:
        """Handle end of drag operation."""
        self._drag_channel_id = None
        self.channelDragFinished.emit()

    def _nearest_channel_at_y(self, y: float) -> Optional[int]:
        """Return the channel whose configured offset is closest to the given y view coordinate."""
        candidates: list[tuple[float, int]] = []
        for channel_id, config in self._channel_configs.items():
            center = config.screen_offset
            candidates.append((abs(float(y) - center), channel_id))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda item: item[0])
        distance, channel_id = candidates[0]
        
        if distance > 0.05:
            return None
        
        return channel_id
