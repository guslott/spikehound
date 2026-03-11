"""ScopeWidget - Dedicated widget for multi-channel waveform visualization.

Extracted from MainWindow to provide a focused component for plot rendering,
channel display management, and trigger visualization.
"""

from __future__ import annotations

import logging
from typing import Optional

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


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
            # Must match trace_renderer.py: y_norm = voltage/(2*span) + offset
            # So: voltage = (y_norm - offset) * (2 * span)
            return [f"{(float(v) - self._offset) * (2.0 * self._span):.3g}" for v in values]
        except Exception as exc:
            logger.debug("VoltageAxis tickStrings failed: %s", exc)
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

    # Updated Signals
    viewClicked = QtCore.Signal(float, QtCore.Qt.MouseButton)  # y_pos, button
    viewDragged = QtCore.Signal(float)  # y_pos
    viewDragFinished = QtCore.Signal()
    thresholdChanged = QtCore.Signal(float)
    popoutRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
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
        except Exception as exc:
            logger.debug("Failed to hide plot buttons: %s", exc)

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
            pen=pg.mkPen((178, 34, 34), width=5),
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

        plot_container = QtWidgets.QWidget(self)
        plot_grid = QtWidgets.QGridLayout(plot_container)
        plot_grid.setContentsMargins(0, 0, 0, 0)
        plot_grid.setSpacing(0)
        plot_grid.addWidget(self.plot_widget, 0, 0)

        overlay = QtWidgets.QWidget(plot_container)
        overlay_layout = QtWidgets.QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(8, 0, 0, 8)
        overlay_layout.setSpacing(0)
        self.popout_button = QtWidgets.QPushButton("", overlay)
        self.popout_button.setObjectName("scopePopoutButton")
        self.popout_button.setFixedSize(18, 18)
        self.popout_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.popout_button.setToolTip("Open the active scope trace in a separate waveform window.")
        self.popout_button.setStyleSheet(
            "#scopePopoutButton {"
            "background-color: rgb(128, 0, 32);"
            "border: 1px solid rgb(84, 0, 21);"
            "border-radius: 2px;"
            "}"
            "#scopePopoutButton:hover {"
            "background-color: rgb(150, 20, 52);"
            "}"
            "#scopePopoutButton:pressed {"
            "background-color: rgb(102, 0, 26);"
            "}"
        )
        self.popout_button.clicked.connect(self.popoutRequested.emit)
        overlay_layout.addWidget(self.popout_button, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        overlay_layout.addStretch(1)
        plot_grid.addWidget(overlay, 0, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)

        layout.addWidget(plot_container)

        # Connect signals
        self.threshold_line.sigPositionChanged.connect(self._on_threshold_moved)
        self._view_box.channelClicked.connect(self.viewClicked)
        self._view_box.channelDragged.connect(self.viewDragged)
        self._view_box.channelDragFinished.connect(self.viewDragFinished)

    def set_threshold(self, value: Optional[float] = None, visible: bool = True) -> None:
        """Set threshold line position and visibility."""
        if value is not None:
            self.threshold_line.setValue(value)
        self.threshold_line.setVisible(visible)

    def set_pretrigger_position(self, time_sec: float, visible: bool = True) -> None:
        """Set pretrigger line position and visibility."""
        self.pretrigger_line.setValue(time_sec)
        self.pretrigger_line.setVisible(visible)
    
    def set_left_axis(self, span: float, offset: float, label: str, units: str = "V", color: Optional[QtGui.QColor] = None) -> None:
        """Update the left axis label and scaling."""
        self._left_axis.set_scaling(span, offset)
        pen = pg.mkPen((0, 0, 139), width=1)
        if color:
             pen = pg.mkPen(color, width=2)
             
        axis = self.plot_widget.getPlotItem().getAxis("left")
        axis.setPen(pen)
        axis.setTextPen(pen)
        axis.setLabel(text=label, units=units)

    def _on_threshold_moved(self) -> None:
        """Emit signal when user moves the threshold line."""
        value = float(self.threshold_line.value())
        self.thresholdChanged.emit(value)
