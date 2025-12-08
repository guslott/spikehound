from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

if TYPE_CHECKING:
    from shared.types import Event

# Constants moved from analysis_tab.py
CLUSTER_COLORS: list[QtGui.QColor] = [
    QtGui.QColor(34, 139, 34),    # green
    QtGui.QColor(255, 140, 0),    # orange
    QtGui.QColor(148, 0, 211),    # purple
    QtGui.QColor(255, 215, 0),    # gold
    QtGui.QColor(199, 21, 133),   # magenta
]
UNCLASSIFIED_COLOR = QtGui.QColor(220, 0, 0)
WAVEFORM_MEDIAN_COLOR = QtGui.QColor(200, 0, 0)
MAX_VISIBLE_METRIC_EVENTS = 3000
METRIC_TIME_WINDOW_SEC = 60.0
STA_TRACE_PEN = pg.mkPen(90, 90, 90, 110)
SCOPE_BACKGROUND_COLOR = QtGui.QColor(211, 230, 204)

class _MeasureLine:
    """Helper for draggable measurement lines with labels."""

    def __init__(self, plot_item: pg.PlotItem, p1: QtCore.QPointF, p2: QtCore.QPointF, mode: str = "line") -> None:
        self.mode = mode  # "line", "vertical", "horizontal"
        self.roi = pg.LineSegmentROI((p1.x(), p1.y()), (p2.x(), p2.y()), pen=pg.mkPen(QtGui.QColor(0, 100, 200), width=2))
        self.label = pg.TextItem(color=(20, 20, 20))
        self._plot = plot_item
        self._guard = False
        self._endpoints = [
            pg.ScatterPlotItem([p1.x()], [p1.y()], symbol="+", size=12, pen=pg.mkPen("black"), brush=None),
            pg.ScatterPlotItem([p2.x()], [p2.y()], symbol="+", size=12, pen=pg.mkPen("black"), brush=None),
        ]
        plot_item.addItem(self.roi)
        plot_item.addItem(self.label)
        for ep in self._endpoints:
            ep.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            plot_item.addItem(ep)
        self.roi.sigRegionChanged.connect(self._on_changed)
        self._set_handle_positions(pg.Point(p1), pg.Point(p2))
        self._on_changed()

    def remove(self) -> None:
        try:
            self._plot.removeItem(self.roi)
        except Exception:
            pass
        try:
            self._plot.removeItem(self.label)
        except Exception:
            pass
        for ep in self._endpoints:
            try:
                self._plot.removeItem(ep)
            except Exception:
                pass

    def set_points(self, p1: QtCore.QPointF, p2: QtCore.QPointF) -> None:
        self._set_handle_positions(pg.Point(p1), pg.Point(p2))
        self._on_changed()

    def _set_handle_positions(self, p1: pg.Point, p2: pg.Point) -> None:
        handles = getattr(self.roi, "handles", None)
        if not handles or len(handles) < 2:
            return
        h0 = handles[0].get("item")
        h1 = handles[1].get("item")
        if h0 is not None:
            try:
                self.roi.movePoint(h0, p1)
            except Exception:
                pass
        if h1 is not None:
            try:
                self.roi.movePoint(h1, p2)
            except Exception:
                pass
        if self._endpoints and len(self._endpoints) >= 2:
            self._endpoints[0].setData([p1.x()], [p1.y()])
            self._endpoints[1].setData([p2.x()], [p2.y()])

    def _on_changed(self) -> None:
        if self._guard:
            return
        self._guard = True
        try:
            pts = self.roi.getState().get("points") or []
            if len(pts) != 2:
                return
            p1 = pg.Point(pts[0])
            p2 = pg.Point(pts[1])
            if self.mode == "vertical":
                x = (p1.x() + p2.x()) * 0.5
                p1 = pg.Point(x, p1.y())
                p2 = pg.Point(x, p2.y())
                self._set_handle_positions(p1, p2)
            elif self.mode == "horizontal":
                y = (p1.y() + p2.y()) * 0.5
                p1 = pg.Point(p1.x(), y)
                p2 = pg.Point(p2.x(), y)
                self._set_handle_positions(p1, p2)
            dt = float(p2.x() - p1.x())
            dv = float(p2.y() - p1.y())
            text: str
            if self.mode == "vertical":
                text = f"\u0394V = {dv:.4g} V"
            elif self.mode == "horizontal":
                text = f"\u0394t = {dt:.4g} s"
            else:
                text = f"\u0394t = {dt:.4g} s, \u0394V = {dv:.4g} V"
            mid = pg.Point((p1.x() + p2.x()) * 0.5, (p1.y() + p2.y()) * 0.5)
            self.label.setText(text)
            self.label.setPos(mid.x(), mid.y())
        finally:
            self._guard = False


class ClusterRectROI(pg.ROI):
    """Rectangular ROI used for metric clusters."""

    def __init__(self, pos, size, pen=None, **kwargs):
        super().__init__(pos, size, movable=False, rotatable=False, **kwargs)
        self.addScaleHandle((0.0, 0.5), (1.0, 0.5))
        self.addScaleHandle((1.0, 0.5), (0.0, 0.5))
        self.addScaleHandle((0.5, 0.0), (0.5, 1.0))
        self.addScaleHandle((0.5, 1.0), (0.5, 0.0))
        for handle in self.handles:
            item = handle.get("item")
            if item is not None:
                try:
                    item.setSize(10)
                except AttributeError:
                    pass
        if pen is None:
            pen = pg.mkPen(CLUSTER_COLORS[0])
        self.setPen(pen)
        self._dragging_with_shift = False

    def mouseDragEvent(self, ev):
        if ev.isStart():
            self._dragging_with_shift = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
            if self._dragging_with_shift:
                ev.accept()
                return
            return super().mouseDragEvent(ev)
        if self._dragging_with_shift:
            last = ev.lastPos()
            if last is None:
                ev.accept()
                return
            delta = pg.Point(ev.pos()) - pg.Point(last)
            self.translate(delta, snap=False)
            ev.accept()
            if ev.isFinish():
                self._dragging_with_shift = False
            return
        return super().mouseDragEvent(ev)


@dataclass
class MetricCluster:
    id: int
    name: str
    color: QtGui.QColor
    roi: pg.RectROI | None = None


@dataclass
class OverlayPayload:
    """Qt-free representation of a raw spike overlay."""

    event_id: int | None
    times: np.ndarray
    samples: np.ndarray
    last_time: float
    first_index: int | None
    sr: float
    pre_samples: int
    baseline: float
    peak_idx: int
    peak_time: float
    metrics: dict[str, float] | None
    metric_time: float


@dataclass
class StaTask:
    """Data packet describing which events to use for STA processing."""

    events: tuple[Event, ...]
    target_channel_id: int
    channel_index: int | None
    window_ms: float


@dataclass
class AnalysisUpdate:
    """Result bundle produced by preprocessing an analysis batch."""

    overlays: list[OverlayPayload]
    sta_windows: list[np.ndarray] | None = None
    sta_task: StaTask | None = None
    last_event_id: int | None = None
