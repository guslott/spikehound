from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui
import numpy as np
import pyqtgraph as pg


@dataclass
class Curve:
    channel_id: str
    name: str
    color: QtGui.QColor
    gain: float = 1.0
    offset: float = 0.0
    ring: Optional[np.ndarray] = None  # (N,)
    item: Optional[pg.PlotDataItem] = None


class ScopePlot(pg.PlotWidget):
    """
    Fast scrolling plot with:
      • Multiple curves
      • Click-to-select
      • Drag selected curve up/down (changes offset)
      • Left axis shows selected curve label and gain
    """

    curveSelected = QtCore.Signal(str)  # channel_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#E6F0E0")  # match main window
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "Time", units="s")
        self.setLabel("left", "Voltage", units="V")

        self._seconds = 5.0
        self._fs = 44100
        self._N = int(self._seconds * self._fs)
        self._t = np.linspace(-self._seconds, 0.0, self._N, endpoint=False)

        self._curves: Dict[str, Curve] = {}
        self._order: List[str] = []  # draw order == add order
        self._selected: Optional[str] = None

        # Mouse interaction
        self.plotItem.vb.setMouseEnabled(x=False, y=False)  # we manage dragging
        self._drag_active = False
        self._drag_last_y = 0.0

        self.scene().sigMouseClicked.connect(self._on_click)
        self.scene().sigMouseMoved.connect(self._on_move)

        # Keep the x-range fixed like a strip chart
        self.setXRange(-self._seconds, 0.0, padding=0)

    # --- public API ---
    def set_time_base(self, sample_rate: int, seconds: float):
        self._fs = max(1, int(sample_rate))
        self._seconds = max(0.5, float(seconds))
        self._N = int(self._seconds * self._fs)
        self._t = np.linspace(-self._seconds, 0.0, self._N, endpoint=False)
        self.setXRange(-self._seconds, 0.0, padding=0)
        # reinit curve rings
        for c in self._curves.values():
            c.ring = np.zeros(self._N, dtype=float)
            if c.item is not None:
                c.item.setData(self._t, c.ring)

    def add_curve(self, channel_id: str, name: str, color: QtGui.QColor) -> int:
        if channel_id in self._curves:
            return self._order.index(channel_id)
        pen = pg.mkPen(color=color, width=2)
        item = self.plot(pen=pen, name=name)
        curve = Curve(channel_id=channel_id, name=name, color=color, gain=1.0, offset=0.0,
                      ring=np.zeros(self._N, dtype=float), item=item)
        self._curves[channel_id] = curve
        self._order.append(channel_id)
        # draw on top if selected later
        return len(self._order) - 1

    def remove_curve(self, channel_id: str):
        c = self._curves.pop(channel_id, None)
        if c and c.item:
            self.removeItem(c.item)
        if channel_id in self._order:
            self._order.remove(channel_id)
        if self._selected == channel_id:
            self._selected = None
            self.setLabel("left", "Voltage", units="V")

    def clear_all(self):
        for cid in list(self._curves.keys()):
            self.remove_curve(cid)

    def append_samples(self, channel_id: str, y: np.ndarray):
        c = self._curves.get(channel_id)
        if not c or c.ring is None:
            return
        k = len(y)
        if k <= 0:
            return
        c.ring = np.roll(c.ring, -k)
        c.ring[-k:] = y

    def refresh(self):
        for cid in self._order:
            c = self._curves[cid]
            if c.item is None or c.ring is None:
                continue
            c.item.setData(self._t, c.gain * c.ring + c.offset)
        # ensure selected is drawn last (on top)
        if self._selected and self._selected in self._curves:
            c = self._curves[self._selected]
            if c.item:
                self.removeItem(c.item)
                self.addItem(c.item)

    def select_curve(self, channel_id: str):
        if channel_id not in self._curves:
            return
        self._selected = channel_id
        c = self._curves[channel_id]
        self.setLabel("left", f"{c.name}  (V) — Gain ×{c.gain:.2f}")
        # visual emphasis
        for cid, cc in self._curves.items():
            w = 3 if cid == channel_id else 2
            cc.item.setPen(pg.mkPen(color=cc.color, width=w))

    # --- look & feel ---
    def apply_scope_theme(self):
        # Light green bg already set; dashed grid lines
        ax = self.getPlotItem().getAxis("left")
        ax.setPen(pg.mkPen("#64866a"))
        ax = self.getPlotItem().getAxis("bottom")
        ax.setPen(pg.mkPen("#64866a"))
        self.showGrid(x=True, y=True, alpha=0.25)

    # --- mouse interaction ---
    def _on_click(self, ev):
        if not ev or not hasattr(ev, "scenePos"):
            return
        pos = ev.scenePos()
        if not self.plotItem.sceneBoundingRect().contains(pos):
            return
        mouse_pt = self.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_pt.x(), mouse_pt.y()

        # Find nearest curve at this x
        nearest_id, dist = self._nearest_curve(x, y)
        if nearest_id:
            self.select_curve(nearest_id)
            self._drag_active = True
            self._drag_last_y = y
            ev.accept()
        else:
            self._drag_active = False

    def _on_move(self, pos):
        if not self._drag_active or self._selected is None:
            return
        if not self.plotItem.sceneBoundingRect().contains(pos):
            return
        mouse_pt = self.plotItem.vb.mapSceneToView(pos)
        y = mouse_pt.y()
        dy = y - self._drag_last_y
        self._drag_last_y = y
        # Apply vertical offset to selected curve
        c = self._curves.get(self._selected)
        if c:
            c.offset += dy
            self.setLabel("left", f"{c.name}  (V) — Gain ×{c.gain:.2f}")
            # immediate visual feedback
            if c.item and c.ring is not None:
                c.item.setData(self._t, c.gain * c.ring + c.offset)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self._drag_active = False

    def _nearest_curve(self, x: float, y: float) -> tuple[Optional[str], float]:
        """Find the curve whose y(x) is closest to mouse y; return (id, distance)."""
        if not self._order:
            return None, float("inf")
        # translate x into nearest time index
        # t is monotonically increasing from -seconds -> 0
        if self._N <= 1:
            return None, float("inf")
        idx = int((x + self._seconds) / self._seconds * (self._N - 1))
        idx = max(0, min(self._N - 1, idx))
        nearest_id = None
        nearest_dist = float("inf")
        for cid in self._order:
            c = self._curves[cid]
            if c.ring is None:
                continue
            y_i = c.gain * c.ring[idx] + c.offset
            d = abs(y - y_i)
            if d < nearest_dist:
                nearest_dist = d
                nearest_id = cid
        # threshold relative to current view range to avoid accidental picks
        y_rng = self.viewRange()[1]
        thr = 0.02 * max(1e-6, (y_rng[1] - y_rng[0]))
        if nearest_dist <= thr:
            return nearest_id, nearest_dist
        return None, nearest_dist
