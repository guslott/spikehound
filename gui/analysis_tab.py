from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


class AnalysisTab(QtWidgets.QWidget):
    """Simple rolling waveform view for time-domain analysis."""

    def __init__(
        self,
        channel_name: str,
        sample_rate: float,
        *,
        window_seconds: float = 10.0,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.channel_name = channel_name
        self.sample_rate = float(sample_rate)
        self.window_seconds = max(window_seconds, 0.1)
        self._plot_interval_ms = 30.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.plot_widget = pg.PlotWidget(enableMenu=False)
        self.plot_widget.setBackground(pg.mkColor(230, 230, 235))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Amplitude", units="V")
        layout.addWidget(self.plot_widget)

        self._curve = self.plot_widget.plot(pen=pg.mkPen((30, 144, 255), width=2))

        capacity = int(round(self.window_seconds * self.sample_rate))
        capacity = max(capacity, 1)
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._time_axis = np.linspace(
            -self.window_seconds,
            0.0,
            capacity,
            endpoint=False,
            dtype=np.float32,
        )
        self._write_idx = 0
        self._filled = 0

        self._pending: Deque[np.ndarray] = deque()
        self._pending_samples = 0

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(self._plot_interval_ms))
        self._timer.timeout.connect(self._refresh_plot)
        self._timer.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_samples(self, samples: np.ndarray) -> None:
        """Append new mono samples to the rolling buffer."""
        if samples is None:
            return
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 2:
            # Expect (channels, frames); take first row.
            arr = arr[0]
        if arr.ndim != 1 or arr.size == 0:
            return
        self._pending.append(arr.copy())
        self._pending_samples += arr.size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_pending(self) -> None:
        if not self._pending:
            return
        while self._pending:
            chunk = self._pending.popleft()
            self._write_samples(chunk)
        self._pending_samples = 0

    def _write_samples(self, samples: np.ndarray) -> None:
        count = int(samples.size)
        if count <= 0:
            return
        capacity = self._buffer.size
        if count >= capacity:
            self._buffer[:] = samples[-capacity:]
            self._write_idx = 0
            self._filled = capacity
            return
        next_idx = self._write_idx + count
        if next_idx <= capacity:
            self._buffer[self._write_idx:next_idx] = samples
        else:
            first = capacity - self._write_idx
            self._buffer[self._write_idx:] = samples[:first]
            self._buffer[: count - first] = samples[first:]
        self._write_idx = next_idx % capacity
        self._filled = min(capacity, self._filled + count)

    def _refresh_plot(self) -> None:
        self._drain_pending()
        size = self._filled
        if size <= 0:
            self._curve.clear()
            return
        capacity = self._buffer.size
        if size == capacity:
            if self._write_idx == 0:
                data = self._buffer
            else:
                data = np.concatenate(
                    (
                        self._buffer[self._write_idx :],
                        self._buffer[: self._write_idx],
                    )
                )
            time_axis = self._time_axis
        else:
            start = (self._write_idx - size) % capacity
            if start + size <= capacity:
                data = self._buffer[start : start + size]
            else:
                first = capacity - start
                data = np.concatenate((self._buffer[start:], self._buffer[: size - first]))
            time_axis = np.linspace(
                -size / self.sample_rate,
                0.0,
                size,
                endpoint=False,
                dtype=np.float32,
            )
        self._curve.setData(time_axis, data, skipFiniteCheck=True)
        if time_axis.size:
            self.plot_widget.setXRange(time_axis[0], time_axis[-1], padding=0.0)
