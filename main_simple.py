"""Minimal SpikeHound demo that wires a single source into a PyQtGraph scope.

The goal is to keep this example approachable: pick a hard-coded device,
configure the acquisition pipeline, and render the live stream with as little
UI chrome as possible. Adjust the constants below to experiment with different
devices or plotting behaviour.
"""

from __future__ import annotations

import sys
from typing import List, Sequence, Type

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from core import PipelineController
from daq.base_source import BaseSource, ChannelInfo, DeviceInfo
from daq.simulated_source import SimulatedPhysiologySource
from daq.soundcard_source import SoundCardSource

pg.setConfigOptions(antialias=False)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 20_000  # Hz
CHUNK_SIZE = 256      # Frames per chunk handed to the dispatcher
WINDOW_DURATION_S = 0.5  # Seconds of data to display
OFFSET_STEP = 0.75        # Vertical spacing between channel traces
CHANNEL_LIMIT = 3         # Plot at most this many channels (None => all)

# Choose the driver and its device here. The simulator ships with the project
# and is a safe default. Swap SOURCE_CLASS / DEVICE_ID to exercise other backends.
SOURCE_CLASS: Type[BaseSource] = SimulatedPhysiologySource
DEVICE_ID: str | None = "sim0"  # Use None to take the first discovered device

# Example alternative:
# SOURCE_CLASS = SoundCardSource
# DEVICE_ID = None
# CHANNEL_LIMIT = 1


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _pick_device(source_cls: Type[BaseSource], device_id: str | None) -> DeviceInfo:
    devices = source_cls.list_available_devices()
    if not devices:
        raise RuntimeError(f"No devices available for {source_cls.__name__}")

    if device_id is None:
        return devices[0]

    for dev in devices:
        if str(dev.id) == str(device_id):
            return dev

    available = ", ".join(str(dev.id) for dev in devices)
    raise ValueError(f"Device id {device_id!r} not found; available: {available}")


def _probe_channels(source_cls: Type[BaseSource], device: DeviceInfo) -> List[ChannelInfo]:
    probe = source_cls()
    probe.open(device.id)
    try:
        channels = probe.list_available_channels(device.id)
    finally:
        probe.close()
    if CHANNEL_LIMIT is not None:
        channels = channels[:CHANNEL_LIMIT]
    if not channels:
        raise RuntimeError("Selected device exposes no channels.")
    return channels


def build_pipeline() -> tuple[PipelineController, List[ChannelInfo], float]:
    """Create the acquisition pipeline and configure the chosen device."""
    controller = PipelineController()

    device = _pick_device(SOURCE_CLASS, DEVICE_ID)
    channels = _probe_channels(SOURCE_CLASS, device)

    configure_kwargs = {
        "sample_rate": SAMPLE_RATE,
        "channels": [ch.id for ch in channels],
        "chunk_size": CHUNK_SIZE,
    }

    actual = controller.switch_source(
        SOURCE_CLASS,
        device_id=device.id,
        configure_kwargs=configure_kwargs,
    )
    controller.start()
    return controller, list(actual.channels), float(actual.sample_rate)


# ---------------------------------------------------------------------------
# Presentation layer
# ---------------------------------------------------------------------------

class ScopeWindow(QtWidgets.QMainWindow):
    """Very small oscilloscope view driven by dispatcher tick payloads."""

    def __init__(self, pipeline: PipelineController, channels: Sequence[ChannelInfo], window_sec: float) -> None:
        super().__init__()
        self._controller = pipeline
        self._offset_step = OFFSET_STEP
        self._channel_ids: List[int] = []
        self._channel_names: List[str] = []
        self._channel_offsets: dict[int, float] = {}
        self._curves: list[pg.PlotCurveItem] = []
        self._current_window = window_sec

        self.setWindowTitle("SpikeHound – Simple Scope")
        self.resize(960, 540)

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("left", "Amplitude", units="V")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setMouseEnabled(x=False, y=False)
        layout.addWidget(self.plot_widget)

        self.setCentralWidget(central)
        self._ensure_curves_for(channels)
        self._apply_window(window_sec)

        signals = self._controller.dispatcher_signals()
        if signals is None:
            raise RuntimeError("Pipeline dispatcher not initialized.")
        signals.tick.connect(self._on_dispatcher_tick)

    def closeEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        try:
            self._controller.shutdown()
        finally:
            super().closeEvent(event)

    # ------------------------------------------------------------------ #

    def _ensure_curves_for(self, channels: Sequence[ChannelInfo]) -> None:
        plot_item = self.plot_widget.getPlotItem()
        for curve in self._curves:
            plot_item.removeItem(curve)

        self._curves = []
        self._channel_ids = [ch.id for ch in channels]
        self._channel_names = [ch.name for ch in channels]
        self._channel_offsets = {}

        for index, channel in enumerate(channels):
            pen = pg.mkPen(color=pg.intColor(index, hues=max(len(channels), 1)), width=1.5)
            curve = pg.PlotCurveItem(pen=pen, name=channel.name)
            plot_item.addItem(curve)
            self._curves.append(curve)
            self._channel_offsets[channel.id] = index * self._offset_step

    def _apply_window(self, window_sec: float) -> None:
        self._current_window = max(window_sec, 1e-3)
        self.plot_widget.getPlotItem().setXRange(0.0, self._current_window, padding=0.0)

    @QtCore.Slot(dict)
    def _on_dispatcher_tick(self, payload: dict) -> None:
        samples = payload.get("samples")
        times = payload.get("times")
        channel_ids = list(payload.get("channel_ids", []))
        channel_names = list(payload.get("channel_names", []))
        status = payload.get("status", {})
        window_sec = float(status.get("window_sec", self._current_window))

        data = np.asarray(samples) if samples is not None else np.zeros((0, 0), dtype=np.float32)
        time_axis = np.asarray(times) if times is not None else np.linspace(0.0, window_sec, data.shape[1], dtype=np.float32)

        if channel_ids and (channel_ids != self._channel_ids or channel_names != self._channel_names):
            inferred = [ChannelInfo(id=cid, name=name) for cid, name in zip(channel_ids, channel_names)]
            self._ensure_curves_for(inferred)

        if data.ndim != 2 or data.size == 0 or not self._curves:
            for curve in self._curves:
                curve.clear()
            return

        channel_count = min(data.shape[0], len(self._curves))
        for idx in range(channel_count):
            cid = channel_ids[idx] if idx < len(channel_ids) else idx
            offset = self._channel_offsets.get(cid, idx * self._offset_step)
            self._curves[idx].setData(time_axis, data[idx] + offset, skipFiniteCheck=True)

        max_offset = max(self._channel_offsets.values(), default=0.0)
        self.plot_widget.getPlotItem().setYRange(
            -self._offset_step,
            max_offset + self._offset_step,
            padding=0.0,
        )
        self._apply_window(window_sec)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    controller, channels, sample_rate = build_pipeline()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SpikeHound Simple Demo")

    window = ScopeWindow(controller, channels, WINDOW_DURATION_S)
    window.setWindowTitle(f"SpikeHound – {len(channels)} channels @ {sample_rate:,.0f} Hz")

    app.aboutToQuit.connect(controller.shutdown)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
