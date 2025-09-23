import queue
import sys
from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from daq.simulated_source import SimulatedPhysiologySource
from daq.soundcard_source import SoundCardSource
from daq.base_source import ChannelInfo

pg.setConfigOptions(antialias=False)

# ---- Configuration ----
SAMPLE_RATE = 20000  # Hz (try 48000 for sound cards)
CHUNK_SIZE = 200      # Frames per chunk
PLOT_DURATION_S = 0.4 # Display window

# Simulator control
NUM_UNITS = 6

# Sound card control
NUM_AUDIO_CHANNELS = 1  # how many input channels to show by default

# Change this single line to switch sources:
SOURCE_CLASS = SimulatedPhysiologySource
# SOURCE_CLASS = SoundCardSource
DEVICE_ID = None  # Optional: pick a specific device id from list_available_devices()


# ---- App State ----
sim = None
active_channels: List[str] = []
plot_samples = int(PLOT_DURATION_S * SAMPLE_RATE)
time_axis = np.linspace(0, PLOT_DURATION_S, plot_samples, dtype=np.float32)
buf = None


def _determine_ylim():
    """Pick a reasonable default vertical range based on the driver."""
    if SOURCE_CLASS is SoundCardSource:
        return (-0.5, 0.5)
    return (-1.5, 1.5)


def stop_source() -> None:
    """Ensure the active source is stopped and closed."""
    global sim
    try:
        if sim:
            if getattr(sim, "running", False):
                sim.stop()
            sim.close()
    except Exception as exc:
        print(f"Error while closing source: {exc}")
    finally:
        sim = None


class ScopeWindow(QtWidgets.QMainWindow):
    """Qt window rendering live data via pyqtgraph."""

    def __init__(self, source, channel_names: List[str], x_axis: np.ndarray, data_buf: np.ndarray) -> None:
        super().__init__()
        self._source = source
        self._channel_names = channel_names
        self._time_axis = x_axis
        self._buf = data_buf
        self._curves = []

        self.setWindowTitle("SpikeHound – Live Data")

        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("left", "Voltage", units="V")
        plot.setLabel("bottom", "Time", units="s")
        plot.setXRange(0.0, PLOT_DURATION_S, padding=0.0)
        plot.setYRange(*_determine_ylim())
        plot.setMouseEnabled(x=False, y=False)
        plot.enableAutoRange(axis="xy", enable=False)

        plot_item = plot.getPlotItem()
        if hasattr(plot_item, "setClipToView"):
            plot_item.setClipToView(True)
        if hasattr(plot_item, "setDownsampling"):
            plot_item.setDownsampling(mode="peak")

        self.setCentralWidget(plot)
        self._plot = plot

        for index, name in enumerate(channel_names):
            pen = pg.mkPen(color=pg.intColor(index, hues=max(len(channel_names), 1)), width=1.5)
            curve = plot_item.plot(
                self._time_axis,
                self._buf[:, index],
                pen=pen,
            )
            self._curves.append(curve)

        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.setInterval(33)  # ~30 Hz refresh
        self._timer.timeout.connect(self._update_plot)
        self._timer.start()

        close_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Close), self)
        close_shortcut.activated.connect(self.close)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        stop_source()
        super().closeEvent(event)

    def _update_plot(self) -> None:
        if not self._curves:
            return

        chunk_arrays: List[np.ndarray] = []
        pending = getattr(self._source.data_queue, "qsize", lambda: 0)()
        if pending == 0:
            return

        for _ in range(pending):
            try:
                chunk = self._source.data_queue.get_nowait()
            except queue.Empty:
                break
            else:
                data = np.asarray(chunk.data)
                if data.ndim == 2 and data.size:
                    chunk_arrays.append(data)

        if not chunk_arrays:
            return

        new_data = chunk_arrays[0] if len(chunk_arrays) == 1 else np.concatenate(chunk_arrays, axis=0)

        rows, cols = new_data.shape
        buf_rows, buf_cols = self._buf.shape

        if cols != buf_cols:
            if cols > buf_cols:
                new_data = new_data[:, :buf_cols]
                cols = buf_cols
            else:
                padded = np.zeros((rows, buf_cols), dtype=self._buf.dtype)
                padded[:, :cols] = new_data
                new_data = padded
                cols = buf_cols

        if rows >= buf_rows:
            self._buf[:, :] = new_data[-buf_rows:, :]
        else:
            self._buf[:-rows, :] = self._buf[rows:, :]
            self._buf[-rows:, :] = new_data

        for index, curve in enumerate(self._curves):
            curve.setData(
                self._time_axis,
                self._buf[:, index],
                skipFiniteCheck=True,
            )


def create_source_and_channels():
    """Instantiate the selected source and choose default channels."""
    devices = SOURCE_CLASS.list_available_devices()
    print("Detected devices:")
    for d in devices:
        print(f"  - {d.id}: {d.name}")
    if not devices:
        raise RuntimeError("No devices found for selected SOURCE_CLASS.")

    if DEVICE_ID is None:
        selected = devices[1] if len(devices) > 1 else devices[0]
    else:
        selected = next((d for d in devices if str(d.id) == str(DEVICE_ID)), devices[0])

    driver = SOURCE_CLASS()
    driver.open(selected.id)
    available: List[ChannelInfo] = driver.list_available_channels(selected.id)

    if SOURCE_CLASS is SimulatedPhysiologySource:
        chan_ids = [c.id for c in available]
        driver.configure(
            sample_rate=SAMPLE_RATE,
            channels=chan_ids,
            chunk_size=CHUNK_SIZE,
            num_units=NUM_UNITS,
        )
        return driver, [c.name for c in available]

    if SOURCE_CLASS is SoundCardSource:
        if not available:
            raise RuntimeError("No input channels available on the selected audio device.")
        count = max(1, NUM_AUDIO_CHANNELS)
        chan_ids = [c.id for c in available[:count]]
        driver.configure(sample_rate=SAMPLE_RATE, channels=chan_ids, chunk_size=CHUNK_SIZE)
        return driver, [c.name for c in available[:len(chan_ids)]]

    raise ValueError("Unsupported SOURCE_CLASS")


def main() -> None:
    global sim, active_channels, buf, plot_samples, time_axis

    sim, active_channels = create_source_and_channels()

    actual_rate = getattr(getattr(sim, "config", None), "sample_rate", SAMPLE_RATE)
    plot_samples = int(PLOT_DURATION_S * actual_rate)
    time_axis = np.linspace(0, PLOT_DURATION_S, plot_samples, dtype=np.float32)
    buf = np.zeros((plot_samples, len(active_channels)), dtype=np.float32)

    sim.start()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ScopeWindow(sim, active_channels, time_axis, buf)
    app.aboutToQuit.connect(stop_source)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
