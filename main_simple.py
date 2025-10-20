"""Minimal SpikeHound harness that proves the pipeline works end-to-end.

This script is intentionally small and heavily commented so that new
contributors (or students in the classroom) can read through it and understand
how the pieces of the architecture fit together:

* A data source (`BaseSource`) feeds raw `Chunk`s into a queue.
* The `PipelineController` wraps the source, the dispatcher, and the shared
  queues that the rest of the application will eventually use.
* The Qt GUI consumes conditioned data from the visualization queue and draws
  it using PyQtGraph while keeping the GUI thread responsive.

Think of this as a live “smoke test” for the architecture. It deliberately
trades bells-and-whistles for clarity.
"""

import sys
from typing import Any, Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from core import ChannelFilterSettings, EndOfStream, FilterSettings, PipelineController
from daq.base_source import DeviceInfo
from daq.simulated_source import SimulatedPhysiologySource
from daq.soundcard_source import SoundCardSource

pg.setConfigOptions(antialias=False)

# ---- Configuration ----
#
# These knobs let you experiment quickly without touching the core
# architecture. Feel free to modify them while you explore the harness.
SAMPLE_RATE = 20000  # Hz (try 48000 for sound cards)
CHUNK_SIZE = 200      # Frames per chunk
PLOT_DURATION_S = 0.4 # Display window

# Simulator control
NUM_UNITS = 6
SIM_LINE_HUM_AMP = 0.5          # Adjust to inject 60 Hz interference (e.g., 0.05)
SIM_LINE_HUM_FREQ = 60.0        # Line interference frequency in Hz

# Sound card control
NUM_AUDIO_CHANNELS = 1  # how many input channels to show by default

# Dispatcher / filter configuration (tweak in-code to explore bandwidth)
FILTER_AC_COUPLE = True         # Enable 1st-order high-pass to remove DC bias
FILTER_AC_CUTOFF_HZ = 1.0       # AC coupling cutoff frequency (Hz)
FILTER_NOTCH_ENABLED = True     # Enable 50/60 Hz notch rejection
FILTER_NOTCH_FREQ_HZ = 60.0     # Notch center frequency (Hz)
FILTER_NOTCH_Q = 35.0           # Notch quality factor (higher => narrower)
FILTER_LOWPASS_HZ = None        # Set to a value (Hz) for Butterworth low-pass, or None
FILTER_LOWPASS_ORDER = 4
FILTER_HIGHPASS_HZ = None       # Set to a value (Hz) for additional Butterworth high-pass
FILTER_HIGHPASS_ORDER = 2

# Change this single line to switch sources:
SOURCE_CLASS = SimulatedPhysiologySource
# SOURCE_CLASS = SoundCardSource
DEVICE_ID = None  # Optional: pick a specific device id from list_available_devices()


# ---- App State ----
controller: PipelineController | None = None
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
    """Ensure the active source and dispatcher are stopped and closed."""
    global controller
    try:
        if controller:
            # `shutdown()` stops the dispatcher, source, and clears queues. We do
            # this here instead of relying on Qt teardown to keep the example
            # deterministic (especially during testing).
            controller.shutdown()
    except Exception as exc:
        print(f"Error while closing source: {exc}")
    finally:
        controller = None


class ScopeWindow(QtWidgets.QMainWindow):
    """Simple oscilloscope-style window driven by dispatcher ticks."""

    def __init__(
        self,
        channel_names: List[str],
        _x_axis: np.ndarray,
        _data_buf: np.ndarray,
        pipeline: PipelineController,
    ) -> None:
        super().__init__()
        self._controller = pipeline
        self._channel_names: List[str] = list(channel_names)
        self._curves: List[pg.PlotCurveItem] = []
        self._chunk_rate: float = 0.0

        self.setWindowTitle("SpikeHound – Live Data")

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Voltage", units="V")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.enableAutoRange(axis="y", enable=False)

        plot_item = self.plot_widget.getPlotItem()
        if hasattr(plot_item, "setClipToView"):
            plot_item.setClipToView(True)
        if hasattr(plot_item, "setDownsampling"):
            plot_item.setDownsampling(mode="peak")

        layout.addWidget(self.plot_widget)
        self._ensure_curves(self._channel_names)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(12)
        self._status_labels: Dict[str, QtWidgets.QLabel] = {}
        for key in ("sr", "chunk", "queues", "drops"):
            label = QtWidgets.QLabel("…", self)
            label.setStyleSheet("color: #333; font-size: 11px;")
            status_row.addWidget(label)
            self._status_labels[key] = label
        status_row.addStretch(1)
        layout.addLayout(status_row)

        self.setCentralWidget(central)

        close_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Close), self)
        close_shortcut.activated.connect(self.close)

        signals = self._controller.dispatcher_signals()
        if signals is not None:
            signals.tick.connect(self._on_dispatcher_tick)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        stop_source()
        super().closeEvent(event)

    def _ensure_curves(self, channel_names: List[str]) -> None:
        plot_item = self.plot_widget.getPlotItem()
        for curve in self._curves:
            plot_item.removeItem(curve)
        self._curves.clear()
        if not channel_names:
            return
        for index, _ in enumerate(channel_names):
            pen = pg.mkPen(color=pg.intColor(index, hues=max(len(channel_names), 1)), width=1.5)
            curve = pg.PlotCurveItem(pen=pen)
            self._curves.append(curve)
            plot_item.addItem(curve)

    @QtCore.Slot(dict)
    def _on_dispatcher_tick(self, payload: dict) -> None:
        channel_names = list(payload.get("channel_names", []))
        samples = payload.get("samples")
        sample_rate = float(payload.get("sample_rate") or 0)
        window_sec = float(payload.get("window_sec") or 0)
        if samples is None or sample_rate <= 0 or window_sec <= 0:
            return

        data = np.asarray(samples)
        if data.ndim != 2 or data.size == 0:
            return

        if channel_names != self._channel_names:
            self._channel_names = channel_names
            self._ensure_curves(self._channel_names)

        if not self._curves:
            return

        num_samples = data.shape[1]
        if num_samples == 0:
            return
        time_axis = np.linspace(0.0, window_sec, num_samples, endpoint=False, dtype=np.float32)

        for idx, curve in enumerate(self._curves):
            if idx < data.shape[0]:
                curve.setData(time_axis, data[idx], skipFiniteCheck=True)

        self.plot_widget.getPlotItem().setXRange(0, window_sec, padding=0)
        self._chunk_rate = sample_rate
        self._update_status(0)

    def _update_status(self, viz_depth: int) -> None:
        """Refresh the labels with queue and throughput diagnostics."""
        stats = self._controller.dispatcher_stats()
        drops = stats.get("dropped", {})
        evicted = stats.get("evicted", {})
        queue_depths = self._controller.queue_depths()

        sr = self._controller.sample_rate or 0.0
        self._status_labels["sr"].setText(f"SR: {sr:,.0f} Hz")
        self._status_labels["chunk"].setText(f"Chunks/s: {self._chunk_rate:5.1f}")

        viz_size, viz_max = queue_depths.get("visualization", (viz_depth, 0))
        analysis_size, analysis_max = queue_depths.get("analysis", (0, 0))
        audio_size, audio_max = queue_depths.get("audio", (0, 0))
        viz_max_text = "∞" if viz_max == 0 else str(viz_max)
        analysis_max_text = "∞" if analysis_max == 0 else str(analysis_max)
        audio_max_text = "∞" if audio_max == 0 else str(audio_max)
        self._status_labels["queues"].setText(
            f"Queues V:{viz_size}/{viz_max_text} A:{analysis_size}/{analysis_max_text} Au:{audio_size}/{audio_max_text}"
        )

        viz_drops = drops.get("visualization", 0)
        log_drops = drops.get("logging", 0)
        viz_evicted = evicted.get("visualization", 0)
        self._status_labels["drops"].setText(f"Drops V:{viz_drops} L:{log_drops} Evict:{viz_evicted}")


def _select_device(source_cls) -> DeviceInfo:
    """Pick a device for the demo, honoring DEVICE_ID if provided."""
    devices = source_cls.list_available_devices()
    print("Detected devices:")
    for d in devices:
        print(f"  - {d.id}: {d.name}")
    if not devices:
        raise RuntimeError("No devices found for selected SOURCE_CLASS.")

    if DEVICE_ID is None:
        return devices[1] if len(devices) > 1 else devices[0]

    for dev in devices:
        if str(dev.id) == str(DEVICE_ID):
            return dev

    raise ValueError(f"Device id {DEVICE_ID!r} not available")


def setup_controller() -> tuple[PipelineController, List[str], float]:
    """Instantiate the controller and configure the chosen data source."""
    default_channel_filters = ChannelFilterSettings(
        ac_couple=FILTER_AC_COUPLE,
        ac_cutoff_hz=FILTER_AC_CUTOFF_HZ,
        notch_enabled=FILTER_NOTCH_ENABLED,
        notch_freq_hz=FILTER_NOTCH_FREQ_HZ,
        notch_q=FILTER_NOTCH_Q,
        lowpass_hz=FILTER_LOWPASS_HZ,
        lowpass_order=FILTER_LOWPASS_ORDER,
        highpass_hz=FILTER_HIGHPASS_HZ,
        highpass_order=FILTER_HIGHPASS_ORDER,
    )

    filter_settings = FilterSettings(default=default_channel_filters)

    pipeline = PipelineController(
        filter_settings=filter_settings,
        visualization_queue_size=128,
        analysis_queue_size=64,
        audio_queue_size=64,
        logging_queue_size=256,
    )

    selected = _select_device(SOURCE_CLASS)
    probe_source = SOURCE_CLASS()
    probe_source.open(selected.id)
    try:
        available = probe_source.list_available_channels(selected.id)
    finally:
        probe_source.close()

    if not available:
        raise RuntimeError("Selected device has no channels")

    chan_ids = [c.id for c in available]
    if SOURCE_CLASS is SoundCardSource:
        chan_ids = chan_ids[: max(1, NUM_AUDIO_CHANNELS)]

    configure_kwargs: Dict[str, Any] = {
        "sample_rate": SAMPLE_RATE,
        "channels": chan_ids,
        "chunk_size": CHUNK_SIZE,
    }
    # Additional options are layered in for the simulator. When we add new
    # sources later (e.g., hardware DAQ), they can provide their own kwargs
    # without touching the GUI code.

    if SOURCE_CLASS is SimulatedPhysiologySource:
        configure_kwargs.update(
            {
                "num_units": NUM_UNITS,
                "line_hum_amp": SIM_LINE_HUM_AMP,
                "line_hum_freq": SIM_LINE_HUM_FREQ,
            }
        )

    actual = pipeline.switch_source(
        SOURCE_CLASS,
        device_id=selected.id,
        configure_kwargs=configure_kwargs,
    )

    channel_names = [ch.name for ch in actual.channels]
    return pipeline, channel_names, float(actual.sample_rate)


def main() -> None:
    """Launch the harness: configure the pipeline, show the window, run Qt."""
    global controller, active_channels, buf, plot_samples, time_axis

    controller, active_channels, actual_rate = setup_controller()

    plot_samples = int(PLOT_DURATION_S * actual_rate)
    time_axis = np.linspace(0, PLOT_DURATION_S, plot_samples, dtype=np.float32)
    buf = np.zeros((plot_samples, len(active_channels)), dtype=np.float32)

    # Start the pipeline before showing the window so the first timer tick has
    # data waiting in the visualization queue.
    controller.start()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ScopeWindow(active_channels, time_axis, buf, controller)
    app.aboutToQuit.connect(stop_source)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
