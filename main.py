import time
import queue
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from daq.simulated_source import SimulatedPhysiologySource
from daq.soundcard_source import SoundCardSource
from daq.base_source import Chunk, DeviceInfo, ChannelInfo

# ---- Configuration ----
SAMPLE_RATE = 20_000  # Hz (try 48_000 for sound cards)
CHUNK_SIZE = 200      # Frames per chunk
PLOT_DURATION_S = 0.5 # Display window

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
active_channels = []
plot_samples = int(PLOT_DURATION_S * SAMPLE_RATE)
time_axis = np.linspace(0, PLOT_DURATION_S, plot_samples)
buf = None
is_running = threading.Event(); is_running.set()


def _determine_ylim():
    # Sound card audio is typically normalized to [-1, 1]; user wants +-0.5
    if SOURCE_CLASS is SoundCardSource:
        return (-0.5, 0.5)
    # Simulated physiology default range
    return (-1.5, 1.5)


def init_plot(ax, lines):
    global buf
    buf = np.zeros((plot_samples, len(active_channels)))
    ax.set_xlim(0, PLOT_DURATION_S)
    ax.set_ylim(*_determine_ylim())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('SpikeHound – Live Data')
    ax.grid(True)
    for i, ln in enumerate(lines):
        ln.set_data(time_axis, buf[:, i])
    return lines


def update_plot(frame, lines):
    global buf
    chunks = []
    while True:
        try:
            ch: Chunk = sim.data_queue.get_nowait()
            chunks.append(ch)
        except queue.Empty:
            break

    if not chunks:
        return lines

    new_data = np.concatenate([c.data for c in chunks], axis=0)
    n_new = new_data.shape[0]

    if n_new >= plot_samples:
        buf[:, :] = new_data[-plot_samples:, :]
    else:
        buf = np.roll(buf, -n_new, axis=0)
        buf[-n_new:, :] = new_data

    for i, ln in enumerate(lines):
        ln.set_ydata(buf[:, i])
    return lines


def on_close(event):
    if sim and sim.running:
        sim.stop()
    is_running.clear()


def create_source_and_channels():
    """Instantiate the selected source and choose default channels.

    Uses the driver's `list_available_devices()` API to choose a device. If
    `DEVICE_ID` is None, the first device is selected.
    """
    driver = SOURCE_CLASS()
    devices = driver.list_available_devices()
    print("Detected devices:")
    for d in devices:
        print(f"  - {d.id}: {d.name}")
    if not devices:
        raise RuntimeError('No devices found for selected SOURCE_CLASS.')
    selected: DeviceInfo
    if DEVICE_ID is None:
        if len(devices) > 1:
            selected = devices[1]
        else:
            selected = devices[0]
    else:
        selected = next((d for d in devices if str(d.id) == str(DEVICE_ID)), devices[0])

    # Open the chosen device
    driver.open(selected.id)

    # Build channels and configure per driver
    available: list[ChannelInfo] = driver.list_available_channels(selected.id)

    if SOURCE_CLASS is SimulatedPhysiologySource:
        chan_ids = [c.id for c in available]
        driver.configure(sample_rate=SAMPLE_RATE, channels=chan_ids, chunk_size=CHUNK_SIZE, num_units=NUM_UNITS)
        return driver, [c.name for c in available]

    elif SOURCE_CLASS is SoundCardSource:
        if not available:
            raise RuntimeError('No input channels available on the selected audio device.')
        chan_ids = [c.id for c in available[:max(1, NUM_AUDIO_CHANNELS)]]
        driver.configure(sample_rate=SAMPLE_RATE, channels=chan_ids, chunk_size=CHUNK_SIZE)
        return driver, [c.name for c in available[:len(chan_ids)]]

    else:
        raise ValueError('Unsupported SOURCE_CLASS')


def main():
    global sim, active_channels

    sim, active_channels = create_source_and_channels()

    sim.start()

    fig, ax = plt.subplots(figsize=(10, 4))
    lines = [ax.plot([], [], lw=1.0, label=ch)[0] for ch in active_channels]
    ax.legend(loc='upper right')

    init_plot(ax, lines)
    fig.canvas.mpl_connect('close_event', on_close)

    _ = animation.FuncAnimation(
        fig,
        lambda f: update_plot(f, lines),
        interval=33,
        blit=True,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
