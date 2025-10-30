import threading
import queue
import time

from audio.player import AudioPlayer, AudioConfig

# Adjust this import path if your sim source lives elsewhere:
from daq.simulated_source import SimulatedPhysiologySource

IN_SAMPLE_RATE = 20_000
CHUNK_SIZE = 200
SELECTED_CHANNEL = 1  # try 0 or 1 if you don't hear anything
LINE_HUM_FREQ = 440.0
LINE_HUM_AMP = 0.8
LINE_HUM_DURATION = 2.0
SPIKE_PLAY_DURATION = 10.0

def dispatcher(src, audio_q: "queue.Queue"):
    """Pass-through dispatcher: drain source.data_queue -> audio queue."""
    while src.running:
        try:
            ch = src.data_queue.get(timeout=0.1)
        except Exception:
            continue
        try:
            audio_q.put_nowait(ch)
        except queue.Full:
            # drop-oldest behavior: remove one and retry
            try:
                _ = audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_q.put_nowait(ch)
            except queue.Full:
                pass

def main():
    audio_q: "queue.Queue" = queue.Queue(maxsize=64)

    print("Starting audio player…")
    cfg = AudioConfig(
        out_samplerate = 44_100,
        out_channels = 2,
        gain = 0.9,
        blocksize = 512,
        ring_seconds = 0.75
    )
    player = AudioPlayer(audio_queue=audio_q,
                         input_samplerate=IN_SAMPLE_RATE,
                         config=cfg,
                         selected_channel=SELECTED_CHANNEL)
    player.start()

    def drain_audio_queue(q: "queue.Queue") -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def run_sim_phase(*, line_hum_amp: float, line_hum_freq: float, duration: float, label: str) -> None:
        print(f"{label}…")
        src = SimulatedPhysiologySource()
        devices = src.list_available_devices()
        if not devices:
            raise RuntimeError("No simulated devices available.")
        dev = devices[0]
        src.open(dev.id)
        chans = src.list_available_channels(dev.id)
        channel_ids = [c.id for c in chans]
        src.configure(sample_rate=IN_SAMPLE_RATE,
                      channels=channel_ids,
                      chunk_size=CHUNK_SIZE,
                      num_units=6,
                      line_hum_amp=line_hum_amp,
                      line_hum_freq=line_hum_freq)

        t_disp: threading.Thread | None = None
        try:
            src.start()
            thread_name = f"dispatcher-{label.replace(' ', '-').lower()}"
            t_disp = threading.Thread(target=dispatcher,
                                      args=(src, audio_q),
                                      name=thread_name,
                                      daemon=True)
            t_disp.start()
            time.sleep(duration)
        finally:
            src.stop()
            if t_disp is not None:
                t_disp.join(timeout=1.0)
            src.close()

    try:
        run_sim_phase(line_hum_amp=LINE_HUM_AMP,
                      line_hum_freq=LINE_HUM_FREQ,
                      duration=LINE_HUM_DURATION,
                      label="Phase 1: line hum 440 Hz")

        drain_audio_queue(audio_q)
        print("Switching to spike-only playback…")

        run_sim_phase(line_hum_amp=0.0,
                      line_hum_freq=LINE_HUM_FREQ,
                      duration=SPIKE_PLAY_DURATION,
                      label="Phase 2: spikes only")
    finally:
        print("Stopping…")
        player.stop()
        player.join(timeout=2.0)
        print("Done.")

if __name__ == "__main__":
    main()
