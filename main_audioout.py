import threading
import queue
import time

from audio.player import AudioPlayer, AudioConfig

# Adjust this import path if your sim source lives elsewhere:
from daq.simulated_source import SimulatedPhysiologySource

IN_SAMPLE_RATE = 20_000
CHUNK_SIZE = 200
SELECTED_CHANNEL = 1  # try 0 or 1 if you don't hear anything

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
    # 1) Simulated source
    print("Setting up simulated source…")
    src = SimulatedPhysiologySource()
    dev = src.list_available_devices()[0]
    src.open(dev.id)
    chans = src.list_available_channels(dev.id)
    src.configure(sample_rate=IN_SAMPLE_RATE,
                  channels=[c.id for c in chans],
                  chunk_size=CHUNK_SIZE,
                  num_units=6)
    src.start()

    # 2) Audio queue and dispatcher
    audio_q: "queue.Queue" = queue.Queue(maxsize=64)
    print("Starting dispatcher…")
    t_disp = threading.Thread(target=dispatcher, args=(src, audio_q),
                              name="dispatcher", daemon=True)
    t_disp.start()

    # 3) Audio player
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

        # --- TEMP: 2-second tone test at 440 Hz to verify audio path ---
    import numpy as np
    TONE_TEST = True
    if TONE_TEST:
        print("Tone test (440 Hz) for 2 seconds…")
        dur = 2.0
        n = int(IN_SAMPLE_RATE * dur)                # samples at the *input* sample rate
        t = np.arange(n, dtype=np.float32) / IN_SAMPLE_RATE
        tone = 0.2 * np.sin(2*np.pi*440.0 * t)      # mono at IN_SAMPLE_RATE
        # Match the player’s expected shape: (frames, channels). Selected channel = 0
        tone = tone[:, None]
        # Push in manageable chunks so we don’t overflow the queue
        step = 1000
        for i in range(0, n, step):
            player.q.put(tone[i:i+step], block=True, timeout=0.5)


    # 4) Let it play
    print("Playing spikes for 10 seconds…")
    try:
        time.sleep(10)
    finally:
        print("Stopping…")
        player.stop()
        player.join(timeout=2.0)
        src.stop(); src.close()
        print("Done.")

if __name__ == "__main__":
    main()
