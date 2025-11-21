# main_logging.py
import os, queue, time
import numpy as np
from dispatcher import Dispatcher
from logger_thread import H5LoggerThread

def make_sim_source(freq=440.0, sr=48000, block=2048, seconds=3.0, channels=1):
    import numpy as np
    phase = 0.0
    total = int(sr * seconds)
    generated = 0
    w = 2 * np.pi * freq / sr

    def _fn():
        nonlocal phase, generated
        if generated >= total:
            return None
        n = min(block, total - generated)
        t = np.arange(n, dtype=np.float32)
        sig = np.sin(phase + w * t)
        phase += w * n
        generated += n
        # (frames, channels)
        return np.stack([sig for _ in range(channels)], axis=1).astype(np.float32)

    return _fn, sr

def make_mic_source(sr=48000, block=2048, channels=1):
    import sounddevice as sd
    stream = sd.InputStream(
        samplerate=int(sr),
        channels=int(channels),
        dtype="float32",
        blocksize=int(block),
    )
    stream.start()

    def _fn():
        frames, _ = stream.read(block)
        return frames

    return _fn, sr


def make_mic_source(sr=48000, block=2048, channels=1):
    import sounddevice as sd
    stream = sd.InputStream(
        samplerate=int(sr),
        channels=int(channels),
        dtype="float32",
        blocksize=int(block),
    )
    stream.start()

    def _fn():
        frames, _ = stream.read(block)  # (block, channels) float32
        return frames

    return _fn, sr

def main():
    os.makedirs("logs", exist_ok=True)

    # 1) A queue of blocks shaped (frames, channels)
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)

    # 2) Start the HDF5 logger
    sr = 48000
    logger = H5LoggerThread(
        q,
        out_path="logs/session_dispatcher.h5",
        sample_rate=sr,
        channel_names=["mic"],
        channel_units=["volts"],
        channel_props={"mic": {"sensitivity_volts_per_count": 1.0}},
        block_hint=2048,
        compression=None,  # change to "gzip" later if you want
    )
    logger.start()

    # 3) Use a simulated source and run it through the Dispatcher → logging queue
    source_fn, sr = make_mic_source(sr=sr, block=2048, channels=1)
    disp = Dispatcher(source_fn=source_fn, outputs=[q], poll_s=0.01)
    disp.start()

    

    # 4) Let the source drain, then stop/flush
    time.sleep(3.2)   # a bit longer than 3.0s so the last block flushes
    disp.stop()
    logger.stop()
    print("Wrote logs/session_dispatcher.h5")


if __name__ == "__main__":
    main()
