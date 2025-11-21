# main_logging_wav.py
import os, queue, time
import numpy as np

from dispatcher import Dispatcher
from wav_logger_thread import WavLoggerThread
from main_logging import make_mic_source, make_sim_source  # reuse your source functions


def main():
    os.makedirs("logs", exist_ok=True)

    # Queue of (frames, channels) float32 blocks
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)

    # Choose sample rate
    sr = 48000

    # Choose source: mic OR simulated
    # Uncomment ONE of these:

    # 1) Real microphone:
    source_fn, sr = make_mic_source(sr=sr, block=2048, channels=1)

    # 2) Simulated sine wave (comment mic and uncomment this to test):
    # source_fn, sr = make_sim_source(freq=440.0, sr=sr, block=2048, seconds=3.0, channels=1)

    # WAV logger
    logger = WavLoggerThread(
        q=q,
        out_path="logs/session.wav",
        sample_rate=sr,
        channels=1,
    )
    logger.start()

    # Dispatcher: source → queue
    disp = Dispatcher(source_fn=source_fn, outputs=[q], poll_s=0.01)
    disp.start()

    # Run for ~3 seconds, then stop
    try:
        time.sleep(3.2)
    finally:
        disp.stop()
        logger.stop()
        print("Wrote logs/session.wav")


if __name__ == "__main__":
    main()
