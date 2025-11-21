from __future__ import annotations
import queue
import threading
from typing import Optional

import numpy as np
import wave
import os


class WavLoggerThread:
    """
    Reads float32 blocks shaped (frames, channels) from a Queue
    and writes them to a WAV file as 16-bit PCM.
    """

    def __init__(
        self,
        q: "queue.Queue[np.ndarray]",
        out_path: str,
        sample_rate: float,
        channels: int,
    ) -> None:
        self._q = q
        self._out_path = out_path
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)

        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._wf: Optional[wave.Wave_write] = None

    def start(self) -> None:
        if self._thr and self._thr.is_alive():
            return

        os.makedirs(os.path.dirname(self._out_path) or ".", exist_ok=True)

        wf = wave.open(self._out_path, "wb")
        wf.setnchannels(self._channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(self._sample_rate)
        self._wf = wf

        self._stop.clear()
        self._thr = threading.Thread(
            target=self._run,
            name="WavLoggerThread",
            daemon=True,
        )
        self._thr.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=join_timeout)
            self._thr = None

        if self._wf is not None:
            self._wf.close()
            self._wf = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                block = self._q.get(timeout=0.05)
            except queue.Empty:
                continue

            if block is None:
                continue

            arr = np.asarray(block, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2:
                continue

            # Convert float32 [-1, 1] to int16
            arr = np.clip(arr, -1.0, 1.0)
            int16 = (arr * 32767.0).astype("<i2")  # little-endian int16
            data = int16.tobytes()

            if self._wf is not None:
                self._wf.writeframesraw(data)
