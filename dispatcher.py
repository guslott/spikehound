# dispatcher.py
from __future__ import annotations
import queue
import threading
import numpy as np
from typing import Optional, Callable


class Dispatcher:
    """
    Poll a source function for blocks shaped (frames, channels) and fan them
    out to one or more output queues (logger, audio, etc).
    Drops blocks if a consumer queue is full (non-blocking).
    """

    def __init__(
        self,
        source_fn: Callable[[], Optional[np.ndarray]],
        outputs: list["queue.Queue[np.ndarray]"],
        poll_s: float = 0.01,
    ) -> None:
        self._source_fn = source_fn
        self._outs = outputs
        self._poll = poll_s
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="Dispatcher", daemon=True)
        self._thr.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=join_timeout)
            self._thr = None

    def _run(self) -> None:
        import time
        while not self._stop.is_set():
            block = self._source_fn()
            if block is None:
                time.sleep(self._poll)
                continue
            arr = np.asarray(block, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2:
                continue
            for q in self._outs:
                try:
                    q.put_nowait(arr)
                except queue.Full:
                    pass
