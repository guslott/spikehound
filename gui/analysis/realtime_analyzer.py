from __future__ import annotations
import threading
import queue
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

from .models import Event

@dataclass
class ThresholdConfig:
    per_channel_thresholds: Optional[List[float]] = None  # None => auto
    polarity: str = "neg"       # "neg" | "pos" | "both"
    auto_k_sigma: float = 4.5   # k * noise sigma
    refractory_s: float = 0.003
    window_pre_s: float = 0.002
    window_post_s: float = 0.004

class RealTimeAnalyzer:
    """
    Consumes filtered chunks from `analysis_queue` (shape: (channels, frames))
    and emits `Event` objects to `event_queue` and `logging_queue`.

    Expected chunk attributes:
      - samples: np.ndarray (channels, frames), float32
      - start_sample: int
      - seq: int
      - start_time: float (seconds, monotonic)  [optional]
    """
    def __init__(
        self,
        analysis_queue: "queue.Queue",
        event_queue: "queue.Queue",
        logging_queue: "queue.Queue",
        sample_rate: float,
        n_channels: int,
        config: ThresholdConfig,
    ) -> None:
        self.analysis_queue = analysis_queue
        self.event_queue = event_queue
        self.logging_queue = logging_queue
        self.sr = float(sample_rate)
        self.n_channels = int(n_channels)
        self.cfg = config

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._thr: Optional[np.ndarray] = None  # thresholds per channel
        self._last_event_sample = np.full(self.n_channels, -10**12, dtype=np.int64)

        # pre/post windows in samples
        self._pre = max(0, int(round(self.cfg.window_pre_s * self.sr)))
        self._post = max(0, int(round(self.cfg.window_post_s * self.sr)))
        self._refrac = max(1, int(round(self.cfg.refractory_s * self.sr)))

    # ---------- lifecycle ----------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="RealTimeAnalyzer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---------- internals ----------
    def _run(self) -> None:
        # Small buffer to estimate noise if thresholds are auto
        noise_accum = []

        while not self._stop.is_set():
            try:
                item = self.analysis_queue.get(timeout=0.050)
            except queue.Empty:
                continue

            # The pipeline may use a sentinel object; ignore anything that
            # doesn’t look like a chunk with ndarray samples.
            samples = getattr(item, "samples", None)
            start_sample = getattr(item, "start_sample", None)
            if samples is None or start_sample is None:
                continue

            arr = np.asarray(samples, dtype=np.float32)  # (channels, frames)
            if arr.ndim != 2 or arr.shape[0] != self.n_channels:
                continue

            # Initialize thresholds (once) if needed
            if self._thr is None:
                self._thr = self._init_thresholds(arr, noise_accum)

            events = self._detect_events(arr, start_sample)
            if events:
                # Fan-out to both queues
                for ev in events:
                    try:
                        self.event_queue.put_nowait(ev)
                    except queue.Full:
                        pass
                    try:
                        self.logging_queue.put_nowait(ev)
                    except queue.Full:
                        pass

    def _init_thresholds(self, arr: np.ndarray, noise_accum: list) -> np.ndarray:
        if self.cfg.per_channel_thresholds is not None:
            thr = np.asarray(self.cfg.per_channel_thresholds, dtype=np.float32)
            if thr.shape[0] != self.n_channels:
                # Clamp or pad
                out = np.zeros(self.n_channels, dtype=np.float32)
                n = min(self.n_channels, thr.shape[0])
                out[:n] = thr[:n]
                out[n:] = thr[-1] if thr.size else 1.0
                return out
            return thr

        # Auto mode: estimate per-channel sigma via MAD over a few chunks
        noise_accum.append(arr)
        if len(noise_accum) < 4:
            # Not enough samples yet; return provisional thresholds to avoid false positives
            return np.full(self.n_channels, 10.0, dtype=np.float32)

        stack = np.concatenate(noise_accum[-8:], axis=1)  # (ch, frames)
        med = np.median(stack, axis=1)
        mad = np.median(np.abs(stack - med[:, None]), axis=1) + 1e-9
        sigma = 1.4826 * mad  # MAD → sigma
        thr = self.cfg.auto_k_sigma * sigma
        return thr.astype(np.float32)

    def _detect_events(self, arr: np.ndarray, start_sample: int) -> List[Event]:
        """
        Simple threshold crossing detector with refractory and window capture.
        """
        ch, n = arr.shape
        events: List[Event] = []

        if self._thr is None:
            return events

        for c in range(ch):
            x = arr[c, :]
            thr = float(self._thr[c])

            # crossing masks
            hits: np.ndarray
            if self.cfg.polarity == "neg":
                hits = (x < -thr)
            elif self.cfg.polarity == "pos":
                hits = (x > thr)
            else:  # "both"
                hits = (x > thr) | (x < -thr)

            if not np.any(hits):
                continue

            idxs = np.flatnonzero(hits).astype(np.int64)
            # enforce refractory per channel
            keep = []
            last_abs = self._last_event_sample[c]
            for i in idxs:
                abs_i = start_sample + int(i)
                if abs_i - last_abs >= self._refrac:
                    keep.append(i)
                    last_abs = abs_i
            if not keep:
                continue
            self._last_event_sample[c] = last_abs

            # build events with windows
            for i in keep:
                i0 = max(0, int(i) - self._pre)
                i1 = min(n, int(i) + self._post)
                wf = x[i0:i1].copy()  # (window,)
                amp = float(x[int(i)])
                ev = Event(
                    channel=c,
                    sample_index=start_sample + int(i),
                    amplitude=amp,
                    waveform=wf,
                    properties={
                        "pre_s": self._pre / self.sr,
                        "post_s": self._post / self.sr,
                        "threshold": thr,
                        "polarity": self.cfg.polarity,
                    },
                )
                events.append(ev)

        return events
