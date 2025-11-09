# analysis/realtime_analyzer.py
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from shared.event_buffer import EventRingBuffer
from shared.types import Event

from .models import ThresholdConfig


@dataclass
class _ChunkView:
    """
    Lightweight view that normalizes incoming objects to a common shape.
    We accept either:
      • obj.samples shaped (channels, frames)
      • obj.data shaped    (frames, channels)
    and expose .arr as (channels, frames).
    """
    arr: np.ndarray           # (channels, frames) float32
    start_time: float         # approximate mono time (s) at first sample
    seq: int                  # monotonic chunk id if available, else -1
    start_sample: int         # best-effort absolute sample index if available, else -1


def _as_chunk_view(obj) -> Optional[_ChunkView]:
    # Try common shapes used elsewhere in the project
    start_time = getattr(obj, "start_time", getattr(obj, "mono_time", time.monotonic()))
    seq = getattr(obj, "seq", -1)
    start_sample = getattr(obj, "start_sample", -1)
    if hasattr(obj, "samples"):
        a = np.asarray(obj.samples)
        if a.ndim == 2:
            # assume (channels, frames)
            return _ChunkView(arr=a.astype(np.float32, copy=False), start_time=float(start_time), seq=int(seq), start_sample=int(start_sample))
    if hasattr(obj, "data"):
        a = np.asarray(obj.data)
        if a.ndim == 2:
            # assume (frames, channels) -> transpose
            a = a.T
            return _ChunkView(arr=a.astype(np.float32, copy=False), start_time=float(start_time), seq=int(seq), start_sample=int(start_sample))
    return None


class RealTimeAnalyzer:
    """
    Consumes filtered chunks from `analysis_queue`, runs a simple threshold detector,
    and emits Event objects to `event_queue` (for GUI markers) and `logging_queue`
    (for persistence by the writer thread later).
    """

    def __init__(
        self,
        analysis_queue: "queue.Queue",
        event_queue: "queue.Queue",
        logging_queue: "queue.Queue",
        event_buffer: Optional[EventRingBuffer] = None,
        sample_rate: float,
        n_channels: int,
        config: ThresholdConfig,
    ) -> None:
        self.analysis_queue = analysis_queue
        self.event_queue = event_queue
        self.logging_queue = logging_queue
        self._event_buffer = event_buffer
        self.sr = float(sample_rate)
        self.nch = int(n_channels)
        self.cfg = config

        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

        # Per-channel state
        self._last_fire = np.full(self.nch, -1e9, dtype=np.float64)  # last event time (s)
        self._noise_mad = np.full(self.nch, np.nan, dtype=np.float32)
        self._auto_th = None  # computed absolute thresholds (if auto)
        self._event_id = 0

        # Waveform window in samples
        self._pre = max(0, int(round(self.cfg.window_pre_s * self.sr)))
        self._post = max(1, int(round(self.cfg.window_post_s * self.sr)))
        self._refrac = max(1, int(round(self.cfg.refractory_s * self.sr)))

        # Detection polarity flags
        pol = (self.cfg.polarity or "neg").lower()
        self._want_neg = pol in ("neg", "both")
        self._want_pos = pol in ("pos", "both")

    # ---------------- Public API ----------------

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="RealTimeAnalyzer", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=2.0)
            self._worker = None

    # --------------- Internals ------------------

    @staticmethod
    def _mad(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Median Absolute Deviation (robust noise proxy)."""
        med = np.median(x, axis=axis, keepdims=True)
        mad = np.median(np.abs(x - med), axis=axis)
        return mad

    def _maybe_update_auto_thresholds(self, chans_by_chunk: np.ndarray) -> None:
        """
        If per-channel thresholds aren't provided, compute abs thresholds as
        k * sigma, where sigma ≈ 1.4826 * MAD.
        """
        if self.cfg.per_channel_thresholds is not None:
            self._auto_th = None
            return
        # Compute per-channel MAD across frames for the current chunk, then EWMA
        mad = self._mad(chans_by_chunk, axis=1).astype(np.float32)  # shape (C,)
        # EWMA to stabilize
        alpha = 0.1
        if np.isnan(self._noise_mad).all():
            self._noise_mad = mad
        else:
            self._noise_mad = alpha * mad + (1.0 - alpha) * self._noise_mad

        sigma = 1.4826 * self._noise_mad
        k = float(self.cfg.auto_k_sigma)
        self._auto_th = k * sigma  # absolute thresholds

    def _thresholds(self) -> np.ndarray:
        if self.cfg.per_channel_thresholds is not None:
            th = np.asarray(self.cfg.per_channel_thresholds, dtype=np.float32)
            if th.size != self.nch:
                # broadcast scalar or clamp length
                if th.size == 1:
                    th = np.full(self.nch, float(th[0]), dtype=np.float32)
                else:
                    th = th[: self.nch]
            return np.abs(th)
        # auto thresholds computed from MAD
        if self._auto_th is None:
            return np.full(self.nch, 1.0, dtype=np.float32)  # temporary default
        return np.asarray(self._auto_th, dtype=np.float32)

    def _detect_in_chunk(self, ch_view: _ChunkView) -> None:
        """
        Run threshold detector on a normalized chunk (channels, frames).
        Emits Event objects for each detection.
        """
        X = ch_view.arr  # (C, F)
        C, F = X.shape
        if C != self.nch:
            # If channel count changes (e.g., user toggles), clamp gracefully.
            C = min(C, self.nch)
            X = X[:C, :]

        # Update auto thresholds if needed
        self._maybe_update_auto_thresholds(X)
        th = self._thresholds()[:C]  # absolute thresholds

        # Simple per-sample scan per channel (vectorized where reasonable)
        # We look for threshold crossings and enforce a refractory window.
        for c in range(C):
            sig = X[c, :]  # shape (F,)
            crossings_idx: np.ndarray = np.empty(0, dtype=np.int64)

            if self._want_neg:
                neg_idx = np.nonzero(sig <= -th[c])[0]
                crossings_idx = neg_idx if crossings_idx.size == 0 else np.union1d(crossings_idx, neg_idx)
            if self._want_pos:
                pos_idx = np.nonzero(sig >= th[c])[0]
                crossings_idx = pos_idx if crossings_idx.size == 0 else np.union1d(crossings_idx, pos_idx)

            if crossings_idx.size == 0:
                continue

            # Enforce refractory by skipping samples too close to the last fire
            events_to_emit: list[Tuple[int, float]] = []
            last_time_s = self._last_fire[c]
            for idx in crossings_idx:
                # Estimate absolute time of this sample
                t = ch_view.start_time + (idx / self.sr)
                if (t - last_time_s) * self.sr < self._refrac:
                    continue
                events_to_emit.append((idx, t))
                last_time_s = t
            if not events_to_emit:
                continue
            self._last_fire[c] = last_time_s

            # Emit events with waveform windows
            for idx, t in events_to_emit:
                i0 = max(0, idx - self._pre)
                i1 = min(F, idx + self._post)
                wf = sig[i0:i1].astype(np.float32, copy=True)
                # Best-effort absolute sample index
                abs_index = -1 if ch_view.start_sample < 0 else (ch_view.start_sample + int(idx))
                sr = self.sr if self.sr > 0 else 1.0
                window_ms = 1000.0 * (wf.size / sr)
                pre_ms = 1000.0 * (self._pre / sr)
                post_ms = 1000.0 * (self._post / sr)
                crossing_value = float(sig[idx])
                threshold_value = -float(th[c]) if crossing_value < 0 else float(th[c])
                ev = Event(
                    id=self._next_event_id(),
                    channelId=int(c),
                    thresholdValue=threshold_value,
                    crossingIndex=int(abs_index),
                    crossingTimeSec=float(t),
                    firstSampleTimeSec=float(ch_view.start_time),
                    sampleRateHz=float(self.sr),
                    windowMs=float(window_ms),
                    preMs=float(pre_ms),
                    postMs=float(post_ms),
                    samples=wf,
                )

                if self._event_buffer is not None:
                    self._event_buffer.push(ev)

                # Non-blocking puts (drop if queues are full; UI stays responsive)
                try:
                    self.event_queue.put_nowait(ev)
                except queue.Full:
                    pass
                try:
                    self.logging_queue.put_nowait(ev)
                except queue.Full:
                    pass

    def _next_event_id(self) -> int:
        self._event_id += 1
        return self._event_id

    def _run(self) -> None:
        """
        Main consumer loop. Short timeouts keep shutdown responsive.
        """
        while not self._stop.is_set():
            try:
                item = self.analysis_queue.get(timeout=0.050)
            except queue.Empty:
                continue

            # Some pipelines may send sentinel objects; ignore anything we can't parse.
            view = _as_chunk_view(item)
            if view is None:
                continue

            try:
                self._detect_in_chunk(view)
            except Exception:
                # Keep going even if one chunk is problematic.
                continue
