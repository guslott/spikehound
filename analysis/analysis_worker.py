from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import numpy as np

from core.models import Chunk, EndOfStream
from shared.event_buffer import EventRingBuffer
from shared.types import Event

from .settings import AnalysisSettings, AnalysisSettingsStore


class AnalysisWorker(threading.Thread):
    """Background worker that receives filtered chunks and forwards them to an output queue."""

    def __init__(self, controller, channel_name: str, sample_rate: float, *, queue_size: int = 64) -> None:
        super().__init__(name=f"AnalysisWorker-{channel_name}", daemon=True)
        self._controller = controller
        self.channel_name = channel_name
        self.sample_rate = float(sample_rate)
        self.input_queue: "queue.Queue[Chunk | EndOfStream]" = queue.Queue(maxsize=queue_size)
        self.output_queue: "queue.Queue[Chunk | EndOfStream]" = queue.Queue(maxsize=queue_size)
        self._stop_evt = threading.Event()
        self._registration_token: Optional[object] = None
        self._channel_index: Optional[int] = None
        self._event_buffer: Optional[EventRingBuffer] = getattr(controller, "event_buffer", None)
        self._settings_store: Optional[AnalysisSettingsStore] = getattr(controller, "analysis_settings_store", None)
        self._settings_unsub: Optional[Callable[[], None]] = None
        self._state_lock = threading.Lock()
        self._event_window_ms: float = 10.0
        self._threshold_enabled = False
        self._threshold_value = 0.5
        self._threshold_direction = "above"
        self._last_window_end_sample = -10**12
        self._event_id = 0
        if isinstance(self._settings_store, AnalysisSettingsStore):
            self._settings_unsub = self._settings_store.subscribe(self._on_settings_changed)

    def start(self) -> None:  # type: ignore[override]
        if self._controller is None:
            raise RuntimeError("Controller reference is required to register analysis worker")
        self._registration_token = self._controller.register_analysis_queue(self.input_queue)
        super().start()

    def run(self) -> None:  # type: ignore[override]
        while not self._stop_evt.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is EndOfStream:
                break
            if isinstance(item, Chunk):
                self._forward_chunk(item)
        self.output_queue.put(EndOfStream)

    def _forward_chunk(self, chunk: Chunk) -> None:
        idx = self._resolve_channel_index(chunk)
        if idx is None:
            return
        try:
            samples = np.array(chunk.samples[idx : idx + 1], dtype=np.float32)
        except Exception:
            return
        if samples.size == 0:
            return
        channel_names: tuple[str, ...]
        try:
            source_name = chunk.channel_names[idx]
        except (IndexError, TypeError):
            source_name = self.channel_name
        channel_names = (source_name,)
        meta = dict(chunk.meta or {})
        meta.setdefault("source_channel_index", idx)
        meta.setdefault("source_channel_name", source_name)
        try:
            routed_chunk = Chunk(
                samples=samples,
                start_time=chunk.start_time,
                dt=chunk.dt,
                seq=chunk.seq,
                channel_names=channel_names,
                units=chunk.units,
                meta=meta,
            )
        except Exception:
            return
        self._detect_events(routed_chunk)
        try:
            self.output_queue.put_nowait(routed_chunk)
        except queue.Full:
            try:
                _ = self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(routed_chunk)
            except queue.Full:
                pass

    def _resolve_channel_index(self, chunk: Chunk) -> Optional[int]:
        if self._channel_index is not None:
            if 0 <= self._channel_index < chunk.samples.shape[0]:
                name = None
                try:
                    name = chunk.channel_names[self._channel_index]
                except (IndexError, TypeError):
                    name = None
                if name == self.channel_name:
                    return self._channel_index
        try:
            idx = chunk.channel_names.index(self.channel_name)  # type: ignore[attr-defined]
        except ValueError:
            idx = None
        except AttributeError:
            idx = None
        if idx is None:
            return None
        self._channel_index = int(idx)
        return self._channel_index

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self.input_queue.put_nowait(EndOfStream)
        except queue.Full:
            pass
        removed_stop_attr = False
        cached_stop_attr = None
        if hasattr(self, "_stop"):
            cached_stop_attr = getattr(self, "_stop")
            if not callable(cached_stop_attr):
                try:
                    delattr(self, "_stop")
                    removed_stop_attr = True
                except AttributeError:
                    removed_stop_attr = False
        try:
            self.join(timeout=1.0)
        finally:
            if removed_stop_attr:
                setattr(self, "_stop", cached_stop_attr)
        if self._registration_token is not None:
            try:
                self._controller.unregister_analysis_queue(self._registration_token)
            except Exception:
                pass
            self._registration_token = None
        if self._settings_unsub:
            self._settings_unsub()
            self._settings_unsub = None

    def publish_event(self, event: Event) -> None:
        if self._event_buffer is not None:
            self._event_buffer.push(event)

    def _on_settings_changed(self, settings: AnalysisSettings) -> None:
        with self._state_lock:
            self._event_window_ms = float(settings.event_window_ms)

    def configure_threshold(self, enabled: bool, value: float) -> None:
        # Threshold 2 is intentionally disabled in the UI; only this primary control is honored.
        numeric_value = float(value)
        direction = "above" if numeric_value >= 0 else "below"
        with self._state_lock:
            self._threshold_enabled = bool(enabled)
            self._threshold_value = numeric_value
            self._threshold_direction = direction

    def update_sample_rate(self, sample_rate: float) -> None:
        """Refresh fallback sample rate and drop stale refractory state when it changes."""
        sample_rate = float(sample_rate)
        if sample_rate <= 0:
            return
        with self._state_lock:
            if abs(sample_rate - self.sample_rate) < 1e-3:
                return
            self.sample_rate = sample_rate
            self._last_window_end_sample = -10**12

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    def _next_event_id(self) -> int:
        self._event_id += 1
        return self._event_id

    def _detect_events(self, chunk: Chunk) -> None:
        with self._state_lock:
            event_window_ms = float(self._event_window_ms)
            threshold_enabled = bool(self._threshold_enabled)
            threshold_value = float(self._threshold_value)
            threshold_direction = self._threshold_direction
            last_window_end = self._last_window_end_sample
        if not threshold_enabled or threshold_value == 0.0 or event_window_ms <= 0:
            # No detections when the user has disabled Threshold 1.
            return
        if chunk.samples.size == 0:
            return

        sig = np.asarray(chunk.samples[0], dtype=np.float32)
        n = sig.size
        if n == 0:
            return
        dt = float(chunk.dt)
        sr = (1.0 / dt) if dt > 0 else self.sample_rate
        if sr <= 0:
            return

        half_samples = int(round((event_window_ms / 2.0) * sr / 1000.0))
        if half_samples <= 0:
            half_samples = 1
        window_samples = max(1, half_samples * 2)

        if threshold_value > 0:
            idxs = np.flatnonzero(sig >= threshold_value)
        else:
            idxs = np.flatnonzero(sig <= threshold_value)
        if idxs.size == 0:
            return

        start_sample = -1
        channel_id = self._channel_index
        meta = chunk.meta
        if meta is not None and hasattr(meta, "get"):
            try:
                start_sample = int(meta.get("start_sample", -1))
            except (TypeError, ValueError):
                start_sample = -1
            if channel_id is None:
                try:
                    channel_id = int(meta.get("source_channel_index"))
                except (TypeError, ValueError):
                    channel_id = None

        events: list[tuple[Event, int]] = []
        last_end = last_window_end
        for idx in idxs:
            abs_idx = idx if start_sample < 0 else start_sample + int(idx)
            if abs_idx < last_end:
                continue

            i0 = max(0, int(idx) - half_samples)
            i1 = min(n, int(idx) + half_samples + 1)
            if i1 <= i0:
                continue
            wf = sig[i0:i1].astype(np.float32, copy=True)
            dt_sec = 1.0 / sr
            first_time = float(chunk.start_time) + i0 * dt_sec
            crossing_time = float(chunk.start_time) + int(idx) * dt_sec
            pre_count = int(idx) - i0  # clamps gracefully when the buffer lacks pre-samples
            post_count = i1 - int(idx) - 1
            pre_ms = pre_count * dt_sec * 1000.0
            post_ms = post_count * dt_sec * 1000.0
            crossing_index = abs_idx
            energy = float(np.sum(wf.astype(np.float32) ** 2))
            window_sec = max(1e-12, wf.size * dt_sec)
            energy_density = energy / window_sec
            peak_freq = _peak_frequency_sinc(wf, sr)
            peak_wavelength = 1.0 / peak_freq if peak_freq > 1e-9 else 0.0
            event = Event(
                id=self._next_event_id(),
                channelId=int(channel_id) if channel_id is not None else 0,
                thresholdValue=threshold_value,
                crossingIndex=int(crossing_index),
                crossingTimeSec=float(crossing_time),
                firstSampleTimeSec=float(first_time),
                sampleRateHz=float(sr),
                windowMs=float(event_window_ms),
                preMs=float(pre_ms),
                postMs=float(post_ms),
                samples=wf,
            )
            props = getattr(event, "properties", None)
            if isinstance(props, dict):
                props["energy"] = energy
                props["window_sec"] = window_sec
                props["energy_density"] = energy_density
                props["peak_freq_hz"] = peak_freq
                props["peak_wavelength_s"] = peak_wavelength
            first_abs = crossing_index - pre_count
            last_end = first_abs + window_samples
            events.append((event, last_end))

        if not events:
            return

        for event, new_last_end in events:
            self.publish_event(event)
            with self._state_lock:
                if new_last_end > self._last_window_end_sample:
                    self._last_window_end_sample = new_last_end
def _peak_frequency_sinc(samples: np.ndarray, sr: float) -> float:
    if samples.size == 0 or sr <= 0:
        return 0.0
    x = np.asarray(samples, dtype=np.float64)
    x -= np.median(x)
    window = np.blackman(x.size)
    tapered = x * window
    n_fft_min = max(2048, x.size * 8)
    n_fft = 1 << max(0, int(np.ceil(np.log2(n_fft_min))))
    spectrum = np.fft.rfft(tapered, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mags = np.abs(spectrum)
    if mags.size <= 1:
        return 0.0
    mags[freqs < 50.0] = 0.0
    peak_idx = int(np.argmax(mags))
    if peak_idx <= 0 or peak_idx >= mags.size - 1:
        return float(max(0.0, freqs[peak_idx]))
    alpha = mags[peak_idx - 1]
    beta = mags[peak_idx]
    gamma = mags[peak_idx + 1]
    denom = max((alpha - 2 * beta + gamma), 1e-12)
    delta = 0.5 * (alpha - gamma) / denom
    peak_bin = peak_idx + delta
    freq = peak_bin * sr / n_fft
    if freq < 50.0:
        return 0.0
    return float(freq)
