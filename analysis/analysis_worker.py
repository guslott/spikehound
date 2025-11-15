from __future__ import annotations

import math
import queue
import threading
from typing import Callable, Optional

import numpy as np

from core.models import Chunk, EndOfStream
from shared.event_buffer import EventRingBuffer
from shared.types import Event

from .models import AnalysisBatch
from .settings import AnalysisSettings, AnalysisSettingsStore


class AnalysisWorker(threading.Thread):
    """Background worker that receives filtered chunks and forwards them to an output queue."""

    def __init__(self, controller, channel_name: str, sample_rate: float, *, queue_size: int = 64) -> None:
        super().__init__(name=f"AnalysisWorker-{channel_name}", daemon=True)
        self._controller = controller
        self.channel_name = channel_name
        self.sample_rate = float(sample_rate)
        self.input_queue: "queue.Queue[Chunk | EndOfStream]" = queue.Queue(maxsize=queue_size)
        self.output_queue: "queue.Queue[AnalysisBatch | EndOfStream]" = queue.Queue(maxsize=queue_size)
        self._stop_evt = threading.Event()
        self._registration_token: Optional[object] = None
        self._channel_index: Optional[int] = None
        self._event_buffer: Optional[EventRingBuffer] = getattr(controller, "event_buffer", None)
        self._settings_store: Optional[AnalysisSettingsStore] = getattr(controller, "analysis_settings_store", None)
        self._settings_unsub: Optional[Callable[[], None]] = None
        self._next_start_sample: int = 0
        self._state_lock = threading.Lock()
        self._event_window_ms: float = 10.0
        self._threshold_enabled = False
        self._threshold_value = 0.5
        self._threshold_direction = "above"
        self._secondary_threshold_enabled = False
        self._secondary_threshold_value = 0.0
        self._last_window_end_sample = -10**12
        self._last_crossing_time_sec: Optional[float] = None
        self._event_id = 0
        self._channel_id: Optional[int] = None
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
        self._ensure_channel_id(source_name)
        channel_names = (source_name,)
        meta = dict(chunk.meta or {})
        meta.setdefault("source_channel_index", idx)
        meta.setdefault("source_channel_name", source_name)
        start_sample = meta.get("start_sample")
        try:
            start_sample_int = int(start_sample)
        except (TypeError, ValueError):
            start_sample_int = -1
        if start_sample_int < 0:
            start_sample_int = self._next_start_sample
            meta["start_sample"] = start_sample_int
        self._next_start_sample = start_sample_int + samples.shape[1]
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
        events = self._detect_events(routed_chunk)
        events_tuple = tuple(events)
        if events_tuple:
            meta_with_events = dict(routed_chunk.meta or {})
            meta_with_events["analysis_events"] = events_tuple
            try:
                routed_chunk = Chunk(
                    samples=routed_chunk.samples,
                    start_time=routed_chunk.start_time,
                    dt=routed_chunk.dt,
                    seq=routed_chunk.seq,
                    channel_names=routed_chunk.channel_names,
                    units=routed_chunk.units,
                    meta=meta_with_events,
                )
            except Exception:
                pass
        batch = AnalysisBatch(chunk=routed_chunk, events=events_tuple)
        try:
            self.output_queue.put_nowait(batch)
        except queue.Full:
            try:
                _ = self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(batch)
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

    def _ensure_channel_id(self, channel_name: str) -> None:
        if self._channel_id is not None or self._controller is None:
            return
        try:
            infos = self._controller.active_channels()
        except Exception:
            return
        for info in infos or []:
            name = getattr(info, "name", None)
            if name == channel_name:
                self._channel_id = getattr(info, "id", None)
                break

    def _collect_waveform_samples(self, start_index: int, count: int) -> np.ndarray:
        if self._controller is None or self._channel_id is None or count <= 0 or start_index < 0:
            return np.empty(0, dtype=np.float32)
        dispatcher = getattr(self._controller, "dispatcher", None)
        if dispatcher is None:
            return np.empty(0, dtype=np.float32)
        try:
            window = dispatcher.collect_window(int(start_index), int(count), int(self._channel_id))
        except Exception:
            return np.empty(0, dtype=np.float32)
        return np.asarray(window, dtype=np.float32)

    def _on_settings_changed(self, settings: AnalysisSettings) -> None:
        with self._state_lock:
            self._event_window_ms = float(settings.event_window_ms)

    def configure_threshold(
        self,
        enabled: bool,
        value: float,
        *,
        secondary_enabled: Optional[bool] = None,
        secondary_value: Optional[float] = None,
    ) -> None:
        numeric_value = float(value)
        direction = "above" if numeric_value >= 0 else "below"
        secondary_flag = bool(secondary_enabled) if secondary_enabled is not None else False
        secondary_numeric = float(secondary_value) if secondary_value is not None else 0.0
        if not enabled:
            secondary_flag = False
        with self._state_lock:
            self._threshold_enabled = bool(enabled)
            self._threshold_value = numeric_value
            self._threshold_direction = direction
            self._secondary_threshold_enabled = secondary_flag
            self._secondary_threshold_value = secondary_numeric

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

    def _detect_events(self, chunk: Chunk) -> list[Event]:
        with self._state_lock:
            event_window_ms = float(self._event_window_ms)
            threshold_enabled = bool(self._threshold_enabled)
            threshold_value = float(self._threshold_value)
            threshold_direction = self._threshold_direction
            secondary_enabled = bool(self._secondary_threshold_enabled)
            secondary_value = float(self._secondary_threshold_value)
            last_window_end = self._last_window_end_sample
            last_crossing_time = self._last_crossing_time_sec
        if not threshold_enabled or threshold_value == 0.0 or event_window_ms <= 0:
            return []
        if chunk.samples.size == 0:
            return []

        sig = np.asarray(chunk.samples[0], dtype=np.float32)
        n = sig.size
        if n == 0:
            return []
        dt = float(chunk.dt)
        sr = (1.0 / dt) if dt > 0 else self.sample_rate
        if sr <= 0:
            return []

        half_samples = int(round((event_window_ms / 2.0) * sr / 1000.0))
        if half_samples <= 0:
            half_samples = 1
        window_samples = max(1, half_samples * 2)

        if threshold_value > 0:
            idxs = np.flatnonzero(sig >= threshold_value)
        else:
            idxs = np.flatnonzero(sig <= threshold_value)
        if idxs.size == 0:
            return []

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

        use_secondary = threshold_enabled and secondary_enabled

        events: list[tuple[Event, int, float]] = []
        last_end = last_window_end
        prev_crossing_time = last_crossing_time
        target_len = (half_samples * 2) + 1
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
            crossing_time = float(chunk.start_time) + int(idx) * dt_sec
            pre_count = int(idx) - i0
            post_count = i1 - int(idx) - 1
            pre_ms = pre_count * dt_sec * 1000.0
            post_ms = post_count * dt_sec * 1000.0
            crossing_index = abs_idx
            first_abs = crossing_index - pre_count
            candidate_last_end = first_abs + window_samples
            first_time = float(chunk.start_time) + i0 * dt_sec
            has_full_pre = pre_count >= half_samples
            has_full_post = post_count >= half_samples
            if (not has_full_pre or not has_full_post) and self._controller is not None:
                desired_start = crossing_index - half_samples
                if desired_start < 0:
                    desired_start = 0
                extended = self._collect_waveform_samples(desired_start, target_len)
                if extended.size == target_len:
                    wf = extended.astype(np.float32, copy=True)
                    pre_count = min(half_samples, crossing_index - desired_start)
                    post_count = target_len - pre_count - 1
                    pre_ms = pre_count * dt_sec * 1000.0
                    post_ms = post_count * dt_sec * 1000.0
                    first_abs = crossing_index - pre_count
                    candidate_last_end = first_abs + window_samples
                    first_time = crossing_time - (pre_count * dt_sec)

            if use_secondary and self._waveform_crosses_threshold(wf, secondary_value):
                continue
            interval_sec = float("nan")
            if prev_crossing_time is not None:
                delta = float(crossing_time) - float(prev_crossing_time)
                if delta < 0:
                    delta = 0.0
                interval_sec = delta
            if pre_count > 0:
                baseline = float(np.median(wf[:pre_count]))
            else:
                baseline = float(np.median(wf))
            centered_wf = wf.astype(np.float64) - baseline
            search_radius = max(1, int(round(0.001 * sr)))
            peak_window_start = max(0, pre_count - search_radius)
            peak_window_end = min(wf.size, pre_count + search_radius + 1)
            if peak_window_end <= peak_window_start:
                peak_window_end = min(wf.size, peak_window_start + 1)
            window_slice = centered_wf[peak_window_start:peak_window_end]
            if window_slice.size:
                local_idx = int(np.argmax(np.abs(window_slice)))
                peak_idx = peak_window_start + local_idx
            else:
                peak_idx = pre_count
            energy = float(np.sum(wf.astype(np.float32) ** 2))
            window_sec = max(1e-12, wf.size * dt_sec)
            energy_density = energy / window_sec
            peak_freq = _peak_frequency_sinc(centered_wf, sr, center_index=peak_idx)
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
                intervalSinceLastSec=float(interval_sec),
            )
            props = getattr(event, "properties", None)
            if isinstance(props, dict):
                props["energy"] = energy
                props["window_sec"] = window_sec
                props["energy_density"] = energy_density
                props["peak_freq_hz"] = peak_freq
                props["peak_wavelength_s"] = peak_wavelength
                if np.isfinite(interval_sec):
                    props["interval_sec"] = float(interval_sec)
            last_end = candidate_last_end
            events.append((event, last_end, float(crossing_time)))
            prev_crossing_time = float(crossing_time)

        if not events:
            return []

        collected: list[Event] = []
        for event, new_last_end, crossing_time in events:
            self.publish_event(event)
            with self._state_lock:
                if new_last_end > self._last_window_end_sample:
                    self._last_window_end_sample = new_last_end
                self._last_crossing_time_sec = float(crossing_time)
            collected.append(event)
        return collected

    @staticmethod
    def _waveform_crosses_threshold(waveform: np.ndarray, threshold: float) -> bool:
        if waveform.size == 0:
            return False
        if threshold >= 0:
            return bool(np.nanmax(waveform) >= threshold)
        return bool(np.nanmin(waveform) <= threshold)

def _peak_frequency_sinc(
    samples: np.ndarray,
    sr: float,
    *,
    min_hz: float = 50.0,
    center_index: Optional[int] = None,
) -> float:
    if sr <= 0:
        return 0.0
    data = np.asarray(samples, dtype=np.float64)
    if data.size < 8:
        return 0.0
    if not np.any(np.isfinite(data)):
        return 0.0

    data = np.nan_to_num(data, nan=0.0, copy=False)
    center = (
        int(center_index)
        if center_index is not None and 0 <= int(center_index) < data.size
        else int(np.argmax(np.abs(data)))
    )

    span = max(64, int(round(sr * 0.008)))  # ~8 ms window
    half = span // 2
    start = max(0, center - half)
    end = min(data.size, start + span)
    if end - start < 32:
        return 0.0
    segment = data[start:end].copy()

    # Remove mean and linear trend to suppress low-frequency energy
    segment -= np.mean(segment)
    idxs = np.arange(segment.size, dtype=np.float64)
    centered = idxs - idxs.mean()
    denom = float(np.dot(centered, centered))
    if denom > 0:
        slope = float(np.dot(centered, segment) / denom)
        segment -= slope * centered

    if not np.any(segment):
        return 0.0

    window = np.hanning(segment.size)
    tapered = segment * window
    target = max(4096, segment.size * 8)
    n_fft = 1 << int(math.ceil(math.log2(target)))
    spectrum = np.fft.rfft(tapered, n=n_fft)
    mags = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    if freqs.size != mags.size or mags.size <= 1:
        return 0.0

    max_freq = min(sr / 6.0, 1000.0)
    valid = (freqs >= max(min_hz, 1.0)) & (freqs <= max_freq)
    if not np.any(valid):
        return 0.0
    mags = mags[valid]
    freqs = freqs[valid]
    if not np.any(np.isfinite(mags)):
        return 0.0
    power = mags * mags
    if power.size >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        power = np.convolve(power, kernel, mode="same")
    peak_idx = int(np.argmax(power))
    peak_freq = freqs[peak_idx]

    # Quadratic interpolation for sub-bin precision when neighbors exist
    if 0 < peak_idx < mags.size - 1:
        alpha, beta, gamma = mags[peak_idx - 1 : peak_idx + 2]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
            delta = float(np.clip(delta, -1.0, 1.0))
            bin_width = freqs[1] - freqs[0]
            peak_freq += delta * bin_width

    peak_freq = float(max(0.0, peak_freq))
    if peak_freq >= max_freq * 0.98 or peak_freq <= min_hz * 1.02:
        return _autocorr_frequency(segment, sr, min_hz, max_hz=max_freq)
    auto_freq = _autocorr_frequency(segment, sr, min_hz, max_hz=max_freq)
    if auto_freq <= 0.0:
        return peak_freq
    if peak_freq < min_hz:
        return auto_freq
    rel_diff = abs(auto_freq - peak_freq) / max(min(auto_freq, peak_freq), 1e-6)
    if rel_diff > 0.25:
        return auto_freq
    return peak_freq


def _autocorr_frequency(segment: np.ndarray, sr: float, min_hz: float, max_hz: float) -> float:
    if sr <= 0 or segment.size < 2 or max_hz <= min_hz:
        return 0.0
    data = np.asarray(segment, dtype=np.float64)
    if not np.any(np.isfinite(data)):
        return 0.0
    corr = np.correlate(data, data, mode="full")
    corr = corr[corr.size // 2 :]
    if corr.size <= 1:
        return 0.0
    counts = np.arange(corr.size, 0, -1, dtype=np.float64)
    corr = corr / counts
    corr[0] = 0.0
    max_period = min(int(sr / max(min_hz, 1e-6)), corr.size - 1)
    min_period = max(1, int(sr / max(max_hz, 1e-6)))
    if max_period <= min_period:
        return 0.0
    segment_corr = corr[min_period : max_period + 1]
    lags = np.arange(min_period, max_period + 1, dtype=np.float64)
    if segment_corr.size != lags.size or not np.any(np.isfinite(segment_corr)):
        return 0.0
    scores = segment_corr * np.sqrt(np.maximum(1.0, lags))
    best_idx = int(np.argmax(scores))
    if best_idx <= 0 or best_idx >= segment_corr.size - 1:
        return 0.0
    lag = int(lags[best_idx])
    if lag <= 0:
        return 0.0
    freq = sr / lag
    return float(freq if freq >= min_hz else 0.0)
