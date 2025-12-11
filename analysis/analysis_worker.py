from __future__ import annotations

import logging
import math
import queue
import threading
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

from shared.models import Chunk, EndOfStream
from shared.event_buffer import EventRingBuffer
from shared.types import AnalysisEvent

from .models import AnalysisBatch
from .settings import AnalysisSettings, AnalysisSettingsStore
from .metrics import energy_density, peak_frequency_sinc
from core.detection import AmpThresholdDetector, DETECTOR_REGISTRY


class AnalysisWorker(threading.Thread):
    """Background worker that receives filtered chunks and forwards them to an output queue."""

    def __init__(self, controller, channel_name: str, sample_rate: float, *, queue_size: int = 512) -> None:
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
        self._auto_detect_enabled = False
        self._auto_detector: Optional[AmpThresholdDetector] = None
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
        # Zero-copy optimization: sample is already float32 and read-only, slice returns a view
        samples = chunk.samples[idx : idx + 1]
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
        except Exception as e:
            logger.debug("Failed to create routed chunk: %s", e)
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
            except Exception as e:
                logger.debug("Failed to add events meta to chunk: %s", e)
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
                if self._channel_index < len(chunk.channel_names):
                    name = chunk.channel_names[self._channel_index]
                if name == self.channel_name:
                    return self._channel_index
        try:
            # Type ignore because tuple has index method but mypy might be confused by protocol
            idx = chunk.channel_names.index(self.channel_name)  # type: ignore[attr-defined]
            self._channel_index = int(idx)
            return self._channel_index
        except ValueError:
            return None

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
            except Exception as e:
                logger.debug("Failed to unregister analysis queue: %s", e)
            self._registration_token = None
        if self._settings_unsub:
            self._settings_unsub()
            self._settings_unsub = None

    def publish_event(self, event: AnalysisEvent) -> None:
        if self._event_buffer is not None:
            self._event_buffer.push(event)

    def _ensure_channel_id(self, channel_name: str) -> None:
        if self._channel_id is not None or self._controller is None:
            return
        try:
            infos = self._controller.active_channels()
        except Exception as e:
            logger.debug("Failed to get active channels: %s", e)
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
            window, miss_pre, miss_post = dispatcher.collect_window(
                int(start_index),
                int(count),
                int(self._channel_id),
                return_info=True,
            )
        except Exception as e:
            logger.debug("Failed to collect waveform window: %s", e)
            return np.empty(0, dtype=np.float32)
        if miss_pre > 0 or miss_post > 0:
            # Avoid splicing truncated windows; fall back to the local chunk data.
            return np.empty(0, dtype=np.float32)
        return np.asarray(window, dtype=np.float32)

    def _on_settings_changed(self, settings: AnalysisSettings) -> None:
        with self._state_lock:
            self._event_window_ms = float(settings.event_window_ms)
            if self._auto_detector is not None:
                self._auto_detector.configure(window_ms=self._event_window_ms)

    def configure_threshold(
        self,
        enabled: bool,
        value: float,
        *,
        secondary_enabled: Optional[bool] = None,
        secondary_value: Optional[float] = None,
        auto_detect: bool = False,
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
            self._auto_detect_enabled = bool(auto_detect)
            if self._auto_detect_enabled and self._auto_detector is None:
                # Use registry to instantiate detector
                det_cls = DETECTOR_REGISTRY["amp_threshold"]
                self._auto_detector = det_cls()
                # Configure for 5*sigma, bidirectional
                self._auto_detector.configure(factor=5.0, sign=0, window_ms=self._event_window_ms)
                if self.sample_rate > 0:
                    self._auto_detector.reset(self.sample_rate, 1)
            elif self._auto_detect_enabled and self._auto_detector is not None:
                # Update window if needed
                self._auto_detector.configure(window_ms=self._event_window_ms)

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

    def _detect_events(self, chunk: Chunk) -> list[AnalysisEvent]:
        with self._state_lock:
            event_window_ms = float(self._event_window_ms)
            threshold_enabled = bool(self._threshold_enabled)
            threshold_value = float(self._threshold_value)
            threshold_direction = self._threshold_direction
            secondary_enabled = bool(self._secondary_threshold_enabled)
            secondary_value = float(self._secondary_threshold_value)
            last_window_end = self._last_window_end_sample
            last_crossing_time = self._last_crossing_time_sec
            secondary_value = float(self._secondary_threshold_value)
            last_window_end = self._last_window_end_sample
            last_crossing_time = self._last_crossing_time_sec
            auto_detect = self._auto_detect_enabled
            detector = self._auto_detector
        
        if auto_detect and detector is not None:
            # Use modular detector
            # Ensure detector sample rate is correct
            # We can't easily check internal state, but we can re-reset if needed or trust configure/reset
            # For now, let's assume it's set correctly or we update it here if we track it.
            # Actually, let's just use it.
            
            # Note: AmpThresholdDetector expects a Chunk.
            # It returns core.detection.Event objects.
            # We need to convert/augment them.
            
            detected_events = detector.process_chunk(chunk)
            if not detected_events:
                return []
            
            events: list[tuple[AnalysisEvent, int, float]] = []
            
            # We need to calculate metrics for these events
            dt = float(chunk.dt)
            sr = (1.0 / dt) if dt > 0 else self.sample_rate
            
            for de in detected_events:
                # de is shared.models.DetectionEvent
                # We need to create shared.types.AnalysisEvent (which is the same class)
                # But AnalysisWorker adds extra properties.
                
                wf = de.window
                if wf.size == 0:
                    continue
                
                # Calculate metrics
                # Baseline
                baseline_val = float(np.median(wf)) # Simple median
                centered_wf = wf.astype(np.float64) - baseline_val
                
                # Energy
                energy = float(np.sum(wf.astype(np.float32) ** 2))
                window_sec = max(1e-12, wf.size * dt)
                ed = float(np.sqrt(energy / window_sec))
                
                # Peak Freq
                peak_freq = peak_frequency_sinc(centered_wf, sr)
                peak_wavelength = 1.0 / peak_freq if peak_freq > 1e-9 else 0.0
                
                # Interval
                crossing_time = de.t
                interval_sec = float("nan")
                if self._last_crossing_time_sec is not None:
                    delta = crossing_time - self._last_crossing_time_sec
                    if delta >= 0:
                        interval_sec = delta
                
                # Create Event
                # We need to map de.t (time) to sample index for last_end calculation
                # t = start_time + idx * dt
                # idx = (t - start_time) / dt
                rel_time = de.t - chunk.start_time
                rel_idx = int(round(rel_time / dt))
                
                start_sample = -1
                if chunk.meta:
                    try:
                        start_sample = int(chunk.meta.get("start_sample", -1))
                    except:
                        pass
                
                abs_idx = rel_idx if start_sample < 0 else start_sample + rel_idx
                
                # Calculate pre/post ms based on window
                # AmpThresholdDetector centers the event roughly?
                # Actually AmpThresholdDetector extracts window around crossing.
                # We can assume it's centered or just use the whole window.
                
                pre_samples = de.params.get("pre_samples")
                if pre_samples is None:
                    # Fallback if not provided
                    pre_samples = wf.size // 2
                else:
                    pre_samples = int(pre_samples)
                
                first_time_sec = de.t - (pre_samples * dt)
                
                event = AnalysisEvent(
                    id=self._next_event_id(),
                    channelId=de.chan, # This might need mapping if we have multiple channels
                    thresholdValue=de.params.get("threshold", 0.0),
                    crossingIndex=abs_idx,
                    crossingTimeSec=de.t,
                    firstSampleTimeSec=first_time_sec,
                    sampleRateHz=sr,
                    windowMs=float(self._event_window_ms),
                    preMs=float(pre_samples * dt * 1000.0),
                    postMs=float((wf.size - pre_samples) * dt * 1000.0),
                    samples=wf,
                    intervalSinceLastSec=interval_sec
                )
                
                props = getattr(event, "properties", None)
                if isinstance(props, dict):
                    props["energy"] = energy
                    props["window_sec"] = window_sec
                    props["energy_density"] = ed
                    props["peak_freq_hz"] = peak_freq
                    props["peak_wavelength_s"] = peak_wavelength
                    if np.isfinite(interval_sec):
                        props["interval_sec"] = float(interval_sec)
                
                # Update last_end
                # We need to know where this event ends in absolute samples
                # abs_idx is the crossing. Window end is approx abs_idx + half?
                # Let's just say abs_idx + wf.size
                this_end = abs_idx + wf.size
                events.append((event, this_end, crossing_time))
                
                # Update internal state for interval calculation in loop
                self._last_crossing_time_sec = crossing_time
            
            # Return collected events
            collected: list[AnalysisEvent] = []
            for event, new_last_end, crossing_time in events:
                self.publish_event(event)
                with self._state_lock:
                    if new_last_end > self._last_window_end_sample:
                        self._last_window_end_sample = new_last_end
                    self._last_crossing_time_sec = float(crossing_time)
                collected.append(event)
            return collected

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

        window_samples = int(round(event_window_ms * sr / 1000.0))
        if window_samples <= 0:
            window_samples = 1
        pre_samples = window_samples // 3
        post_samples = window_samples - pre_samples

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

        events: list[tuple[AnalysisEvent, int, float]] = []
        last_end = last_window_end
        prev_crossing_time = last_crossing_time
        target_len = window_samples
        for idx in idxs:
            abs_idx = idx if start_sample < 0 else start_sample + int(idx)
            if abs_idx < last_end:
                continue

            i0 = max(0, int(idx) - pre_samples)
            i1 = min(n, int(idx) + post_samples)
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
            has_full_pre = pre_count >= pre_samples
            has_full_post = post_count >= post_samples
            if (not has_full_pre or not has_full_post) and self._controller is not None:
                desired_start = crossing_index - pre_samples
                if desired_start < 0:
                    desired_start = 0
                extended = self._collect_waveform_samples(desired_start, target_len)
                if extended.size == target_len:
                    wf = extended.astype(np.float32, copy=True)
                    pre_count = min(pre_samples, crossing_index - desired_start)
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
            energy_density = float(np.sqrt(energy / window_sec))
            peak_freq = peak_frequency_sinc(centered_wf, sr, center_index=peak_idx)
            peak_wavelength = 1.0 / peak_freq if peak_freq > 1e-9 else 0.0
            event = AnalysisEvent(
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

        collected: list[AnalysisEvent] = []
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
