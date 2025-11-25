from __future__ import annotations

import logging
import queue
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from PySide6 import QtCore

from .conditioning import FilterSettings, SignalConditioner
from shared.models import Chunk, ChunkPointer, EndOfStream, TriggerConfig
from shared.ring_buffer import SharedRingBuffer

logger = logging.getLogger(__name__)


@dataclass
class DispatcherStats:
    received: int = 0
    processed: int = 0
    forwarded: Counter = field(default_factory=Counter)
    evicted: Counter = field(default_factory=Counter)
    dropped: Counter = field(default_factory=Counter)

    def snapshot(self) -> Dict[str, object]:
        return {
            "received": self.received,
            "processed": self.processed,
            "forwarded": dict(self.forwarded),
            "evicted": dict(self.evicted),
            "dropped": dict(self.dropped),
        }


class DispatcherSignals(QtCore.QObject):
    tick = QtCore.Signal(dict)


class Dispatcher:
    """Router thread that conditions incoming samples from ChunkPointers and fans them out to consumers."""

    def __init__(
        self,
        raw_queue: "queue.Queue[ChunkPointer | EndOfStream]",
        visualization_queue: "queue.Queue[ChunkPointer | EndOfStream]",
        audio_queue: "queue.Queue[ChunkPointer | EndOfStream]",
        logging_queue: "queue.Queue[Chunk | EndOfStream]",
        *,
        filter_settings: Optional[FilterSettings] = None,
        poll_timeout: float = 0.05,
    ) -> None:
        self._raw_queue = raw_queue
        self._output_queues: Dict[str, queue.Queue] = {
            "visualization": visualization_queue,
            "audio": audio_queue,
        }
        self._logging_queue = logging_queue
        self._conditioner = SignalConditioner(filter_settings)
        self._poll_timeout = poll_timeout
        self.signals = DispatcherSignals()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_thread: Optional[threading.Thread] = None
        self._tick_interval = 1.0 / 30.0
        self._stats = DispatcherStats()
        self._stats_lock = threading.Lock()

        # Ring buffer / selection state
        self._ring_lock = threading.Lock()
        self._source_buffer: Optional[SharedRingBuffer] = None
        self.viz_buffer: SharedRingBuffer = SharedRingBuffer((1, 1), dtype=np.float32)
        self._write_idx: int = 0
        self._filled: int = 0
        self._latest_sample_index: int = -1
        self._sample_rate: Optional[float] = None
        self._window_sec: float = 0.2
        self._channel_names: tuple[str, ...] = tuple()
        self._channel_ids: tuple[int, ...] = tuple()
        self._channel_index_map: Dict[int, int] = {}
        self._current_trigger: Optional[TriggerConfig] = None
        self._active_channel_ids: list[int] = []
        self._next_start_sample: int = 0
        self._next_seq: int = 0
        self._analysis_lock = threading.Lock()
        self._analysis_queues: Dict[int, queue.Queue] = {}
        self._next_analysis_id = 1

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="DispatcherThread", daemon=True)
        self._thread.start()
        self._start_tick_thread()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._tick_thread is not None:
            self._tick_thread.join()
            self._tick_thread = None
        self._broadcast_end_of_stream()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def update_filter_settings(self, settings: FilterSettings) -> None:
        self._conditioner.update_settings(settings)

    def set_active_channels(self, channel_ids: Sequence[int]) -> None:
        with self._ring_lock:
            self._active_channel_ids = list(channel_ids)

    def set_source_buffer(self, buffer: SharedRingBuffer, *, sample_rate: Optional[float] = None) -> None:
        """
        Point the dispatcher at the source's shared buffer and prime viz buffer sizing.
        """
        if buffer is None:
            raise ValueError("buffer must not be None")
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        with self._ring_lock:
            # Check if buffer or sample rate actually changed
            if self._source_buffer is buffer and (sample_rate is None or self._sample_rate == float(sample_rate)):
                # Buffer is identical, no need to reset
                return

            self._source_buffer = buffer
            if sample_rate is not None:
                self._sample_rate = float(sample_rate)
            channels = buffer.shape[0]
            if not self._channel_ids:
                self._channel_ids = tuple(range(channels))
                if not self._channel_names:
                    self._channel_names = tuple(str(idx) for idx in range(channels))
                self._channel_index_map = {cid: idx for idx, cid in enumerate(self._channel_ids)}
            self._ensure_viz_buffer_locked(channels, min_capacity=buffer.capacity)
            self._reset_viz_counters_locked()
            logger.info("Dispatcher linked to source buffer: shape=%s, sr=%s", buffer.shape, self._sample_rate)

    def clear_active_channels(self) -> None:
        with self._ring_lock:
            self._active_channel_ids = []
            self._reset_viz_counters_locked()

    def set_channel_layout(self, channel_ids: Sequence[int], channel_names: Sequence[str]) -> None:
        with self._ring_lock:
            new_ids = tuple(channel_ids)
            new_names = tuple(channel_names)
            
            # Only reset if layout actually changed - prevents unnecessary buffer resets
            # that would truncate PSP tails spanning multiple chunks
            if new_ids == self._channel_ids and new_names == self._channel_names:
                # logger.info("Layout unchanged, skipping reset")
                return
                
            self._channel_ids = new_ids
            self._channel_names = new_names
            self._channel_index_map = {cid: idx for idx, cid in enumerate(self._channel_ids)}
            channels = len(self._channel_ids) if self._channel_ids else len(self._channel_names)
            min_capacity = self._source_buffer.capacity if self._source_buffer is not None else 1
            self._ensure_viz_buffer_locked(channels or 1, min_capacity=min_capacity)
            self._reset_viz_counters_locked() # Log this call?

    def reset_buffers(self) -> None:
        with self._ring_lock:
            shape = self.viz_buffer.shape
            self.viz_buffer = SharedRingBuffer(shape, dtype=np.float32)
            self._reset_viz_counters_locked()

    def emit_empty_tick(self) -> None:
        self.signals.tick.emit(self._empty_payload())

    # Analysis registration ----------------------------------------------------

    def register_analysis_queue(self, data_queue: "queue.Queue[Chunk | EndOfStream]") -> int:
        with self._analysis_lock:
            token = self._next_analysis_id
            self._next_analysis_id += 1
            self._analysis_queues[token] = data_queue
        return token

    def unregister_analysis_queue(self, token: int) -> None:
        with self._analysis_lock:
            queue_obj = self._analysis_queues.pop(token, None)
        if queue_obj is not None:
            try:
                queue_obj.put_nowait(EndOfStream)
            except queue.Full:
                pass

    # Trigger configuration stubs -------------------------------------

    def set_trigger_config(self, config: TriggerConfig, sample_rate: float) -> None:
        with self._ring_lock:
            self._current_trigger = config
            if sample_rate > 0:
                self._sample_rate = float(sample_rate)
            self._window_sec = max(config.window_sec, 1e-3)
            channels = self.viz_buffer.shape[0]
            self._ensure_viz_buffer_locked(channels, min_capacity=self.viz_buffer.capacity)

    def set_window_duration(self, window_sec: float) -> None:
        if window_sec <= 0:
            return
        with self._ring_lock:
            self._window_sec = window_sec
            channels = self.viz_buffer.shape[0]
            self._ensure_viz_buffer_locked(channels, min_capacity=self.viz_buffer.capacity)

    def snapshot(self) -> Dict[str, object]:
        with self._stats_lock:
            return self._stats.snapshot()

    def buffer_status(self) -> Dict[str, float]:
        with self._ring_lock:
            filled = float(self._filled)
            capacity = float(self.viz_buffer.capacity)
            sample_rate = float(self._sample_rate or 0.0)
            seconds = filled / sample_rate if sample_rate > 0 else 0.0
            capacity_seconds = capacity / sample_rate if sample_rate > 0 else 0.0
            utilization = (filled / capacity) if capacity > 0 else 0.0
            return {
                "samples": filled,
                "capacity": capacity,
                "seconds": seconds,
                "capacity_seconds": capacity_seconds,
                "utilization": utilization,
            }

    def _run(self) -> None:
        while True:
            try:
                item = self._raw_queue.get(timeout=self._poll_timeout)
            except queue.Empty:
                if self._stop_event.is_set():
                    self._broadcast_end_of_stream()
                    break
                continue

            try:
                if item is EndOfStream:
                    self._broadcast_end_of_stream()
                    break

                if not isinstance(item, ChunkPointer):
                    continue

                try:
                    processed = self._process_pointer(item)
                    if processed is None:
                        continue
                    raw_chunk, filtered_chunk, viz_pointer = processed
                    self._fan_out(raw_chunk, filtered_chunk, viz_pointer)
                except Exception as exc:
                    logger.warning("Dispatcher skipped bad chunk: %s", exc)
            finally:
                self._raw_queue.task_done()

    def _process_pointer(self, ptr: ChunkPointer) -> Optional[tuple[Chunk, Chunk, ChunkPointer]]:
        with self._stats_lock:
            self._stats.received += 1

        source_buffer = self._source_buffer
        sample_rate = self._sample_rate
        if source_buffer is None or sample_rate is None or sample_rate <= 0:
            logger.warning("No source buffer linked; skipping pointer")
            return None

        raw = source_buffer.read(ptr.start_index, ptr.length)
        if raw.ndim != 2 or raw.shape[1] == 0:
            logger.warning("Read empty data from source (shape=%s)", getattr(raw, "shape", None))
            return None

        with self._ring_lock:
            # Only reset channel IDs if the COUNT mismatches.
            # If the count matches, we trust the IDs set by set_channel_layout().
            # This allows non-contiguous IDs (e.g. [0, 2]) to persist.
            if not self._channel_ids or len(self._channel_ids) != raw.shape[0]:
                # Fallback: if count is wrong, we have to reset to default indices
                self._channel_ids = tuple(range(raw.shape[0]))
                self._channel_index_map = {cid: idx for idx, cid in enumerate(self._channel_ids)}
                self._channel_names = tuple(str(idx) for idx in range(raw.shape[0]))
            elif not self._channel_names:
                self._channel_names = tuple(str(idx) for idx in range(raw.shape[0]))
            channel_names: tuple[str, ...] = self._channel_names
        dt = 1.0 / float(sample_rate)

        start_sample = self._next_start_sample
        seq = self._next_seq
        self._next_start_sample += raw.shape[1]
        self._next_seq += 1

        meta: Dict[str, object] = {"start_sample": start_sample}
        raw_chunk = Chunk(
            samples=np.ascontiguousarray(raw),
            start_time=ptr.render_time,
            dt=dt,
            seq=seq,
            channel_names=channel_names,
            units="unknown",
            meta=meta,
        )

        filtered_samples = np.ascontiguousarray(self._conditioner.process(raw_chunk))
        filters_active = self._conditioner.settings.any_enabled()
        filtered_meta = dict(meta)
        filtered_meta["filtered"] = filters_active
        filtered_meta["filters"] = self._conditioner.describe()
        filtered_chunk = Chunk(
            samples=filtered_samples,
            start_time=ptr.render_time,
            dt=dt,
            seq=seq,
            channel_names=channel_names,
            units="unknown",
            meta=filtered_meta,
        )

        with self._ring_lock:
            self._ensure_viz_buffer_locked(filtered_samples.shape[0], min_capacity=ptr.length)
            viz_start = self.viz_buffer.write(filtered_samples)
            self._update_viz_bookkeeping_locked(ptr.length, start_sample, viz_start)

        with self._stats_lock:
            self._stats.processed += 1

        viz_pointer = ChunkPointer(
            start_index=viz_start,
            length=ptr.length,
            render_time=ptr.render_time,
        )
        return raw_chunk, filtered_chunk, viz_pointer

    def _fan_out(self, raw_chunk: Chunk, filtered_chunk: Chunk, viz_pointer: ChunkPointer) -> None:
        # DISABLED: Logging queue disabled until explicit data logging feature is added
        # self._enqueue(self._logging_queue, raw_chunk, "logging")
        for name, out_queue in self._output_queues.items():
            if name in ("visualization", "audio"):
                self._enqueue(out_queue, viz_pointer, name)
            else:
                self._enqueue(out_queue, filtered_chunk, name)
        self._dispatch_to_analysis(filtered_chunk)

    def _dispatch_to_analysis(self, filtered_chunk: Chunk) -> None:
        with self._analysis_lock:
            targets = list(self._analysis_queues.items())
        if not targets:
            return
        for token, queue_obj in targets:
            try:
                queue_obj.put_nowait(filtered_chunk)
            except queue.Full:
                try:
                    _ = queue_obj.get_nowait()
                    with self._stats_lock:
                        self._stats.evicted["analysis"] += 1
                except queue.Empty:
                    pass
                try:
                    queue_obj.put_nowait(filtered_chunk)
                except queue.Full:
                    with self._stats_lock:
                        self._stats.dropped["analysis"] += 1
                else:
                    with self._stats_lock:
                        self._stats.forwarded["analysis"] += 1
            else:
                with self._stats_lock:
                    self._stats.forwarded["analysis"] += 1

    def _enqueue(self, target_queue: queue.Queue, item: object, queue_name: str) -> None:
        """Enqueue with blocking for critical queues (visualization), non-blocking for logging."""
        # Logging is not critical - use drop-oldest policy
        if queue_name == "logging":
            try:
                target_queue.put_nowait(item)
            except queue.Full:
                # Drop oldest and try again (non-blocking for logging)
                try:
                    target_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    target_queue.put_nowait(item)
                except queue.Full:
                    pass  # Give up silently
            else:
                with self._stats_lock:
                    self._stats.forwarded[queue_name] += 1
        else:
            # Visualization and audio are CRITICAL - use blocking mode
            try:
                # BLOCKING MODE: Wait up to 10 seconds. If still full, fatal error.
                target_queue.put(item, block=True, timeout=10.0)
            except queue.Full:
                # This should NEVER happen - indicates consumer deadlock
                print(f"\n!!! CRITICAL ERROR: Dispatcher queue '{queue_name}' BLOCKED for 10+ seconds !!!")
                print(f"!!! Consumer is not keeping up - system deadlocked !!!")
                with self._stats_lock:
                    self._stats.evicted[queue_name] += 1
                raise RuntimeError(f"Dispatcher queue '{queue_name}' blocked - lossless constraint violated")
            else:
                with self._stats_lock:
                    self._stats.forwarded[queue_name] += 1

    def _broadcast_end_of_stream(self) -> None:
        # Deliver sentinel to all downstream queues, draining oldest entries if needed
        targets = {"logging": self._logging_queue, **self._output_queues}
        for name, target in targets.items():
            if target is None:
                continue
            placed = False
            while not placed:
                try:
                    target.put_nowait(EndOfStream)
                    placed = True
                except queue.Full:
                    try:
                        _ = target.get_nowait()
                        with self._stats_lock:
                            self._stats.evicted[name] += 1
                    except queue.Empty:
                        pass
        with self._analysis_lock:
            queues = list(self._analysis_queues.values())
        for q in queues:
            try:
                q.put_nowait(EndOfStream)
            except queue.Full:
                pass

    def _start_tick_thread(self) -> None:
        if self._tick_thread is not None and self._tick_thread.is_alive():
            return
        self._tick_thread = threading.Thread(target=self._tick_loop, name="DispatcherTick", daemon=True)
        self._tick_thread.start()

    def _tick_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._collect_window_payload()
                if payload is not None:
                    self.signals.tick.emit(payload)
            except Exception as exc:
                logger.error("Dispatcher tick error: %s", exc)
            self._stop_event.wait(self._tick_interval)

    def _collect_window_payload(self) -> Optional[dict]:
        with self._ring_lock:
            status = {
                "sample_rate": float(self._sample_rate or 0.0),
                "window_sec": float(self._window_sec),
            }
            sample_rate = float(self._sample_rate or 0.0)
            if sample_rate <= 0 or self._filled == 0:
                return self._empty_payload(status)
            mode = "continuous"
            if self._current_trigger is not None:
                mode = str(self._current_trigger.mode)
            if mode not in ("continuous", "stream", "single"):
                return self._empty_payload(status)

            capacity = self.viz_buffer.capacity
            window_samples = int(max(1, min(capacity, self._window_sec * sample_rate)))
            window_samples = min(window_samples, self._filled)
            
            if window_samples <= 0:
                return self._empty_payload(status)

            start = (self._write_idx - window_samples) % capacity
            data = np.array(self.viz_buffer.read(start, window_samples), copy=True, dtype=np.float32)

            if self._active_channel_ids:
                indices = [self._channel_index_map[cid] for cid in self._active_channel_ids if cid in self._channel_index_map]
            else:
                indices = list(range(data.shape[0]))
            if not indices:
                return self._empty_payload(status)

            safe_indices = [idx for idx in indices if 0 <= idx < data.shape[0]]
            if not safe_indices:
                return self._empty_payload(status)
            data = data[safe_indices, :]
            channel_ids = [self._channel_ids[idx] if idx < len(self._channel_ids) else idx for idx in safe_indices]
            channel_names = [self._channel_names[idx] if idx < len(self._channel_names) else str(channel_ids[i]) for i, idx in enumerate(safe_indices)]
            actual_window_sec = window_samples / sample_rate if sample_rate else 0.0
            times = np.linspace(0.0, actual_window_sec, window_samples, endpoint=False, dtype=np.float32)
            return {
                "channel_ids": channel_ids,
                "channel_names": channel_names,
                "samples": data,
                "times": times,
                "status": status,
            }

    def collect_window(
        self,
        start_index: int,
        window_samples: int,
        channel_id: int,
        *,
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, int, int]:
        """Return a window of samples for channel_id starting at start_index.

        When ``return_info`` is True, the method returns a tuple of
        (samples, missing_prefix, missing_suffix) where the missing values
        indicate how many samples at the start/end were unavailable.
        """
        window_samples = int(window_samples)
        if window_samples <= 0:
            result = np.empty(0, dtype=np.float32)
            if return_info:
                return result, 0, 0
            return result
        with self._ring_lock:
            if self._filled == 0 or self._latest_sample_index < 0 or self.viz_buffer.capacity <= 0:
                zeros = np.zeros(window_samples, dtype=np.float32)
                if return_info:
                    return zeros, window_samples, window_samples
                return zeros
            channel_idx = self._channel_index_map.get(int(channel_id))
            if channel_idx is None or not (0 <= channel_idx < self.viz_buffer.shape[0]):
                zeros = np.zeros(window_samples, dtype=np.float32)
                if return_info:
                    return zeros, window_samples, window_samples
                return zeros
            if self._filled < window_samples:
                zeros = np.zeros(window_samples, dtype=np.float32)
                if return_info:
                    return zeros, window_samples, window_samples
                return zeros
            start_idx = int(start_index)
            if start_idx < 0:
                zeros = np.zeros(window_samples, dtype=np.float32)
                if return_info:
                    return zeros, window_samples, window_samples
                return zeros
            earliest = self._latest_sample_index - (self._filled - 1)
            latest = self._latest_sample_index + 1  # exclusive
            desired_start = start_idx
            desired_end = start_idx + window_samples
            actual_start = max(desired_start, earliest)
            actual_end = min(desired_end, latest)
            if actual_start >= actual_end:
                zeros = np.zeros(window_samples, dtype=np.float32)
                if return_info:
                    return zeros, window_samples, window_samples
                return zeros
            missing_prefix = max(0, earliest - desired_start)
            missing_suffix = max(0, desired_end - latest)
            available_count = window_samples - missing_prefix - missing_suffix
            if available_count <= 0:
                zeros = np.zeros(window_samples, dtype=np.float32)
                if return_info:
                    return zeros, window_samples, window_samples
                return zeros
            buffer_len = self.viz_buffer.capacity
            if buffer_len <= 0:
                zeros = np.zeros(window_samples, dtype=np.float32)
                if return_info:
                    return zeros, window_samples, window_samples
                return zeros
            start_ptr = (self._write_idx - self._filled) % buffer_len
            start_offset = actual_start - earliest
            start_pos = (start_ptr + start_offset) % buffer_len
            block = self.viz_buffer.read(start_pos, available_count)
            channel_data = np.array(block[channel_idx], copy=True)
            extracted: np.ndarray = channel_data
            if extracted.size > available_count:
                extracted = extracted[:available_count]
            result = np.empty(window_samples, dtype=np.float32)
            pos = 0
            if missing_prefix > 0:
                result[:missing_prefix] = 0.0
                pos = missing_prefix
            if extracted.size:
                result[pos : pos + extracted.size] = extracted
                pos += extracted.size
            if pos < window_samples:
                result[pos:] = 0.0
            if return_info:
                return result, missing_prefix, missing_suffix
            return result

    def _ensure_viz_buffer_locked(self, channels: int, *, min_capacity: int = 1) -> None:
        channels = max(1, int(channels))
        target_capacity = self._compute_viz_capacity()
        target_capacity = max(target_capacity, int(min_capacity))
        current = self.viz_buffer
        if (
            current is None
            or current.shape[0] != channels
            or current.capacity < target_capacity
        ):
            self.viz_buffer = SharedRingBuffer((channels, target_capacity), dtype=np.float32)
            self._reset_viz_counters_locked()

    def _update_viz_bookkeeping_locked(self, frames: int, start_sample: int, start_idx: int) -> None:
        if frames <= 0:
            return
        capacity = self.viz_buffer.capacity
        if capacity <= 0:
            return
        # If we have a gap in samples, we must reset the visualization buffer
        # to avoid plotting a straight line across the gap.
        # Detect gaps - STRICT MODE: This should NEVER happen in lossless mode
        if self._filled > 0 and start_sample != self._latest_sample_index + 1:
            gap_size = start_sample - (self._latest_sample_index + 1)
            print(f"\n!!! SAMPLE GAP DETECTED !!!")
            print(f"!!! Expected sample: {self._latest_sample_index + 1}, Got: {start_sample}, Gap: {gap_size} samples !!!")
            print(f"!!! This indicates data loss or reordering - LOSSLESS CONSTRAINT VIOLATED !!!")
            self._stats.sample_gaps += 1
            self._reset_viz_counters_locked()
            raise RuntimeError(f"Sample gap detected: expected {self._latest_sample_index + 1}, got {start_sample} (gap={gap_size})")
            
        end_sample = start_sample + frames - 1
        self._write_idx = (start_idx + frames) % capacity
        self._filled = min(capacity, self._filled + frames)
        self._latest_sample_index = end_sample

    def _reset_viz_counters_locked(self) -> None:
        self._write_idx = 0
        self._filled = 0
        self._latest_sample_index = -1
        # if self.viz_buffer is not None:
        #     self.viz_buffer.clear() # SharedRingBuffer has no clear method, but resetting _filled is enough

    def _compute_viz_capacity(self) -> int:
        if self._sample_rate is not None and self._sample_rate > 0:
            sr = float(self._sample_rate)
            window_len = int(max(sr * max(self._window_sec, 1e-3) * 2, 1))
            thirty_sec = int(max(sr * 30.0, 1))
            return max(window_len, thirty_sec)
        return max(1, self.viz_buffer.capacity if self.viz_buffer is not None else 1)

    def _empty_payload(self, status: Optional[dict] = None) -> dict:
        if status is None:
            status = {
                "sample_rate": float(self._sample_rate or 0.0),
                "window_sec": float(self._window_sec),
            }
        return {
            "channel_ids": [],
            "channel_names": [],
            "samples": np.zeros((0, 0), dtype=np.float32),
            "times": np.zeros(0, dtype=np.float32),
            "status": status,
        }
