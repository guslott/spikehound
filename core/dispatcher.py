from __future__ import annotations

import queue
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from PySide6 import QtCore

from .conditioning import FilterSettings, SignalConditioner
from .models import Chunk, EndOfStream, TriggerConfig


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
    """Router thread that conditions incoming chunks and fans them out to consumers."""

    def __init__(
        self,
        raw_queue: "queue.Queue[Chunk | EndOfStream]",
        visualization_queue: "queue.Queue[Chunk | EndOfStream]",
        audio_queue: "queue.Queue[Chunk | EndOfStream]",
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
        self._ring_buffer: Optional[np.ndarray] = None
        self._buffer_len: int = 0
        self._write_idx: int = 0
        self._filled: int = 0
        self._sample_rate: Optional[float] = None
        self._window_sec: float = 0.2
        self._channel_names: tuple[str, ...] = tuple()
        self._channel_ids: tuple[int, ...] = tuple()
        self._channel_index_map: Dict[int, int] = {}
        self._current_trigger: Optional[TriggerConfig] = None
        self._active_channel_ids: list[int] = []
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

    def clear_active_channels(self) -> None:
        with self._ring_lock:
            self._active_channel_ids = []
            self._filled = 0

    def set_channel_layout(self, channel_ids: Sequence[int], channel_names: Sequence[str]) -> None:
        with self._ring_lock:
            self._channel_ids = tuple(channel_ids)
            self._channel_names = tuple(channel_names)
            self._channel_index_map = {cid: idx for idx, cid in enumerate(self._channel_ids)}
            self._ring_buffer = None
            self._buffer_len = 0
            self._write_idx = 0
            self._filled = 0

    def reset_buffers(self) -> None:
        with self._ring_lock:
            if self._ring_buffer is not None:
                self._ring_buffer.fill(0.0)
            self._write_idx = 0
            self._filled = 0

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
                self._sample_rate = sample_rate
                self._ensure_buffer_capacity_locked()
            self._window_sec = max(config.window_sec, 1e-3)
            self._ensure_buffer_capacity_locked()

    def set_window_duration(self, window_sec: float) -> None:
        if window_sec <= 0:
            return
        with self._ring_lock:
            self._window_sec = window_sec
            self._ensure_buffer_capacity_locked()

    def snapshot(self) -> Dict[str, object]:
        with self._stats_lock:
            return self._stats.snapshot()

    def buffer_status(self) -> Dict[str, float]:
        with self._ring_lock:
            filled = float(self._filled)
            capacity = float(self._buffer_len)
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

                if not isinstance(item, Chunk):
                    continue

                filtered_chunk = self._process_chunk(item)
                self._fan_out(item, filtered_chunk)
            finally:
                self._raw_queue.task_done()

    def _process_chunk(self, chunk: Chunk) -> Chunk:
        with self._stats_lock:
            self._stats.received += 1

        filtered_samples = self._conditioner.process(chunk)
        filters_active = self._conditioner.settings.any_enabled()
        meta: Dict[str, object] = dict(chunk.meta or {})
        meta.setdefault("start_sample", chunk.seq * chunk.n_samples)
        meta["filtered"] = filters_active
        meta["filters"] = self._conditioner.describe()

        filtered_chunk = Chunk(
            samples=filtered_samples,
            start_time=chunk.start_time,
            dt=chunk.dt,
            seq=chunk.seq,
            channel_names=chunk.channel_names,
            units=chunk.units,
            meta=meta,
        )
        self._update_ring_buffer(filtered_chunk)
        with self._stats_lock:
            self._stats.processed += 1
        return filtered_chunk

    def _fan_out(self, raw_chunk: Chunk, filtered_chunk: Chunk) -> None:
        self._enqueue(self._logging_queue, raw_chunk, "logging")
        for name, out_queue in self._output_queues.items():
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
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            with self._stats_lock:
                self._stats.evicted[queue_name] += 1
            try:
                _ = target_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                target_queue.put_nowait(item)
            except queue.Full:
                with self._stats_lock:
                    self._stats.dropped[queue_name] += 1
            else:
                with self._stats_lock:
                    self._stats.forwarded[queue_name] += 1
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
            payload = self._collect_window_payload()
            if payload is not None:
                self.signals.tick.emit(payload)
            self._stop_event.wait(self._tick_interval)

    def _collect_window_payload(self) -> Optional[dict]:
        with self._ring_lock:
            status = {
                "sample_rate": float(self._sample_rate or 0.0),
                "window_sec": float(self._window_sec),
            }
            if self._ring_buffer is None or self._sample_rate is None:
                return self._empty_payload(status)
            if self._filled == 0:
                return self._empty_payload(status)
            mode = "continuous"
            if self._current_trigger is not None:
                mode = str(self._current_trigger.mode)
            if mode not in ("continuous", "stream", "single"):
                return self._empty_payload(status)

            window_samples = int(max(1, min(self._buffer_len, self._window_sec * self._sample_rate)))
            window_samples = min(window_samples, self._filled)
            if window_samples <= 0:
                return self._empty_payload(status)

            start = (self._write_idx - window_samples) % self._buffer_len
            if start + window_samples <= self._buffer_len:
                data = self._ring_buffer[:, start : start + window_samples].copy()
            else:
                first = self._buffer_len - start
                part1 = self._ring_buffer[:, start:].copy()
                part2 = self._ring_buffer[:, : window_samples - first].copy()
                data = np.concatenate((part1, part2), axis=1)

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
            actual_window_sec = window_samples / self._sample_rate if self._sample_rate else 0.0
            times = np.linspace(0.0, actual_window_sec, window_samples, endpoint=False, dtype=np.float32)
            return {
                "channel_ids": channel_ids,
                "channel_names": channel_names,
                "samples": data,
                "times": times,
                "status": status,
            }

    def _update_ring_buffer(self, chunk: Chunk) -> None:
        with self._ring_lock:
            self._ensure_buffer_for_chunk_locked(chunk)
            if self._ring_buffer is None:
                return

            data = chunk.samples
            frames = data.shape[1]
            buffer_len = self._buffer_len

            if not self._channel_ids or len(self._channel_ids) != data.shape[0]:
                self._channel_ids = tuple(self._active_channel_ids) if self._active_channel_ids else tuple(range(data.shape[0]))
                self._channel_index_map = {cid: idx for idx, cid in enumerate(self._channel_ids)}

            if frames >= buffer_len:
                self._ring_buffer[:, :] = data[:, -buffer_len:]
                self._write_idx = 0
                self._filled = buffer_len
                return

            end = self._write_idx + frames
            if end <= buffer_len:
                self._ring_buffer[:, self._write_idx:end] = data
            else:
                first = buffer_len - self._write_idx
                self._ring_buffer[:, self._write_idx:] = data[:, :first]
                self._ring_buffer[:, : frames - first] = data[:, first:]
            self._write_idx = (self._write_idx + frames) % buffer_len
            self._filled = min(buffer_len, self._filled + frames)

    def _ensure_buffer_for_chunk_locked(self, chunk: Chunk) -> None:
        prev_sample_rate = self._sample_rate
        self._sample_rate = 1.0 / chunk.dt if chunk.dt > 0 else self._sample_rate
        channel_names = tuple(chunk.channel_names)
        channels = len(channel_names)
        if channels == 0 or self._sample_rate is None:
            return

        resize_needed = False
        if self._ring_buffer is None:
            resize_needed = True
        else:
            if self._ring_buffer.shape[0] != channels:
                resize_needed = True
            elif prev_sample_rate is not None and abs(prev_sample_rate - self._sample_rate) > 1e-6:
                resize_needed = True

        if not self._channel_names:
            self._channel_names = channel_names
        if resize_needed:
            desired_len = self._compute_buffer_len(self._sample_rate)
            self._ring_buffer = np.zeros((channels, desired_len), dtype=np.float32)
            self._buffer_len = desired_len
            self._write_idx = 0
            self._filled = 0
        else:
            self._ensure_buffer_capacity_locked()

    def _ensure_buffer_capacity_locked(self) -> None:
        if self._ring_buffer is None or self._sample_rate is None:
            return
        desired_len = self._compute_buffer_len(self._sample_rate)
        if desired_len <= self._buffer_len:
            return
        channels = self._ring_buffer.shape[0]
        recent = self._collect_recent_locked(min(self._filled, desired_len))
        new_buffer = np.zeros((channels, desired_len), dtype=np.float32)
        if recent.size:
            new_buffer[:, -recent.shape[1] :] = recent
            self._filled = recent.shape[1]
            self._write_idx = self._filled % desired_len
        else:
            self._filled = 0
            self._write_idx = 0
        self._ring_buffer = new_buffer
        self._buffer_len = desired_len

    def _collect_recent_locked(self, count: int) -> np.ndarray:
        if self._ring_buffer is None or count <= 0 or self._filled == 0:
            return np.empty((0, 0), dtype=np.float32)
        count = min(count, self._filled)
        start = (self._write_idx - count) % self._buffer_len
        if start + count <= self._buffer_len:
            return self._ring_buffer[:, start : start + count].copy()
        first = self._buffer_len - start
        part1 = self._ring_buffer[:, start:].copy()
        part2 = self._ring_buffer[:, : count - first].copy()
        return np.concatenate((part1, part2), axis=1)

    def _compute_buffer_len(self, sample_rate: float) -> int:
        max_window = max(self._window_sec, 1e-3)
        return max(int(sample_rate * max_window * 2), 1)

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
