from __future__ import annotations

import queue
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional

from .conditioning import FilterSettings, SignalConditioner
from .models import Chunk, EndOfStream


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


class Dispatcher:
    """Router thread that conditions incoming chunks and fans them out to consumers."""

    def __init__(
        self,
        raw_queue: "queue.Queue[Chunk | EndOfStream]",
        visualization_queue: "queue.Queue[Chunk | EndOfStream]",
        analysis_queue: "queue.Queue[Chunk | EndOfStream]",
        audio_queue: "queue.Queue[Chunk | EndOfStream]",
        logging_queue: "queue.Queue[Chunk | EndOfStream]",
        *,
        filter_settings: Optional[FilterSettings] = None,
        poll_timeout: float = 0.05,
    ) -> None:
        self._raw_queue = raw_queue
        self._output_queues: Dict[str, queue.Queue] = {
            "visualization": visualization_queue,
            "analysis": analysis_queue,
            "audio": audio_queue,
        }
        self._logging_queue = logging_queue
        self._conditioner = SignalConditioner(filter_settings)
        self._poll_timeout = poll_timeout

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stats = DispatcherStats()
        self._stats_lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="DispatcherThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def update_filter_settings(self, settings: FilterSettings) -> None:
        self._conditioner.update_settings(settings)

    def snapshot(self) -> Dict[str, object]:
        with self._stats_lock:
            return self._stats.snapshot()

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
        with self._stats_lock:
            self._stats.processed += 1
        return filtered_chunk

    def _fan_out(self, raw_chunk: Chunk, filtered_chunk: Chunk) -> None:
        self._enqueue(self._logging_queue, raw_chunk, "logging")
        for name, out_queue in self._output_queues.items():
            self._enqueue(out_queue, filtered_chunk, name)

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
