from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from core.models import Chunk, EndOfStream


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
