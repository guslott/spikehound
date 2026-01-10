from __future__ import annotations

"""
Base classes and datamodels for streaming DAQ sources.

Goals:
- Simple, stable contract for GUI/consumers (ChunkPointers over a queue).
- Clean lifecycle: open → configure → start/stop → close.
- Capability discovery for device pickers and validation.
- Consistent timebase and sequencing across all drivers.
- Centralized queue/backpressure/drop-oldest behavior.

Subclasses implement the *_impl() methods to integrate real hardware
(or simulators) while relying on the shared utilities here.
"""

import logging
import queue
import threading
import time as _time
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Literal, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

from shared.models import (
    ActualConfig,
    Capabilities,
    ChannelInfo,
    Chunk,
    ChunkPointer,
    DeviceInfo,
    EndOfStream,
    QueueName,
    enqueue_with_policy,
)
from shared.ring_buffer import SharedRingBuffer


# Default ring buffer duration in seconds. The buffer must be large enough that
# consumers can read pointers before the buffer wraps. 30 seconds provides ample
# headroom for typical visualization/analysis latencies. Increase if observing
# "wrap risk" warnings in the Health Metrics panel.
RING_BUFFER_DURATION_SEC: float = 30.0


# ----------------------------
# Data model (shared contract)
# ----------------------------

# ----------------------------
# Base class
# ----------------------------

State = Literal["closed", "open", "running"]


class BaseDevice(ABC):
    """
    Abstract base for all DAQ input sources.

    Typical flow:
        devs = Driver.list_available_devices()
        source = Driver()
        source.open(devs[0].id)
        caps = source.get_capabilities(devs[0].id)
        chans = source.list_available_channels(devs[0].id)
        source.configure(sample_rate=20_000, channels=[c.id for c in chans[:2]], chunk_size=1024)
        source.start()
        # Consume source.data_queue (ChunkPointer objects) in the GUI thread
        source.stop()
        source.close()
    """

    @classmethod
    @abstractmethod
    def device_class_name(cls) -> str:
        """Return the human-friendly category name for this driver type."""
        raise NotImplementedError

    # ---- Lifecycle ---------------------------------------------------------

    def __init__(self, queue_maxsize: int = 64) -> None:
        self.data_queue: "queue.Queue[ChunkPointer | EndOfStream]" = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._channel_lock = threading.RLock()

        self._state: State = "closed"
        self._device_id: Optional[str] = None
        self._available_channels: List[ChannelInfo] = []  # Full device channel list
        self._active_channel_ids: List[int] = []          # Ordered selection for streaming

        # Run-level counters (reset at each start)
        self._next_seq: int = 0
        self._next_start_sample: int = 0

        # Diagnostics
        self._xruns: int = 0
        self._drops: int = 0

        # Dtype contract: drivers should deliver float32 unless otherwise negotiated
        self._dtype: str = "float32"
        self.config: Optional[ActualConfig] = None
        self.ring_buffer: Optional[SharedRingBuffer] = None

    # ------------------------
    # Device enumeration APIs
    # ------------------------

    @classmethod
    @abstractmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        """Return all input-capable devices visible to the driver."""
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self, device_id: str) -> Capabilities:
        """Return input capabilities for a given device (max channels, rates, dtype)."""
        raise NotImplementedError

    @abstractmethod
    def list_available_channels(self, device_id: str) -> List[ChannelInfo]:
        """Return all input channels for the device, in hardware order."""
        raise NotImplementedError

    # ----------
    # Lifecycle
    # ----------

    def open(self, device_id: str) -> None:
        with self._state_lock:
            self._assert_state(expected=("closed",))
            self._open_impl(device_id)
            self._device_id = device_id
            # Cache channels for convenience / selection APIs
            self._available_channels = self.list_available_channels(device_id)
            # Default active channels = all, in order
            self._active_channel_ids = [ch.id for ch in self._available_channels]
            self._state = "open"

    @abstractmethod
    def _open_impl(self, device_id: str) -> None:
        """Driver-specific resource acquisition."""
        raise NotImplementedError

    def close(self) -> None:
        with self._state_lock:
            if self._state == "running":
                # Be resilient: stop if still running.
                self._stop_impl_safe()
            self._close_impl()
            self._device_id = None
            self._available_channels = []
            self._active_channel_ids = []
            self.config = None
            self.ring_buffer = None
            self._state = "closed"
            self._xruns = 0
            self._drops = 0

    @abstractmethod
    def _close_impl(self) -> None:
        """Driver-specific resource release."""
        raise NotImplementedError

    # -------------
    # Configuration
    # -------------

    def configure(
        self,
        sample_rate: int,
        channels: Optional[Sequence[int]] = None,
        chunk_size: int = 1024,
        **options: Any,
    ) -> ActualConfig:
        """
        Apply input configuration. May be called multiple times while open or stopped.
        Returns the actual configuration achieved by the driver.
        """
        with self._state_lock:
            self._assert_state(expected=("open",))
            # Normalize channel list
            if channels is None or len(channels) == 0:
                channels = [ch.id for ch in self._available_channels]
            self.set_active_channels(channels)

            # Validate requested sample rate against capabilities when provided
            if self._device_id is not None:
                try:
                    caps = self.get_capabilities(self._device_id)
                except Exception as exc:
                    logger.debug("Failed to get capabilities for %s: %s", self._device_id, exc)
                    caps = None
                if caps is not None and caps.sample_rates:
                    if int(sample_rate) not in [int(sr) for sr in caps.sample_rates]:
                        raise ValueError(f"sample_rate {sample_rate} not in supported set {caps.sample_rates}")

            # Clamp or normalize chunk_size minimally (positive int)
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer")

            actual = self._configure_impl(
                sample_rate=sample_rate,
                channels=list(self._active_channel_ids),
                chunk_size=chunk_size,
                **options,
            )
            # Cache dtype preference for validators/emit helpers
            self._dtype = actual.dtype
            # Store the echo so GUI can query it directly
            self.config = actual
            # Allocate a shared ring buffer sized for RING_BUFFER_DURATION_SEC seconds
            capacity = max(int(actual.sample_rate * RING_BUFFER_DURATION_SEC), actual.chunk_size)
            if capacity <= 0:
                raise ValueError("computed ring buffer capacity must be positive")
            n_channels = len(self._active_channel_ids)
            if n_channels <= 0:
                raise ValueError("configuration must include at least one channel")
            self.ring_buffer = SharedRingBuffer(
                shape=(n_channels, capacity),
                dtype=actual.dtype,
            )
            return actual

    @abstractmethod
    def _configure_impl(
        self,
        sample_rate: int,
        channels: Sequence[int],
        chunk_size: int,
        **options: Any,
    ) -> ActualConfig:
        """Driver-specific configuration. Should not start streaming."""
        raise NotImplementedError

    # ---- Run control ----

    def start(self) -> None:
        """Begin streaming. Resets run counters and queue timing origin."""
        with self._state_lock:
            if self._state == "running":
                return
            self._assert_state(expected=("open",))
            if self.config is None:
                raise RuntimeError("configure() must be called before start().")
            self._reset_counters()
            self._stop_event.clear()
            self._start_impl()
            self._state = "running"

    @abstractmethod
    def _start_impl(self) -> None:
        """Driver-specific start. Emit data by calling self.emit_array(...) or self.emit_chunk(...)."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop streaming; device remains open and can be reconfigured or restarted."""
        with self._state_lock:
            if self._state == "running":
                self._stop_impl_safe()
                self._state = "open"

    def _stop_impl_safe(self) -> None:
        # Signal cooperative loops and call driver stop.
        self._stop_event.set()
        try:
            self._stop_impl()
        finally:
            # Do not clear the stop_event here; it is reset on next start()
            pass

    @abstractmethod
    def _stop_impl(self) -> None:
        """Driver-specific stop."""
        raise NotImplementedError

    # -----------------
    # Channel selection
    # -----------------

    def set_active_channels(self, channel_ids: Sequence[int]) -> None:
        """
        Atomically set active channels (order matters).
        Subclasses that cannot remap at the driver level may still slice before emit.
        """
        with self._channel_lock:
            available_ids = {c.id for c in self._available_channels}
            missing = [cid for cid in channel_ids if cid not in available_ids]
            if missing:
                raise ValueError(f"Unknown channel ids: {missing}")
            # Maintain order and uniqueness
            uniq: List[int] = []
            for cid in channel_ids:
                if cid not in uniq:
                    uniq.append(cid)
            self._active_channel_ids = uniq
            # If a ring buffer already exists and the channel count changed, rebuild it
            rb = self.ring_buffer
            desired_channels = len(self._active_channel_ids)
            if rb is not None and desired_channels > 0 and rb.shape[0] != desired_channels:
                capacity = rb.capacity
                dtype = rb.dtype
                self.ring_buffer = SharedRingBuffer(shape=(desired_channels, capacity), dtype=dtype)
                
                # Clear the queue of any pointers to the old buffer
                # This prevents the consumer from trying to read from a replaced/closed buffer
                # Use public API (get_nowait drain) instead of internal queue.mutex/queue.clear()
                try:
                    while True:
                        self.data_queue.get_nowait()
                except queue.Empty:
                    pass

    def get_active_channels(self) -> List[ChannelInfo]:
        with self._channel_lock:
            idset = set(self._active_channel_ids)
            # Preserve the order chosen by the user
            id_to_info = {ch.id: ch for ch in self._available_channels}
            return [id_to_info[cid] for cid in self._active_channel_ids if cid in idset]

    # --------------
    # Emit utilities
    # --------------

    def emit_array(
        self,
        data: np.ndarray,
        *,
        device_time: Optional[float] = None,
        mono_time: Optional[float] = None,
    ) -> ChunkPointer:
        """Helper for simple drivers to emit a 2D float32 array.
        
        Args:
            data: np.ndarray of shape (frames, channels) or (channels, frames).
                  If (frames, channels) and channels matches config, it is transposed.
            device_time: Optional hardware timestamp (seconds)
            mono_time: Optional host monotonic time (recommended if available)
        """
        if self.config is None:
            raise RuntimeError("emit_array() called before configure().")
        if self.ring_buffer is None:
            raise RuntimeError("ring buffer not initialized; call configure() first.")

        # Validate/normalize dtype
        desired_dtype = np.float32 if self._dtype == "float32" else np.dtype(self._dtype)
        if data.dtype != desired_dtype:
            # Conservative: convert but avoid copy if possible
            data = np.asarray(data, dtype=desired_dtype)

        # Stamp times/counters
        mono = _time.monotonic() if mono_time is None else mono_time
        
        with self._state_lock:
            # Validate shape (and possibly adapt) within the lock to ensure synchronization 
            # with active_channel_ids and ring_buffer updates from configure().
            if data.ndim != 2:
                raise ValueError("data must be 2D array shaped (frames, channels).")
            frames, chans = data.shape
            expected_chans = len(self._active_channel_ids)
            
            if expected_chans != chans:
                # Allow drivers to deliver superset and slice here as a fallback.
                with self._channel_lock:
                    expected_chans_now = len(self._active_channel_ids)
                    # Re-check in case channel lock revealed a change (though state_lock should prevent it)
                    if chans >= expected_chans_now and chans == len(self._available_channels):
                        # Slice by active channel order
                        idx = [self._index_of_channel_id(cid) for cid in self._active_channel_ids]
                        data = data[:, idx]
                        frames, chans = data.shape
                    elif chans < expected_chans_now:
                        # If the incoming data has fewer channels than expected,
                        # we assume the driver is providing a subset and we need to
                        # map it to the full expected channel set.
                        # This path is less common and requires more context,
                        # for now, we raise an error if we can't infer the mapping.
                        raise ValueError(
                            f"data has {chans} channels, expected {expected_chans_now}. "
                            "Cannot automatically map subset of channels without metadata."
                        )
                    else:
                        raise ValueError(
                            f"data has {chans} channels, expected {expected_chans_now}."
                        )

            if frames == 0:
                raise ValueError("data must contain at least one frame")

            # Capture current counters BEFORE incrementing for the pointer
            seq = self._next_seq
            start_sample = self._next_start_sample
            self._next_start_sample += frames
            self._next_seq += 1

            # Write channel-major data into the ring buffer
            channel_major = np.ascontiguousarray(data.T)
            rb = self.ring_buffer
            if rb is None:
                raise RuntimeError("ring buffer not initialized; call configure() first.")
            if rb.shape[0] != channel_major.shape[0]:
                raise ValueError(
                    f"ring buffer channel dimension {rb.shape[0]} does not match incoming data {channel_major.shape[0]}"
                )
            start_index = rb.write(channel_major)

            pointer = ChunkPointer(
                start_index=start_index,
                length=frames,
                render_time=mono,
                seq=seq,
                start_sample=start_sample,
                device_time=device_time,
            )
            self._safe_put(pointer)
            
        return pointer

    def emit_chunk(self, chunk: Chunk) -> None:
        """
        Adapt a pre-built Chunk into the ring buffer and enqueue a ChunkPointer.
        """
        if self.ring_buffer is None:
            raise RuntimeError("ring buffer not initialized; call configure() first.")
        frames = chunk.samples.shape[1]
        if frames == 0:
            raise ValueError("chunk must contain at least one frame")
        with self._state_lock:
            # Capture current counters BEFORE incrementing for the pointer
            seq = self._next_seq
            start_sample = self._next_start_sample
            self._next_start_sample += frames
            self._next_seq += 1
            if self.ring_buffer is None:
                raise RuntimeError("ring buffer not initialized; call configure() first.")
            start_index = self.ring_buffer.write(np.ascontiguousarray(chunk.samples))
            # Extract device_time from chunk meta if available
            device_time = None
            if chunk.meta is not None:
                device_time = chunk.meta.get("device_time")
            pointer = ChunkPointer(
                start_index=start_index,
                length=frames,
                render_time=chunk.start_time,
                seq=seq,
                start_sample=start_sample,
                device_time=device_time,
            )
            self._safe_put(pointer)

    def note_xrun(self, count: int = 1) -> None:
        """Drivers can call this when the backend reports over/underruns."""
        self._xruns += max(0, int(count))

    # --------------
    # Introspection
    # --------------

    @property
    def state(self) -> State:
        with self._state_lock:
            return self._state

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def stop_event(self) -> threading.Event:
        """Subclasses may check this in their producer loops for cooperative stop."""
        return self._stop_event

    def stats(self) -> dict[str, Any]:
        """Lightweight diagnostics the GUI can poll occasionally."""
        return {
            "state": self.state,
            "queue_size": self.data_queue.qsize(),
            "queue_maxsize": self.data_queue.maxsize,
            "xruns": self._xruns,
            "drops": self._drops,
            "next_seq": self._next_seq,
            "next_start_sample": self._next_start_sample,
            "sample_rate": None if self.config is None else self.config.sample_rate,
            "active_channels": [ch.id for ch in self.get_active_channels()],
        }

    def get_buffer(self) -> SharedRingBuffer:
        """Expose the shared ring buffer for downstream consumers."""
        if self.ring_buffer is None:
            raise RuntimeError("ring buffer not initialized; call configure() first.")
        return self.ring_buffer

    # -------------
    # Base helpers
    # -------------

    def _reset_counters(self) -> None:
        """Reset seq/start_sample at run start; clears the queue too."""
        with self._state_lock:
            self._next_seq = 0
            self._next_start_sample = 0
            self._xruns = 0
            self._drops = 0
            try:
                while True:
                    self.data_queue.get_nowait()
            except queue.Empty:
                pass

    def _safe_put(self, item: Union[ChunkPointer, type[EndOfStream]]) -> None:
        """Put item into data_queue using the canonical 'daq' lossless policy."""
        try:
            enqueue_with_policy("daq", self.data_queue, item)
        except Exception as exc:
            # If we fail lossless constraint, it's usually a critical error
            # but we log it here since it might be during shutdown.
            logger.error("DAQ _safe_put failed: %s", exc)
            raise

    def _assert_state(self, expected: Iterable[State]) -> None:
        if self._state not in expected:
            raise RuntimeError(f"Invalid state: {self._state}; expected one of {tuple(expected)}.")

    def _index_of_channel_id(self, cid: int) -> int:
        # Linear search is fine for small channel counts typical of audio/USB DAQ
        for i, ch in enumerate(self._available_channels):
            if ch.id == cid:
                return i
        raise ValueError(f"Channel id {cid} not in available channels.")
