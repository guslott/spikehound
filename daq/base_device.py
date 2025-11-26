from __future__ import annotations

"""
Base classes and datamodels for streaming DAQ devices.

Goals:
- Simple, stable contract for GUI/consumers (ChunkPointers over a queue).
- Clean lifecycle: open → configure → start/stop → close.
- Capability discovery for device pickers and validation.
- Consistent timebase and sequencing across all drivers.
- Centralized queue/backpressure/drop-oldest behavior.

Subclasses implement the *_impl() methods to integrate real hardware
(or simulators) while relying on the shared utilities here.
"""

import queue
import threading
import time as _time
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Literal, Optional, Sequence

import numpy as np

from shared.models import (
    Chunk,
    ChunkPointer,
    ActualConfig,
    Capabilities,
    ChannelInfo,
    DeviceInfo,
)
from shared.ring_buffer import SharedRingBuffer


# ----------------------------
# Data model (shared contract)
# ----------------------------

# ----------------------------
# Base class
# ----------------------------

State = Literal["closed", "open", "running"]


class BaseDevice(ABC):
    """
    Abstract base for all DAQ devices (sources/sinks).

    Typical flow:
        devs = Driver.list_available_devices()
        device = Driver()
        device.open(devs[0].id)
        caps = device.get_capabilities(devs[0].id)
        chans = device.list_available_channels(devs[0].id)
        device.configure(sample_rate=20_000, channels=[c.id for c in chans[:2]], chunk_size=1024)
        device.start()
        # Consume device.data_queue (ChunkPointer objects) in the GUI thread
        device.stop()
        device.close()
    """

    @classmethod
    @abstractmethod
    def device_class_name(cls) -> str:
        """Return the human-friendly category name for this driver type."""
        raise NotImplementedError

    # ---- Lifecycle ---------------------------------------------------------

    def __init__(self, queue_maxsize: int = 64) -> None:
        self.data_queue: "queue.Queue[ChunkPointer]" = queue.Queue(maxsize=queue_maxsize)
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
                except Exception:
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
            # Allocate a shared ring buffer sized for ~30 seconds of data
            capacity = max(int(actual.sample_rate * 30), actual.chunk_size)
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
                dtype = rb._data.dtype  # type: ignore[attr-defined]  # internal dtype reuse
                self.ring_buffer = SharedRingBuffer(shape=(desired_channels, capacity), dtype=dtype)
                
                # CRITICAL: Clear the queue of any pointers to the old buffer
                # This prevents the consumer from trying to read from a replaced/closed buffer
                with self.data_queue.mutex:
                    self.data_queue.queue.clear()

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
        meta: Optional[dict[str, Any]] = None,
    ) -> ChunkPointer:
        """
        Write a (frames, channels) array into the shared ring buffer and enqueue a ChunkPointer.

        Parameters
        ----------
        data : np.ndarray
            Samples shaped (frames, channels) as produced by the driver callback.
        device_time : float | None
            Optional hardware clock reference for the first frame (reserved for future use).
        mono_time : float | None
            Override for the host monotonic timestamp; defaults to `time.monotonic()`.
        meta : dict[str, Any] | None
            Reserved for future metadata propagation (currently unused).

        Returns
        -------
        ChunkPointer
            Pointer into the shared ring buffer for downstream consumers.
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

        # Validate shape
        if data.ndim != 2:
            raise ValueError("data must be 2D array shaped (frames, channels).")
        frames, chans = data.shape
        expected_chans = len(self._active_channel_ids)
        if expected_chans != chans:
            # Allow drivers to deliver superset and slice here as a fallback.
            with self._channel_lock:
                if chans >= expected_chans and chans == len(self._available_channels):
                    # Slice by active channel order
                    idx = [self._index_of_channel_id(cid) for cid in self._active_channel_ids]
                    data = data[:, idx]
                    frames, chans = data.shape
                elif chans < expected_chans:
                    actual_ids = None
                    if meta is not None:
                        actual_ids = meta.get("active_channel_ids")
                    if isinstance(actual_ids, Sequence):
                        out = np.zeros((frames, expected_chans), dtype=data.dtype)
                        id_to_col = {cid: i for i, cid in enumerate(actual_ids)}
                        for out_idx, cid in enumerate(self._active_channel_ids):
                            src_idx = id_to_col.get(cid)
                            if src_idx is not None and src_idx < chans:
                                out[:, out_idx] = data[:, src_idx]
                        data = out
                        chans = data.shape[1]
                    else:
                        raise ValueError(
                            f"data has {chans} channels, expected {expected_chans}."
                        )
                else:
                    raise ValueError(
                        f"data has {chans} channels, expected {expected_chans}."
                    )

        if frames == 0:
            raise ValueError("data must contain at least one frame")

        # Stamp times/counters
        mono = _time.monotonic() if mono_time is None else mono_time
        with self._state_lock:
            # Advance counters for diagnostics/compatibility with existing stats
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
            self._next_start_sample += frames
            self._next_seq += 1
        start_index = self.ring_buffer.write(np.ascontiguousarray(chunk.samples))
        pointer = ChunkPointer(
            start_index=start_index,
            length=frames,
            render_time=chunk.start_time,
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
            # Drain any stale data before starting a new run
            try:
                while True:
                    self.data_queue.get_nowait()
            except queue.Empty:
                pass

    def _safe_put(self, ptr: ChunkPointer) -> None:
        """Enqueue chunk pointer with BLOCKING to ensure lossless data flow."""
        try:
            # BLOCKING MODE: Wait up to 10 seconds. If still full, something is seriously wrong.
            self.data_queue.put(ptr, block=True, timeout=10.0)
        except queue.Full:
            # This should NEVER happen with blocking mode - indicates deadlock or stuck consumer
            print(f"\n!!! CRITICAL ERROR: Device data_queue BLOCKED for 10+ seconds !!!")
            print(f"!!! Consumer (Dispatcher) is not keeping up - system deadlocked !!!")
            raise RuntimeError("Device data_queue blocked - lossless constraint violated")

    def _assert_state(self, expected: Iterable[State]) -> None:
        if self._state not in expected:
            raise RuntimeError(f"Invalid state: {self._state}; expected one of {tuple(expected)}.")

    def _index_of_channel_id(self, cid: int) -> int:
        # Linear search is fine for small channel counts typical of audio/USB DAQ
        for i, ch in enumerate(self._available_channels):
            if ch.id == cid:
                return i
        raise ValueError(f"Channel id {cid} not in available channels.")
