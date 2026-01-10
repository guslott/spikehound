from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import logging
import queue
import numpy as np


def _freeze_array(array: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    """Return a read-only, C-contiguous view or copy of `array`, validating dimensions."""
    # Only copy if we must to achieve C-contiguity or if it's not already a numpy array
    arr = np.asanyarray(array, order="C")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"array must be {ndim}D, got {arr.ndim}D")
    
    # Return a read-only view
    view = arr.view()
    view.setflags(write=False)
    return view


def _copy_mapping(mapping: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise TypeError("meta/properties/params must be a mapping type")
    if isinstance(mapping, MutableMapping):
        return dict(mapping)
    return dict(mapping)


# ----------------------------
# Device / Channel metadata
# ----------------------------

@dataclass(frozen=True)
class DeviceInfo:
    """A discoverable input device."""

    id: str
    name: str
    vendor: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelInfo:
    """A single input channel."""

    id: int
    name: str
    units: str = "V"
    range: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class Capabilities:
    """What a device can do for input."""

    max_channels_in: int
    sample_rates: Optional[List[int]]
    dtype: str = "float32"
    notes: Optional[str] = None


@dataclass(frozen=True)
class ActualConfig:
    """Configuration achieved after a driver configures the device."""

    sample_rate: int
    channels: List[ChannelInfo]
    chunk_size: int
    latency_s: Optional[float] = None
    dtype: str = "float32"


# ----------------------------
# Streaming data models
# ----------------------------

@dataclass(frozen=True)
class Chunk:
    """Atomic unit of streaming data passed between threads."""

    samples: np.ndarray
    start_time: float
    dt: float
    seq: int
    channel_names: Tuple[str, ...]
    units: str
    meta: Optional[Mapping[str, Any]] = field(default=None)

    def __post_init__(self) -> None:
        if self.dt <= 0 or not np.isfinite(self.dt):
            raise ValueError("dt must be positive and finite")
        if self.seq < 0:
            raise ValueError("seq must be non-negative")
        if not self.channel_names:
            raise ValueError("channel_names must not be empty")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("units must be a non-empty string")

        samples = _freeze_array(self.samples, ndim=2)
        if samples.shape[0] != len(self.channel_names):
            raise ValueError("samples shape mismatch: axis 0 must match len(channel_names)")

        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "channel_names", tuple(self.channel_names))
        object.__setattr__(self, "meta", _copy_mapping(self.meta))

    @property
    def n_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples * self.dt

    def __reduce__(self):
        # Ensure __post_init__ runs on unpickle to re-freeze arrays
        return (
            self.__class__,
            (
                self.samples,
                self.start_time,
                self.dt,
                self.seq,
                self.channel_names,
                self.units,
                self.meta,
            ),
        )




@dataclass(frozen=True)
class ChunkPointer:
    """Lightweight pointer to data stored in a SharedRingBuffer.
    
    Carries sequencing metadata from the DAQ layer:
    - seq: Monotonically increasing sequence number (reset per run)
    - start_sample: Global sample index for the first sample in this chunk
    - device_time: Optional hardware timestamp (seconds) if provided by driver
    """

    start_index: int
    length: int
    render_time: float
    seq: int
    start_sample: int
    device_time: float | None = None

    def __post_init__(self) -> None:
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.render_time < 0:
            raise ValueError("render_time must be non-negative")
        if self.seq < 0:
            raise ValueError("seq must be non-negative")
        if self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")


@dataclass(frozen=True)
class DetectionEvent:
    """Canonical event type emitted by detectors in the detection pipeline.
    
    IMPORTANT: This is the ONLY event type that detectors should emit.
    AnalysisEvent is derived from DetectionEvent in the analysis layer.
    
    Conversion Flow:
        Detector emits DetectionEvent → AnalysisWorker converts to AnalysisEvent
    
    This type is intentionally lightweight to minimize overhead in the detection
    hot path. The analysis layer enriches it with timing metadata and metrics.
    
    Attributes:
        t: Timestamp of the event (seconds since stream start)
        chan: Channel index where the event was detected
        window: Waveform samples around the detection point
        properties: Computed metrics (e.g., amplitude, energy)
        params: Detection parameters used (e.g., threshold value)
    
    See Also:
        shared.types.AnalysisEvent: Enriched event type for GUI display
        analysis.analysis_worker.detection_to_analysis_event: Conversion function
    """

    t: float
    chan: int
    window: np.ndarray
    properties: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t < 0 or not np.isfinite(self.t):
            raise ValueError("t must be non-negative and finite")
        if self.chan < 0:
            raise ValueError("chan must be non-negative")

        window = _freeze_array(self.window)

        object.__setattr__(self, "window", window)
        object.__setattr__(self, "properties", _copy_mapping(self.properties) or {})
        object.__setattr__(self, "params", _copy_mapping(self.params) or {})

    def __reduce__(self):
        # Ensure __post_init__ runs on unpickle to re-freeze arrays
        return (
            self.__class__,
            (self.t, self.chan, self.window, self.properties, self.params),
        )




def _restore_end_of_stream() -> "_EndOfStreamSentinel":
    return EndOfStream


class _EndOfStreamSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return "EndOfStream"

    def __reduce__(self):
        return (_restore_end_of_stream, ())


EndOfStream = _EndOfStreamSentinel()


_VALID_TRIGGER_MODES = frozenset({"continuous", "stream", "single"})


@dataclass(frozen=True)
class TriggerConfig:
    """Trigger parameters shared with the dispatcher/analyzer layer."""

    channel_index: int | None
    threshold: float
    hysteresis: float
    pretrigger_frac: float
    window_sec: float
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in _VALID_TRIGGER_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_TRIGGER_MODES)}, got {self.mode!r}"
            )
        if self.window_sec <= 0:
            raise ValueError("window_sec must be positive")
        if not (0.0 <= self.pretrigger_frac <= 1.0):
            raise ValueError("pretrigger_frac must be in [0, 1]")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not np.isfinite(self.hysteresis):
            raise ValueError("hysteresis must be finite")
        
        # Validation for channel selection
        if self.mode in ("single", "continuous"):
            if self.channel_index is None:
                raise ValueError(f"channel_index must be specified for mode {self.mode!r}")
            if self.channel_index < 0:
                raise ValueError("channel_index must be non-negative")
        else:
            # In stream mode, channel_index can be None or non-negative
            if self.channel_index is not None and self.channel_index < 0:
                raise ValueError("channel_index must be non-negative if specified")





# ----------------------------
# Queue & Backpressure Policies
# ----------------------------

logger = logging.getLogger(__name__)

# Explicit backpressure policies for each queue.
# - "lossless": blocks until space available (fail loudly if timeout)
# - "drop-newest": drops incoming item if queue is full
# - "drop-oldest": evicts oldest item to make room for new one
QUEUE_POLICIES: dict[str, str] = {
    "visualization": "drop-oldest",
    "audio": "drop-oldest",
    "logging": "lossless",
    "analysis": "drop-oldest",
    "events": "drop-newest",
    "daq": "lossless",  # DAQ -> Dispatcher is always lossless
}


def enqueue_with_policy(
    queue_name: str, 
    target_queue: queue.Queue, 
    item: object, 
    *,
    stats_callback: Optional[Callable[[str, str], None]] = None
) -> None:
    """Unified enqueue method that dispatches on QUEUE_POLICIES.
    
    Args:
        queue_name: Name of the target queue (used for policy lookup and stats)
        target_queue: The queue object to put the item into
        item: The data to enqueue
        stats_callback: Optional callback(queue_name, action) where action is 
                        "forwarded", "dropped", or "evicted".
    """
    policy = QUEUE_POLICIES.get(queue_name, "drop-newest")
    
    # SPECIAL CASE: EndOfStream must never be dropped by drop-newest queues.
    # It signifies shutdown, so we force eviction to make room.
    if item is EndOfStream and policy == "drop-newest":
        policy = "drop-oldest"
    
    if policy == "lossless":
        _enqueue_lossless(target_queue, item, queue_name, stats_callback)
    elif policy == "drop-oldest":
        _enqueue_drop_oldest(target_queue, item, queue_name, stats_callback)
    else:  # "drop-newest" or unknown
        _enqueue_drop_newest(target_queue, item, queue_name, stats_callback)


def _enqueue_lossless(
    target_queue: queue.Queue, 
    item: object, 
    queue_name: str,
    stats_callback: Optional[Callable[[str, str], None]] = None
) -> None:
    """Block until space available; fail loudly if timeout."""
    try:
        target_queue.put(item, block=True, timeout=10.0)
    except queue.Full:
        logger.critical(
            "Queue '%s' BLOCKED for 10+ seconds - downstream too slow",
            queue_name
        )
        if stats_callback:
            stats_callback(queue_name, "dropped")
        raise RuntimeError(f"Queue '{queue_name}' blocked - lossless constraint violated")
    else:
        if stats_callback:
            stats_callback(queue_name, "forwarded")


def _enqueue_drop_newest(
    target_queue: queue.Queue, 
    item: object, 
    queue_name: str,
    stats_callback: Optional[Callable[[str, str], None]] = None
) -> None:
    """Drop the incoming item if the queue is full."""
    try:
        target_queue.put_nowait(item)
    except queue.Full:
        if stats_callback:
            stats_callback(queue_name, "dropped")
    else:
        if stats_callback:
            stats_callback(queue_name, "forwarded")


def _enqueue_drop_oldest(
    target_queue: queue.Queue, 
    item: object, 
    queue_name: str,
    stats_callback: Optional[Callable[[str, str], None]] = None
) -> None:
    """Evict oldest items until the new item fits."""
    placed = False
    while not placed:
        try:
            target_queue.put_nowait(item)
            placed = True
        except queue.Full:
            try:
                _ = target_queue.get_nowait()
                if stats_callback:
                    stats_callback(queue_name, "evicted")
            except queue.Empty:
                pass
    if stats_callback:
        stats_callback(queue_name, "forwarded")


__all__ = [
    "DeviceInfo",
    "ChannelInfo",
    "Capabilities",
    "ActualConfig",
    "Chunk",
    "ChunkPointer",
    "DetectionEvent",

    "EndOfStream",
    "TriggerConfig",
    
    "QUEUE_POLICIES",
    "enqueue_with_policy",
]
