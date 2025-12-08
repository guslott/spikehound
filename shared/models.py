from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def _freeze_array(array: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    """Return a read-only, C-contiguous view of `array`, validating dimensions."""
    arr = np.array(array, copy=True, order="C")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"array must be {ndim}D, got {arr.ndim}D")
    arr.setflags(write=False)
    return arr


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
        if self.dt <= 0:
            raise ValueError("dt must be positive")
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
        meta = None if self.meta is None else dict(self.meta)
        return (
            self.__class__,
            (
                np.array(self.samples, copy=True, order="C"),
                self.start_time,
                self.dt,
                self.seq,
                tuple(self.channel_names),
                self.units,
                meta,
            ),
        )


@dataclass(frozen=True)
class ChunkPointer:
    """Lightweight pointer to data stored in a SharedRingBuffer."""

    start_index: int
    length: int
    render_time: float

    def __post_init__(self) -> None:
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.render_time < 0:
            raise ValueError("render_time must be non-negative")


@dataclass(frozen=True)
class Event:
    """Detected feature emitted by the dispatcher/detection layer.
    
    This is the simpler, lower-level Event type used by the detection pipeline.
    For the more detailed analysis-layer Event with timing metadata, see
    `shared.types.Event`.
    
    Attributes:
        t: Timestamp of the event (seconds since stream start)
        chan: Channel index where the event was detected
        window: Waveform samples around the detection point
        properties: Computed metrics (e.g., amplitude, energy)
        params: Detection parameters used (e.g., threshold value)
    
    See Also:
        shared.types.Event: Analysis-layer Event with detailed timing info
    """

    t: float
    chan: int
    window: np.ndarray
    properties: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t < 0:
            raise ValueError("t must be non-negative")
        if self.chan < 0:
            raise ValueError("chan must be non-negative")

        window = _freeze_array(self.window)

        object.__setattr__(self, "window", window)
        object.__setattr__(self, "properties", _copy_mapping(self.properties) or {})
        object.__setattr__(self, "params", _copy_mapping(self.params) or {})

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.t,
                self.chan,
                np.array(self.window, copy=True, order="C"),
                dict(self.properties),
                dict(self.params),
            ),
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


@dataclass(frozen=True)
class TriggerConfig:
    """Trigger parameters shared with the dispatcher/analyzer layer."""

    channel_index: int
    threshold: float
    hysteresis: float
    pretrigger_frac: float
    window_sec: float
    mode: str


__all__ = [
    "DeviceInfo",
    "ChannelInfo",
    "Capabilities",
    "ActualConfig",
    "Chunk",
    "ChunkPointer",
    "Event",
    "EndOfStream",
    "TriggerConfig",
]
