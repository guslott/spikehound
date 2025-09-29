from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Tuple

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
            raise ValueError(
                "samples shape mismatch: axis 0 must match len(channel_names)"
            )

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

    def __reduce__(self):  # pragma: no cover - exercised via pickle roundtrip tests
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
class Event:
    """Detected feature emitted by the analyzer."""

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

    def __reduce__(self):  # pragma: no cover - exercised via pickle roundtrip tests
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


def _restore_end_of_stream() -> "_EndOfStreamSentinel":  # pragma: no cover - pickle glue
    return EndOfStream


class _EndOfStreamSentinel:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "EndOfStream"

    def __reduce__(self):  # Ensure pickle returns the singleton
        return (_restore_end_of_stream, ())


EndOfStream = _EndOfStreamSentinel()
