from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Event:
    """
    Lightweight description of a single threshold crossing that both the
    analysis worker and GUI can exchange without additional translation.
    """

    id: int
    channelId: int
    thresholdValue: float
    crossingIndex: int
    crossingTimeSec: float
    firstSampleTimeSec: float
    sampleRateHz: float
    windowMs: float
    preMs: float
    postMs: float
    samples: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.channelId < 0:
            raise ValueError("channelId must be non-negative")
        if self.crossingIndex < 0:
            raise ValueError("crossingIndex must be non-negative")
        if self.sampleRateHz <= 0:
            raise ValueError("sampleRateHz must be positive")
        for name, value in (("windowMs", self.windowMs), ("preMs", self.preMs), ("postMs", self.postMs)):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        arr = np.asarray(self.samples, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("samples must be a 1D array")
        object.__setattr__(self, "samples", np.array(arr, copy=True, order="C"))
