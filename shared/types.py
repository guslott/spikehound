"""Analysis-layer event types for GUI and worker communication.

This module defines the detailed AnalysisEvent type used by the analysis worker and GUI.

IMPORTANT: AnalysisEvent is ALWAYS derived from DetectionEvent
============================================================

AnalysisEvent is created exclusively in the analysis layer by enriching a DetectionEvent
with timing metadata and computed metrics. Detectors should NEVER emit AnalysisEvent
directly—they emit the simpler DetectionEvent.

Conversion Contract:
- Detectors (core/detection/) emit → DetectionEvent
- AnalysisWorker converts DetectionEvent → AnalysisEvent via `detection_to_analysis_event()`
- GUI/EventBuffer receives → AnalysisEvent

This explicit conversion makes the data flow clear for both humans and AI tools.

See Also:
    shared.models.DetectionEvent: The canonical detection-layer event type
    analysis.analysis_worker.detection_to_analysis_event: The conversion function
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class AnalysisEvent:
    """Detailed threshold crossing event for analysis and GUI display.
    
    This Event type contains comprehensive timing and sampling metadata needed
    for accurate event visualization and metric computation. It is exchanged
    between the analysis worker and GUI without additional translation.
    
    For the simpler detection-layer type, see `shared.models.DetectionEvent`.
    
    Attributes:
        id: Unique event identifier
        channelId: Channel where the crossing occurred
        thresholdValue: Threshold value that was crossed
        crossingIndex: Absolute sample index of the crossing
        crossingTimeSec: Timestamp of the crossing (seconds)
        firstSampleTimeSec: Timestamp of the first sample in the window
        sampleRateHz: Sample rate for time conversions
        windowMs: Total event window duration in milliseconds
        preMs: Pre-crossing window duration in milliseconds
        postMs: Post-crossing window duration in milliseconds
        samples: Waveform samples around the crossing
        properties: Computed metrics (e.g., energy, frequency)
        intervalSinceLastSec: Time since previous event on this channel
    
    See Also:
        shared.models.DetectionEvent: Simpler detection-layer Event type
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
    properties: Dict[str, float] = field(default_factory=dict)
    intervalSinceLastSec: float = field(default=float("nan"))

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

        interval = float(self.intervalSinceLastSec)
        if math.isfinite(interval) and interval < 0:
            raise ValueError("intervalSinceLastSec must be non-negative")

        arr = np.asarray(self.samples, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("samples must be a 1D array")
        object.__setattr__(self, "samples", np.array(arr, copy=True, order="C"))
        object.__setattr__(self, "properties", dict(self.properties))
        object.__setattr__(self, "intervalSinceLastSec", interval)




