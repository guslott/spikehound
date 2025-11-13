# analysis/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

from core.models import Chunk
from shared.types import Event


@dataclass
class ThresholdConfig:
    """
    Configuration for the RealTimeAnalyzer's simple threshold detector.
    """
    per_channel_thresholds: Optional[np.ndarray] = None  # absolute thresholds (V). If None, auto from noise
    polarity: str = "neg"         # "neg", "pos", or "both"
    auto_k_sigma: float = 4.5     # k * noise (MAD→sigma) when auto-thresholding
    refractory_s: float = 0.003   # seconds to suppress subsequent detections per channel
    window_pre_s: float = 0.002   # seconds before threshold crossing to include in waveform
    window_post_s: float = 0.004  # seconds after threshold crossing to include in waveform


@dataclass(frozen=True)
class AnalysisBatch:
    """Chunk routed to a single channel plus the events detected within that chunk."""

    chunk: Chunk
    events: Sequence[Event]


__all__ = ["Event", "ThresholdConfig", "AnalysisBatch"]
