# analysis/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass(frozen=True)
class Event:
    """
    A detected spike/event.
    """
    timestamp_s: float                 # Wall/mono time for the event (approx at peak)
    channel: int                       # Channel index where event was detected
    sample_index: int                  # Sample index within the current run (best-effort; 0 if unknown)
    waveform: np.ndarray               # 1D window around the event (float32)
    properties: Dict[str, float] = field(default_factory=dict)  # e.g., peak_amp, energy, width


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
