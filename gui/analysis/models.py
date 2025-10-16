from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

@dataclass(frozen=True)
class Event:
    """Detected event from the RealTimeAnalyzer."""
    channel: int
    sample_index: int          # absolute sample index within the run
    amplitude: float
    waveform: np.ndarray       # shape: (window_samples,)
    properties: Dict[str, Any] = field(default_factory=dict)
