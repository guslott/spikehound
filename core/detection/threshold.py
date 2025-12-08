import numpy as np
from typing import Iterable, Mapping, Dict, Optional

from shared.models import DetectionEvent, Chunk
from .base import EventDetector, DetectorParameter, register_detector


@register_detector
class AmpThresholdDetector:
    name = "amp_threshold"
    display_name = "Amplitude Threshold (Auto)"

    def __init__(self):
        self._factor: float = 5.0
        self._sign: int = -1  # -1 for negative peaks, 1 for positive
        self._sample_rate: float = 0.0
        self._n_channels: int = 0
        self._noise_levels: Optional[np.ndarray] = None
        self._refractory_samples: int = 0
        self._last_event_samples: Optional[np.ndarray] = None
        self._residue: Optional[np.ndarray] = None
        self._last_event_time: Optional[np.ndarray] = None
        
        # Parameters
        self._params = {
            "factor": DetectorParameter(
                name="factor",
                default=5.0,
                min=1.0,
                max=20.0,
                help="Threshold multiplier (x * noise_level)"
            ),
            "sign": DetectorParameter(
                name="sign",
                default=-1,
                min=-1,
                max=1,
                help="Polarity: -1 for negative peaks, 1 for positive"
            ),
            "refractory_ms": DetectorParameter(
                name="refractory_ms",
                default=1.0,
                min=0.1,
                max=10.0,
                help="Minimum time between spikes (ms)"
            )
        }
        self._refractory_ms = 1.0
        self._window_ms = 2.0

    @property
    def parameters(self) -> Mapping[str, DetectorParameter]:
        # Update params to include window_ms
        params = dict(self._params)
        params["window_ms"] = DetectorParameter(
            name="window_ms",
            default=2.0,
            min=0.5,
            max=10.0,
            help="Event window duration (ms)"
        )
        return params

    def configure(self, **params) -> None:
        if "factor" in params:
            self._factor = float(params["factor"])
        if "sign" in params:
            self._sign = int(params["sign"])
        if "refractory_ms" in params:
            self._refractory_ms = float(params["refractory_ms"])
            if self._sample_rate > 0:
                self._refractory_samples = int(self._refractory_ms * 1e-3 * self._sample_rate)
        if "window_ms" in params:
            self._window_ms = float(params["window_ms"])

    def reset(self, sample_rate: float, n_channels: int) -> None:
        self._sample_rate = sample_rate
        self._n_channels = n_channels
        self._noise_levels = None  # Will be estimated from first chunk(s) or running average
        self._refractory_samples = int(self._refractory_ms * 1e-3 * sample_rate) if sample_rate > 0 else 0
        self._refractory_samples = int(self._refractory_ms * 1e-3 * sample_rate) if sample_rate > 0 else 0
        self._last_event_time = np.zeros(n_channels, dtype=np.float64) - 1000.0
        self._last_event_time = np.zeros(n_channels, dtype=np.float64) - 1000.0 # Initialize far in the past
        self._residue = None # Reset residue buffer

    def process_chunk(self, chunk: Chunk) -> Iterable[DetectionEvent]:
        if self._sample_rate <= 0:
            return []
        
        samples = chunk.samples # (n_channels, n_samples)
        if samples.shape[1] == 0:
            # If the current chunk is empty, preserve existing residue.
            # If residue is also empty, there is nothing to process.
            if self._residue is None or self._residue.shape[1] == 0:
                return []

        # Calculate window parameters
        window_samples = int(self._window_ms * 1e-3 * self._sample_rate)
        pre_samples = window_samples // 3
        post_samples = window_samples - pre_samples
        
        # Stitch residue
        if self._residue is not None and self._residue.shape[1] > 0:
            full_samples = np.concatenate([self._residue, samples], axis=1)
            # Calculate start time of the concatenated buffer.
            # The residue precedes the current chunk start time.
            residue_len = self._residue.shape[1]
            full_start_time = chunk.start_time - (residue_len * chunk.dt)
        else:
            full_samples = samples
            full_start_time = chunk.start_time
            residue_len = 0

        # Estimate noise if not yet done (simple one-shot for now, could be sliding)
        if self._noise_levels is None:
            # MAD = median(|x - median(x)|)
            # Robust to DC offsets
            med = np.median(full_samples, axis=1, keepdims=True)
            mad = np.median(np.abs(full_samples - med), axis=1)
            self._noise_levels = mad / 0.6745
            # Avoid zero threshold
            self._noise_levels[self._noise_levels == 0] = 1.0

        # Scan the full buffer for events.
        # Duplicate detection is prevented by filtering against `_last_event_time`.
        
        # Define the valid detection range to ensure sufficient pre- and post-event samples.
        # Crossing index 'idx' must satisfy:
        # pre_samples <= idx < full_samples.shape[1] - post_samples
        
        valid_end_idx = full_samples.shape[1] - post_samples
        if valid_end_idx <= pre_samples:
            # Not enough data even for one event
            self._residue = full_samples
            return []
            
        thresholds = self._noise_levels * self._factor
        events = []
        
        for ch in range(self._n_channels):
            ch_data = full_samples[ch]
            thresh = thresholds[ch]
            
            if self._sign == 0:
                candidates = np.where(np.abs(ch_data) > thresh)[0]
            elif self._sign < 0:
                candidates = np.where(ch_data < -thresh)[0]
            else:
                candidates = np.where(ch_data > thresh)[0]
                
            if len(candidates) == 0:
                continue
                
            last_t = self._last_event_time[ch]
            # Enforce refractory period to be at least the window length to avoid double detection
            window_sec = self._window_ms * 1e-3
            refractory_sec = max(self._refractory_ms * 1e-3, window_sec)
            
            for idx in candidates:
                # Check bounds
                if idx < pre_samples:
                    continue
                if idx >= valid_end_idx:
                    # Too close to end, will be handled next time
                    continue
                    
                # Calculate time
                t = full_start_time + (idx * chunk.dt)
                
                # Check refractory
                if t - last_t < refractory_sec:
                    continue
                
                # Refine peak
                # Search in a small window around the crossing
                search_width = 20
                s_start = idx
                s_end = min(full_samples.shape[1], idx + search_width)
                segment = ch_data[s_start:s_end]
                
                if self._sign == 0:
                    peak_offset = np.argmax(np.abs(segment))
                elif self._sign < 0:
                    peak_offset = np.argmin(segment)
                else:
                    peak_offset = np.argmax(segment)
                
                real_idx = idx + peak_offset
                
                # Re-check bounds for refined peak
                if real_idx < pre_samples or real_idx >= full_samples.shape[1] - post_samples:
                    continue
                    
                real_t = full_start_time + (real_idx * chunk.dt)
                
                # Re-check refractory
                if real_t - last_t < refractory_sec:
                    continue
                
                # Extract window
                w_start = real_idx - pre_samples
                w_end = real_idx + post_samples
                waveform = ch_data[w_start:w_end]
                
                # Create event
                event = DetectionEvent(
                    t=real_t,
                    chan=ch,
                    window=waveform,
                    properties={"amplitude": float(waveform[pre_samples] if pre_samples < len(waveform) else 0)},
                    params={"threshold": float(thresh), "pre_samples": int(pre_samples)}
                )
                events.append(event)
                self._last_event_time[ch] = real_t
                last_t = real_t
                
        # Retain the end of the current buffer as residue for the next chunk.
        # The residue length must be at least the window size to ensure continuity
        # and allow detection of events straddling the chunk boundary.
        
        keep_len = window_samples
        if full_samples.shape[1] > keep_len:
            self._residue = full_samples[:, -keep_len:]
        else:
            self._residue = full_samples
            
        return events

    def finalize(self) -> Iterable[DetectionEvent]:
        return []

