"""Centralized metric computation functions for event analysis.

This module provides signal processing utilities for computing metrics on
detected events, including:
- baseline: Median-based baseline estimation from pre-event samples
- envelope: Signal envelope (max - min) amplitude range
- min_max: Peak amplitude extraction
- peak_frequency_sinc: High-resolution frequency estimation with FFT and sinc interpolation
- autocorr_frequency: Autocorrelation-based frequency estimation fallback
"""
import math
from typing import Optional, Tuple
import numpy as np


def baseline(samples: np.ndarray, pre_samples: int) -> float:
    arr = np.asarray(samples, dtype=np.float32)
    if pre_samples <= 0 or arr.size == 0:
        return 0.0
    return float(np.median(arr[: min(pre_samples, arr.size)]))


def blackman(n: int) -> np.ndarray:
    return np.blackman(max(1, n))


def envelope(samples: np.ndarray) -> float:
    """Compute signal envelope as the difference between max and min amplitudes.

    Args:
        samples: 1D array of waveform samples around event.

    Returns:
        Envelope amplitude (max - min). Returns 0.0 for empty input.
    """
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.max(arr) - np.min(arr))


def min_max(x: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.max(arr)), float(np.min(arr))


def peak_frequency_sinc(
    samples: np.ndarray,
    sr: float,
    *,
    min_hz: float = 50.0,
    center_index: Optional[int] = None,
) -> float:
    if sr <= 0:
        return 0.0
    data = np.asarray(samples, dtype=np.float64)
    if data.size < 8:
        return 0.0
    if not np.any(np.isfinite(data)):
        return 0.0

    data = np.nan_to_num(data, nan=0.0, copy=False)
    center = (
        int(center_index)
        if center_index is not None and 0 <= int(center_index) < data.size
        else int(np.argmax(np.abs(data)))
    )

    span = max(100, int(round(sr * 0.020)))  # ~20 ms window
    half = span // 2
    start = max(0, center - half)
    end = min(data.size, start + span)
    if end - start < 32:
        return 0.0
    segment = data[start:end].copy()

    # Remove mean and linear trend to suppress low-frequency energy
    segment -= np.mean(segment)
    idxs = np.arange(segment.size, dtype=np.float64)
    centered = idxs - idxs.mean()
    denom = float(np.dot(centered, centered))
    if denom > 0:
        slope = float(np.dot(centered, segment) / denom)
        segment -= slope * centered

    if not np.any(segment):
        return 0.0

    window = np.hanning(segment.size)
    tapered = segment * window
    target = max(4096, segment.size * 8)
    n_fft = 1 << int(math.ceil(math.log2(target)))
    spectrum = np.fft.rfft(tapered, n=n_fft)
    mags = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    if freqs.size != mags.size or mags.size <= 1:
        return 0.0

    max_freq = min(sr / 2.5, 4000.0)
    valid = (freqs >= max(min_hz, 1.0)) & (freqs <= max_freq)
    if not np.any(valid):
        return 0.0
    mags = mags[valid]
    freqs = freqs[valid]
    if not np.any(np.isfinite(mags)):
        return 0.0
    power = mags * mags
    if power.size >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        power = np.convolve(power, kernel, mode="same")
    peak_idx = int(np.argmax(power))
    peak_freq = freqs[peak_idx]

    # Quadratic interpolation for sub-bin precision when neighbors exist
    if 0 < peak_idx < mags.size - 1:
        alpha, beta, gamma = mags[peak_idx - 1 : peak_idx + 2]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
            delta = float(np.clip(delta, -1.0, 1.0))
            bin_width = freqs[1] - freqs[0]
            peak_freq += delta * bin_width

    peak_freq = float(max(0.0, peak_freq))
    if peak_freq >= max_freq * 0.98 or peak_freq <= min_hz * 1.02:
        return autocorr_frequency(segment, sr, min_hz, max_hz=max_freq)
    auto_freq = autocorr_frequency(segment, sr, min_hz, max_hz=max_freq)
    if auto_freq <= 0.0:
        return peak_freq
    if peak_freq < min_hz:
        return auto_freq
    rel_diff = abs(auto_freq - peak_freq) / max(min(auto_freq, peak_freq), 1e-6)
    if rel_diff > 0.25:
        return auto_freq
    return peak_freq


def autocorr_frequency(segment: np.ndarray, sr: float, min_hz: float, max_hz: float) -> float:
    if sr <= 0 or segment.size < 2 or max_hz <= min_hz:
        return 0.0
    data = np.asarray(segment, dtype=np.float64)
    if not np.any(np.isfinite(data)):
        return 0.0
    corr = np.correlate(data, data, mode="full")
    corr = corr[corr.size // 2 :]
    if corr.size <= 1:
        return 0.0
    # Use biased autocorrelation to favor earlier peaks (higher frequencies)
    # over later, noisier peaks. This prevents picking up sub-harmonics.
    corr[0] = 0.0
    max_period = min(int(sr / max(min_hz, 1e-6)), corr.size - 1)
    min_period = max(1, int(sr / max(max_hz, 1e-6)))
    if max_period <= min_period:
        return 0.0
    segment_corr = corr[min_period : max_period + 1]
    lags = np.arange(min_period, max_period + 1, dtype=np.float64)
    if segment_corr.size != lags.size or not np.any(np.isfinite(segment_corr)):
        return 0.0
    scores = segment_corr
    best_idx = int(np.argmax(scores))
    if best_idx <= 0 or best_idx >= segment_corr.size - 1:
        return 0.0
    lag = int(lags[best_idx])
    if lag <= 0:
        return 0.0
    freq = sr / lag
    return float(freq if freq >= min_hz else 0.0)


def event_width(
    samples: np.ndarray,
    sr: float,
    *,
    threshold: Optional[float] = None,
    sigma: float = 6.0,
    min_run: int = 3,
    off_count: int = 3,
    off_window: int = 4,
    pre_samples: Optional[int] = None,
) -> float:
    """Calculate event duration (ms) based on robust threshold crossing using global noise.

    Algorithm:
    1. Determine Threshold: Use provided `threshold` or calculate from local MAD.
    2. Find Peak: Locate absolute maximum in the provided samples window.
    3. Find Start (Backwards): Scan backwards from peak to find the start of a
       continuous run of `min_run` samples that are all above threshold.
    4. Find End (Forwards): Scan forwards from peak to find the point where:
       - The current sample is < threshold
       - `off_count` samples within a sliding window of `off_window` are below threshold
       - AND the signal remains below threshold for the remainder of the window.
       If the signal re-crosses threshold (spike recurrence), extend the width.

    Args:
        samples: Waveform array (centered on event).
        sr: Sample rate in Hz.
        threshold: Absolute threshold value to use. If None, calculated from local MAD.
        sigma: Multiplier for MAD-based threshold if threshold is None.
        min_run: Number of consecutive samples > th required to validly start an event.
        off_count: Number of samples < th required within `off_window` to trigger potential end.
        off_window: Size of the sliding window for end detection.
        pre_samples: Number of pre-event samples to use for local noise est (if threshold is None).

    Returns:
        float: Duration in milliseconds. Returns 0.0 if criteria not met.
    """
    if sr <= 0 or samples.size == 0:
        return 0.0

    arr = np.abs(np.asarray(samples, dtype=np.float32))

    # 1. Determine Threshold
    th = 0.0
    if threshold is not None:
        th = float(abs(threshold))
    elif pre_samples is not None and pre_samples > 0:
        # Local MAD fallback
        noise = arr[: min(pre_samples, arr.size)]
        if noise.size > 0:
            med = float(np.median(noise))
            mad = float(np.median(np.abs(noise - med)))
            th = med + (mad * 1.4826 * sigma)
    
    if th <= 0:
        # Failsafe if noise is 0 or threshold not provided
        th = 1e-6

    # 2. Find Peak
    peak_idx = int(np.argmax(arr))
    if arr[peak_idx] < th:
        return 0.0

    # 3. Find Start (Backwards)
    # We want the start of the contiguous block connected to either the peak
    # or the start of the "run" containing the peak.
    start_idx = 0
    curr = peak_idx
    while curr >= 0:
        if arr[curr] < th:
            if (curr + 1 + min_run) <= arr.size and np.all(arr[curr+1 : curr+1+min_run] >= th):
                start_idx = curr + 1
                break
            else:
                 start_idx = curr + 1
                 break
        curr -= 1
    
    if curr < 0:
        start_idx = 0

    # 4. Find End (Forwards)
    end_idx = arr.size
    
    curr = peak_idx + 1
    while curr <= arr.size - off_window:
        # Optimization: Don't check window if current sample is above threshold.
        # This prevents cutting off the tail of a valid signal just because subsequent noise makes the window look "low".
        if arr[curr] >= th:
            curr += 1
            continue

        # Check window at curr
        window = arr[curr : curr + off_window]
        count_below = np.sum(window < th)
        
        if count_below >= off_count:
            # Potential end found.
            # Check re-crossing condition: "do not go above it again for the rest of the window"
            remainder = arr[curr + off_window :]
            if remainder.size == 0 or np.max(remainder) < th:
                # Confirmed end
                end_idx = curr
                break
            else:
                # False ending (e.g. biphasic gap). Continue scan.
                pass
        
        curr += 1

    width_samples = end_idx - start_idx
    if width_samples < min_run:
        return 0.0

    return float(width_samples / sr * 1000.0)
