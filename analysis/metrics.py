"""Centralized metric computation functions for event analysis.

This module provides signal processing utilities for computing metrics on
detected events, including:
- baseline: Median-based baseline estimation from pre-event samples
- energy_density: Windowed energy density computation
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


def energy_density(x: np.ndarray, sr: float) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0 or sr <= 0:
        return 0.0
    base = baseline(arr, max(1, int(0.1 * arr.size)))
    x_detrend = arr - base
    window = blackman(arr.size)
    weighted = x_detrend * window
    energy = np.sum(weighted * weighted, dtype=np.float64)
    window_sec = max(1e-12, arr.size / float(sr))
    return float(np.sqrt(energy / window_sec))


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

    span = max(64, int(round(sr * 0.008)))  # ~8 ms window
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

    max_freq = min(sr / 6.0, 1000.0)
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
    counts = np.arange(corr.size, 0, -1, dtype=np.float64)
    corr = corr / counts
    corr[0] = 0.0
    max_period = min(int(sr / max(min_hz, 1e-6)), corr.size - 1)
    min_period = max(1, int(sr / max(max_hz, 1e-6)))
    if max_period <= min_period:
        return 0.0
    segment_corr = corr[min_period : max_period + 1]
    lags = np.arange(min_period, max_period + 1, dtype=np.float64)
    if segment_corr.size != lags.size or not np.any(np.isfinite(segment_corr)):
        return 0.0
    scores = segment_corr * np.sqrt(np.maximum(1.0, lags))
    best_idx = int(np.argmax(scores))
    if best_idx <= 0 or best_idx >= segment_corr.size - 1:
        return 0.0
    lag = int(lags[best_idx])
    if lag <= 0:
        return 0.0
    freq = sr / lag
    return float(freq if freq >= min_hz else 0.0)
