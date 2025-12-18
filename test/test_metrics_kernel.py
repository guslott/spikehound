import math
import numpy as np
import pytest

from analysis.metrics import baseline, envelope, min_max, peak_frequency_sinc


def test_baseline_median():
    samples = np.array([1, 2, 3, 4, 100], dtype=np.float32)
    assert baseline(samples, 3) == 2.0
    assert baseline(samples, 0) == 0.0


def test_envelope_returns_range():
    """Test envelope returns max - min of signal."""
    samples = np.array([-0.5, 0.0, 1.0, 0.5], dtype=np.float32)
    env = envelope(samples)
    assert math.isclose(env, 1.5, rel_tol=1e-6)  # 1.0 - (-0.5) = 1.5


def test_envelope_empty():
    """Test envelope handles empty arrays."""
    assert envelope(np.array([], dtype=np.float32)) == 0.0


def test_min_max_matches_signal():
    t = np.linspace(0, 1e-3, 100, dtype=np.float32)
    wave = 0.2 * np.sin(2 * np.pi * 1000 * t)
    mx, mn = min_max(wave)
    assert math.isclose(mx, -mn, rel_tol=0.1)


def test_peak_frequency_sinc_with_localized_burst():
    """Test peak_frequency_sinc detects frequency in a localized burst (like a spike event)."""
    sr = 20000.0
    samples = 400  # 20ms window
    wave = np.zeros(samples, dtype=np.float64)
    
    # Create a localized burst centered in the window
    burst_len = int(sr * 0.008)  # 8ms burst
    start = samples // 2 - burst_len // 2
    t = np.arange(burst_len) / sr
    freq = 300.0  # Target frequency
    burst = np.sin(2 * np.pi * freq * t) * np.hanning(burst_len)
    wave[start : start + burst_len] = burst
    center = start + burst_len // 2
    
    detected = peak_frequency_sinc(wave, sr, center_index=center)
    # Allow 15% tolerance for frequency detection
    assert abs(detected - freq) / freq < 0.15, f"Expected ~{freq}Hz, got {detected}Hz"
