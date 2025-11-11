import math
import numpy as np

from gui.analysis_tab import _baseline, _energy_density, _min_max, _peak_frequency_sinc


def test_baseline_median():
    samples = np.array([1, 2, 3, 4, 100], dtype=np.float32)
    assert _baseline(samples, 3) == 2.0
    assert _baseline(samples, 0) == 0.0


def test_energy_density_monotonic():
    sr = 20000.0
    t = np.linspace(0, 1e-3, 200, dtype=np.float32)
    base = np.zeros_like(t)
    ed_small = _energy_density(base + 0.1, sr)
    ed_big = _energy_density(base + 1.0, sr)
    assert ed_big > ed_small


def test_min_max_matches_signal():
    t = np.linspace(0, 1e-3, 100, dtype=np.float32)
    wave = 0.2 * np.sin(2 * np.pi * 1000 * t)
    mx, mn = _min_max(wave)
    assert math.isclose(mx, -mn, rel_tol=0.1)


def test_peak_frequency_sinc_accuracy():
    sr = 20000.0
    freq = 1200.0
    t = np.arange(0, 200, dtype=np.float32) / sr
    wave = np.sin(2 * np.pi * freq * t)
    detected = _peak_frequency_sinc(wave, sr)
    assert abs(detected - freq) / freq < 0.02
