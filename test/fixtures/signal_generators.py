"""
Synthetic signal generation utilities for testing.

These generators produce deterministic, reproducible test signals with known
properties that can be used to validate signal processing correctness.

All generators follow a consistent API:
- duration_sec: Signal duration in seconds
- sample_rate: Sample rate in Hz
- Returns: numpy array of float32 samples

Design for AI-maintainability:
- Each function is self-contained with docstrings explaining the math
- Output shapes are always (samples,) for 1D signals
- Use seeded RNG for reproducibility when noise is involved
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def make_sine(
    freq_hz: float,
    amplitude: float,
    duration_sec: float,
    sample_rate: float,
    *,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Generate a pure sine wave.

    Args:
        freq_hz: Frequency in Hz.
        amplitude: Peak amplitude.
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        phase_rad: Initial phase in radians.

    Returns:
        1D float32 array of samples.

    Example:
        >>> sig = make_sine(60.0, 1.0, 1.0, 1000.0)
        >>> sig.shape
        (1000,)
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * math.pi * freq_hz * t + phase_rad)).astype(np.float32)


def make_triphasic_spike(
    width_ms: float,
    amplitude: float,
    sample_rate: float,
) -> np.ndarray:
    """Generate a realistic triphasic extracellular spike template.

    Triphasic waveforms arise from the spatial derivative of the intracellular
    action potential. The template has:
    - Small initial positive deflection
    - Large negative deflection (main spike)
    - Medium positive after-hyperpolarization

    Args:
        width_ms: Approximate spike width at half maximum (ms).
        amplitude: Peak-to-peak amplitude of the resulting waveform.
        sample_rate: Sample rate in Hz.

    Returns:
        1D float32 array containing the spike template.
    """
    # Duration is approximately 3x the width to capture full waveform
    duration_ms = width_ms * 4.0
    n_samples = max(16, int(duration_ms * sample_rate / 1000.0))
    t = np.linspace(0, 1, n_samples, dtype=np.float64)

    # Width scaling factor (narrower = faster dynamics)
    width_factor = 2.0 / max(0.5, width_ms)

    # Triphasic waveform using difference of Gaussians
    center = 0.4  # Slightly off-center for realistic asymmetry
    sigma1 = 0.08 / width_factor  # Initial positive
    sigma2 = 0.12 / width_factor  # Main negative
    sigma3 = 0.15 / width_factor  # AHP positive

    g1 = 0.3 * np.exp(-((t - center + 0.1) ** 2) / (2 * sigma1**2))
    g2 = -1.0 * np.exp(-((t - center) ** 2) / (2 * sigma2**2))
    g3 = 0.5 * np.exp(-((t - center - 0.15) ** 2) / (2 * sigma3**2))

    template = g1 + g2 + g3

    # Normalize to desired peak-to-peak amplitude
    current_pp = float(np.max(template) - np.min(template))
    if current_pp > 0:
        template = template * (amplitude / current_pp)

    # Ensure zero at endpoints
    template -= template[0]
    decay = np.linspace(1, 0, n_samples // 4)
    template[-len(decay):] *= decay

    return template.astype(np.float32)


def make_spike_train(
    spike_times_sec: Sequence[float],
    template: np.ndarray,
    duration_sec: float,
    sample_rate: float,
    *,
    amplitudes: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a signal with spikes at known times.

    Args:
        spike_times_sec: List of spike times in seconds.
        template: Spike waveform template to insert.
        duration_sec: Total signal duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitudes: Optional per-spike amplitude multipliers.

    Returns:
        Tuple of (signal, ground_truth_times) where:
        - signal: 1D float32 array
        - ground_truth_times: 1D float64 array of actual spike times (samples)

    Example:
        >>> template = make_triphasic_spike(1.0, 0.5, 10000)
        >>> signal, times = make_spike_train([0.1, 0.2, 0.3], template, 0.5, 10000)
        >>> len(times)
        3
    """
    n_samples = int(duration_sec * sample_rate)
    signal = np.zeros(n_samples, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    template_len = len(template)

    # Find the "peak" of template (location of minimum for negative-going spikes)
    template_peak_offset = int(np.argmin(template))

    actual_times = []

    for i, t_sec in enumerate(spike_times_sec):
        t_sample = int(t_sec * sample_rate)

        # Calculate insertion range, accounting for template peak alignment
        start = t_sample - template_peak_offset
        end = start + template_len

        if start < 0 or end > n_samples:
            continue  # Skip spikes that would fall outside signal

        amp = amplitudes[i] if amplitudes is not None else 1.0
        signal[start:end] += template * amp
        actual_times.append(t_sample)

    return signal.astype(np.float32), np.array(actual_times, dtype=np.float64)


def make_dc_with_drift(
    dc_offset: float,
    drift_slope: float,
    duration_sec: float,
    sample_rate: float,
) -> np.ndarray:
    """Generate DC offset with linear drift.

    Args:
        dc_offset: Initial DC offset value.
        drift_slope: Linear drift rate (units per second).
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        1D float32 array.

    Example:
        >>> sig = make_dc_with_drift(1.0, 0.1, 10.0, 1000.0)
        >>> sig[0], sig[-1]  # Should be ~1.0 and ~2.0
        (1.0, 1.9999...)
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    return (dc_offset + drift_slope * t).astype(np.float32)


def make_mains_hum(
    freq_hz: float,
    amplitude: float,
    duration_sec: float,
    sample_rate: float,
    *,
    harmonics: int = 3,
) -> np.ndarray:
    """Generate mains power-line interference.

    Args:
        freq_hz: Fundamental frequency (50 or 60 Hz typically).
        amplitude: Fundamental amplitude.
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        harmonics: Number of harmonics to include (1 = fundamental only).

    Returns:
        1D float32 array.
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    signal = np.zeros(n_samples, dtype=np.float64)
    for h in range(1, harmonics + 1):
        # Harmonic amplitudes decrease as 1/h
        harm_amp = amplitude / h
        signal += harm_amp * np.sin(2.0 * math.pi * freq_hz * h * t)

    return signal.astype(np.float32)


def add_gaussian_noise(
    signal: np.ndarray,
    snr_db: float,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian white noise to a signal at specified SNR.

    Args:
        signal: Input signal.
        snr_db: Signal-to-noise ratio in decibels.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Signal with added noise (float32).

    Example:
        >>> sig = make_sine(100, 1.0, 1.0, 1000)
        >>> noisy = add_gaussian_noise(sig, 20.0, seed=42)
        >>> noisy.shape == sig.shape
        True
    """
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=np.float64)

    # Calculate signal power
    sig_power = float(np.mean(signal**2))
    if sig_power == 0:
        sig_power = 1e-12  # Prevent division by zero

    # SNR = 10 * log10(Psig / Pnoise)
    # Pnoise = Psig / 10^(SNR/10)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)

    noise = rng.normal(0, noise_std, signal.shape)
    return (signal + noise).astype(np.float32)


def make_poisson_spike_train(
    rate_hz: float,
    duration_sec: float,
    sample_rate: float,
    template: np.ndarray,
    *,
    refractory_ms: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Poisson spike train with known statistics.

    Args:
        rate_hz: Mean firing rate in Hz.
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        template: Spike template to insert.
        refractory_ms: Absolute refractory period in ms.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (signal, spike_times_samples).
    """
    rng = np.random.default_rng(seed)

    # Generate ISIs from exponential distribution
    mean_isi_sec = 1.0 / rate_hz
    refractory_sec = refractory_ms / 1000.0

    spike_times = []
    current_time = refractory_sec  # Start after one refractory period

    while current_time < duration_sec - 0.01:  # Leave margin at end
        # Exponential ISI with refractory period
        isi = rng.exponential(mean_isi_sec)
        isi = max(isi, refractory_sec)  # Enforce refractory
        current_time += isi
        if current_time < duration_sec - 0.01:
            spike_times.append(current_time)

    return make_spike_train(spike_times, template, duration_sec, sample_rate)
