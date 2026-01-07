"""
Extended unit tests for metric computation functions.

These tests verify metrics produce correct values on signals with known properties.
Each test starts with a synthetic signal where the "correct answer" is calculable,
then verifies the metric output matches within tolerance.

Coverage includes:
- Peak frequency on pure tones
- Event width on known pulse shapes
- Envelope/min-max on known ranges
- Baseline estimation accuracy
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from analysis.metrics import (
    baseline,
    envelope,
    min_max,
    peak_frequency_sinc,
    autocorr_frequency,
    event_width,
)
from test.fixtures.signal_generators import (
    make_sine,
    make_triphasic_spike,
    add_gaussian_noise,
)


class TestPeakFrequency:
    """Tests for peak_frequency_sinc."""

    @pytest.mark.parametrize("freq_hz", [500, 800, 1200, 1500, 2000])
    def test_pure_sine_frequency_detected(self, freq_hz: int):
        """Peak frequency of pure sine should match input frequency."""
        sample_rate = 10000.0
        duration = 0.2  # Longer duration for better resolution
        signal = make_sine(freq_hz, 1.0, duration, sample_rate)

        detected = peak_frequency_sinc(signal, sample_rate, min_hz=100.0)

        # Should be within 10% of true frequency
        tolerance = freq_hz * 0.1
        assert abs(detected - freq_hz) <= tolerance, \
            f"Expected ~{freq_hz}Hz, got {detected:.1f}Hz"

    def test_noisy_sine_frequency_detected(self):
        """Peak frequency should work on noisy signal."""
        sample_rate = 10000.0
        freq_hz = 250.0
        duration = 0.1

        signal = make_sine(freq_hz, 1.0, duration, sample_rate)
        noisy = add_gaussian_noise(signal, snr_db=20.0, seed=42)

        detected = peak_frequency_sinc(noisy, sample_rate, min_hz=50.0)

        # Allow 10% tolerance with noise
        tolerance = freq_hz * 0.10
        assert abs(detected - freq_hz) <= tolerance, \
            f"Expected ~{freq_hz}Hz, got {detected:.1f}Hz"

    def test_empty_signal_returns_zero(self):
        """Empty signal should return 0."""
        result = peak_frequency_sinc(np.array([]), 10000.0)
        assert result == 0.0

    def test_short_signal_returns_zero(self):
        """Very short signal should return 0."""
        result = peak_frequency_sinc(np.array([1, 2, 3]), 10000.0)
        assert result == 0.0

    def test_dc_signal_returns_zero_or_low(self):
        """DC signal has no frequency content above min_hz."""
        dc_signal = np.ones(1000, dtype=np.float32) * 5.0
        result = peak_frequency_sinc(dc_signal, 10000.0, min_hz=50.0)
        # Either 0 or very low frequency
        assert result < 100.0

    def test_invalid_sample_rate(self):
        """Zero or negative sample rate should return 0."""
        signal = np.sin(np.linspace(0, 10, 100))
        assert peak_frequency_sinc(signal, 0.0) == 0.0
        assert peak_frequency_sinc(signal, -1000.0) == 0.0


class TestAutocorrFrequency:
    """Tests for autocorrelation-based frequency estimation."""

    def test_pure_sine_autocorr(self):
        """Autocorrelation should detect sine frequency."""
        sample_rate = 10000.0
        freq_hz = 1000.0
        duration = 0.2

        t = np.arange(int(duration * sample_rate)) / sample_rate
        signal = np.sin(2 * math.pi * freq_hz * t)

        detected = autocorr_frequency(signal, sample_rate, min_hz=100.0, max_hz=3000.0)

        # Autocorr is less precise, allow 15% tolerance
        tolerance = freq_hz * 0.15
        assert abs(detected - freq_hz) <= tolerance, \
            f"Expected ~{freq_hz}Hz, got {detected:.1f}Hz"


class TestEventWidth:
    """Tests for event_width calculation."""

    def test_symmetric_pulse_width(self):
        """Symmetric pulse should have predictable width."""
        sample_rate = 10000.0
        width_ms = 2.0

        # Create Gaussian pulse
        n_samples = 100
        t = np.linspace(-3, 3, n_samples)
        # Width of Gaussian at half-max is ~2.35 sigma
        sigma = (width_ms / 1000.0 * sample_rate) / 2.35
        pulse = np.exp(-t**2 / (2 * (sigma/n_samples*6)**2))
        pulse = pulse.astype(np.float32)

        # Threshold at 50% of peak
        threshold = 0.5 * np.max(pulse)

        result = event_width(pulse, sample_rate, threshold=threshold)

        # Allow 30% tolerance due to different width definitions
        assert 1.0 < result < 4.0, f"Width {result:.2f}ms outside expected range"

    def test_triphasic_spike_width(self):
        """Triphasic spike should have measurable width."""
        sample_rate = 10000.0
        template = make_triphasic_spike(1.5, 1.0, sample_rate)

        # Use MAD-based threshold
        result = event_width(template, sample_rate, pre_samples=10)

        # Should be positive and reasonable
        assert 0.5 < result < 5.0, f"Triphasic width {result:.2f}ms unexpected"

    def test_empty_signal_returns_zero(self):
        """Empty signal should return 0."""
        result = event_width(np.array([]), 10000.0)
        assert result == 0.0

    def test_below_threshold_returns_zero(self):
        """Signal entirely below threshold should return 0."""
        low_signal = np.ones(100, dtype=np.float32) * 0.001
        result = event_width(low_signal, 10000.0, threshold=1.0)
        assert result == 0.0


class TestEnvelope:
    """Tests for envelope (peak-to-peak) calculation."""

    def test_known_range(self):
        """Envelope should equal max - min."""
        signal = np.array([1.0, 5.0, 2.0, -3.0, 4.0], dtype=np.float32)
        # max = 5, min = -3, envelope = 8
        result = envelope(signal)
        assert result == 8.0

    def test_sine_wave_envelope(self):
        """Sine wave envelope should be 2*amplitude."""
        amplitude = 2.5
        signal = make_sine(100, amplitude, 0.1, 10000)
        result = envelope(signal)
        # Should be close to 2*amplitude
        assert abs(result - 2 * amplitude) < 0.1

    def test_empty_returns_zero(self):
        """Empty signal returns 0."""
        result = envelope(np.array([]))
        assert result == 0.0

    def test_constant_returns_zero(self):
        """Constant signal returns 0 envelope."""
        result = envelope(np.ones(100, dtype=np.float32) * 3.0)
        assert result == 0.0


class TestMinMax:
    """Tests for min_max function."""

    def test_known_values(self):
        """min_max should return correct max and min."""
        signal = np.array([1.0, 5.0, 2.0, -3.0, 4.0], dtype=np.float32)
        max_val, min_val = min_max(signal)
        assert max_val == 5.0
        assert min_val == -3.0

    def test_empty_returns_zeros(self):
        """Empty signal returns (0, 0)."""
        max_val, min_val = min_max(np.array([]))
        assert max_val == 0.0
        assert min_val == 0.0

    def test_single_value(self):
        """Single value returns same for max and min."""
        max_val, min_val = min_max(np.array([3.5]))
        assert max_val == 3.5
        assert min_val == 3.5


class TestBaseline:
    """Tests for baseline estimation."""

    def test_baseline_from_pre_samples(self):
        """Baseline should be median of pre-samples."""
        # Signal with clear pre-event baseline
        pre = np.array([1.0, 1.1, 0.9, 1.0, 1.0])  # Baseline ~1.0
        event = np.array([5.0, 10.0, -3.0])  # Event with large excursion
        signal = np.concatenate([pre, event]).astype(np.float32)

        result = baseline(signal, pre_samples=5)
        assert abs(result - 1.0) < 0.1  # Should be close to 1.0

    def test_baseline_zero_pre_samples(self):
        """Zero pre_samples returns 0."""
        result = baseline(np.array([1, 2, 3]), pre_samples=0)
        assert result == 0.0

    def test_baseline_exceeds_length(self):
        """Pre_samples > signal length should use whole signal."""
        signal = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        result = baseline(signal, pre_samples=100)
        assert abs(result - 2.0) < 0.01


class TestMetricsNumericalStability:
    """Tests for numerical edge cases."""

    def test_peak_frequency_with_nan(self):
        """NaN values should be handled gracefully."""
        signal = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
        result = peak_frequency_sinc(signal, 10000.0)
        # Should return something without crashing
        assert np.isfinite(result)

    def test_peak_frequency_with_inf(self):
        """Inf values should be handled gracefully."""
        signal = np.array([1.0, np.inf, 2.0, 3.0])
        result = peak_frequency_sinc(signal, 10000.0)
        # Either handles it or returns 0
        assert np.isfinite(result)

    def test_envelope_with_inf(self):
        """Inf in signal."""
        signal = np.array([1.0, np.inf, 2.0])
        result = envelope(signal)
        # Returns inf distance
        assert result == np.inf or np.isnan(result)

    def test_min_max_with_nan(self):
        """NaN in signal."""
        signal = np.array([1.0, np.nan, 2.0])
        max_val, min_val = min_max(signal)
        # NumPy behavior for nan comparison
        assert np.isnan(max_val) or max_val >= 1.0
