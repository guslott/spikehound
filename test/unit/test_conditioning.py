"""
Unit tests for signal conditioning (filters).

These tests verify that filters produce quantitatively correct results,
not just "looks right" visual inspection. Each test includes:
1. Known input signal with calculable properties
2. Expected output characteristics (dB attenuation, preserved components)
3. Quantitative assertions using FFT analysis

Filter correctness is important because incorrect filtering could:
- Remove neural signals (false negatives in detection)
- Create artifacts that look like spikes (false positives)
- Distort spike waveforms (incorrect metrics)
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from core.conditioning import (
    ChannelFilterSettings,
    FilterSettings,
    SignalConditioner,
)
from shared.models import Chunk


def compute_power_spectrum(
    signal: np.ndarray,
    sample_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum of signal.

    Returns:
        Tuple of (frequencies, power_db) arrays.
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2 / n
    power_db = 10 * np.log10(power + 1e-12)  # Add epsilon to avoid log(0)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return freqs, power_db


def find_power_at_frequency(
    signal: np.ndarray,
    sample_rate: float,
    target_freq: float,
    bandwidth: float = 5.0,
) -> float:
    """Find power (dB) at target frequency within bandwidth."""
    freqs, power_db = compute_power_spectrum(signal, sample_rate)
    mask = np.abs(freqs - target_freq) <= bandwidth
    if not np.any(mask):
        return -100.0  # Very low power
    return float(np.max(power_db[mask]))


def make_chunk(
    samples: np.ndarray,
    sample_rate: float,
    channel_names: Tuple[str, ...] = ("Ch0",),
) -> Chunk:
    """Create a Chunk from samples array."""
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    return Chunk(
        samples=samples.astype(np.float32),
        start_time=0.0,
        dt=1.0 / sample_rate,
        seq=0,
        channel_names=channel_names,
        units="V",
    )


class TestNotchFilter:
    """Tests for notch filter (mains hum removal)."""

    def test_notch_attenuates_target_frequency(self):
        """Notch filter should attenuate 60Hz by at least 20dB."""
        sample_rate = 4000.0
        duration = 2.0
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate

        # Pure 60Hz signal
        signal_60hz = np.sin(2 * math.pi * 60 * t).astype(np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True,
                notch_freq_hz=60.0,
                notch_q=30.0,
            )
        )
        conditioner = SignalConditioner(settings)
        chunk = make_chunk(signal_60hz, sample_rate)

        filtered = conditioner.process(chunk)

        # Measure power at 60Hz before and after
        input_power = find_power_at_frequency(signal_60hz, sample_rate, 60.0)
        output_power = find_power_at_frequency(filtered.flatten(), sample_rate, 60.0)

        attenuation = input_power - output_power
        assert attenuation >= 20.0, f"Expected >=20dB attenuation, got {attenuation:.1f}dB"

    def test_notch_preserves_distant_frequencies(self):
        """Notch at 60Hz should preserve 200Hz within 3dB."""
        sample_rate = 4000.0
        duration = 2.0
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate

        # Signal at 200Hz (well away from 60Hz notch)
        signal_200hz = np.sin(2 * math.pi * 200 * t).astype(np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True,
                notch_freq_hz=60.0,
                notch_q=30.0,
            )
        )
        conditioner = SignalConditioner(settings)
        chunk = make_chunk(signal_200hz, sample_rate)

        filtered = conditioner.process(chunk)

        input_power = find_power_at_frequency(signal_200hz, sample_rate, 200.0)
        output_power = find_power_at_frequency(filtered.flatten(), sample_rate, 200.0)

        # Should be within 3dB (not significantly attenuated)
        power_change = abs(input_power - output_power)
        assert power_change < 3.0, f"200Hz changed by {power_change:.1f}dB (expected <3dB)"

    def test_notch_50hz_european_mains(self):
        """Notch at 50Hz for European mains."""
        sample_rate = 4000.0
        duration = 2.0
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate

        signal_50hz = np.sin(2 * math.pi * 50 * t).astype(np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True,
                notch_freq_hz=50.0,
                notch_q=30.0,
            )
        )
        conditioner = SignalConditioner(settings)
        chunk = make_chunk(signal_50hz, sample_rate)

        filtered = conditioner.process(chunk)

        input_power = find_power_at_frequency(signal_50hz, sample_rate, 50.0)
        output_power = find_power_at_frequency(filtered.flatten(), sample_rate, 50.0)

        attenuation = input_power - output_power
        assert attenuation >= 20.0, f"Expected >=20dB attenuation at 50Hz, got {attenuation:.1f}dB"


class TestHighpassFilter:
    """Tests for AC coupling / high-pass filter."""

    def test_highpass_removes_dc_offset(self):
        """High-pass filter should remove DC offset."""
        sample_rate = 2000.0
        duration = 2.0
        n_samples = int(duration * sample_rate)

        # Signal with large DC offset
        dc_offset = 5.0
        signal = np.full(n_samples, dc_offset, dtype=np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                ac_couple=True,
                ac_cutoff_hz=1.0,
            )
        )
        conditioner = SignalConditioner(settings)
        chunk = make_chunk(signal, sample_rate)

        filtered = conditioner.process(chunk)

        # After settled, mean should be near zero
        # Skip first 10% for filter settling
        settled = filtered.flatten()[n_samples // 10:]
        mean_val = abs(float(np.mean(settled)))

        assert mean_val < 0.1, f"DC not removed: mean={mean_val:.3f}"

    def test_highpass_preserves_signal_above_cutoff(self):
        """High-pass should preserve signals well above cutoff."""
        sample_rate = 2000.0
        duration = 2.0
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate

        # 100Hz signal (well above 1Hz cutoff)
        signal_100hz = np.sin(2 * math.pi * 100 * t).astype(np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                ac_couple=True,
                ac_cutoff_hz=1.0,
            )
        )
        conditioner = SignalConditioner(settings)
        chunk = make_chunk(signal_100hz, sample_rate)

        filtered = conditioner.process(chunk)

        # Skip settling period
        input_settled = signal_100hz[n_samples // 10:]
        output_settled = filtered.flatten()[n_samples // 10:]

        input_rms = float(np.sqrt(np.mean(input_settled ** 2)))
        output_rms = float(np.sqrt(np.mean(output_settled ** 2)))

        # RMS should be preserved within 10%
        ratio = output_rms / input_rms
        assert 0.9 < ratio < 1.1, f"Signal not preserved: ratio={ratio:.2f}"

    def test_highpass_removes_drift(self):
        """High-pass should remove low-frequency drift."""
        sample_rate = 2000.0
        duration = 4.0
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate

        # Slow drift (0.1 Hz) + fast signal (50 Hz)
        drift = 2.0 * np.sin(2 * math.pi * 0.1 * t)
        fast_signal = 0.5 * np.sin(2 * math.pi * 50 * t)
        combined = (drift + fast_signal).astype(np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                ac_couple=True,
                ac_cutoff_hz=1.0,
            )
        )
        conditioner = SignalConditioner(settings)
        chunk = make_chunk(combined, sample_rate)

        filtered = conditioner.process(chunk)

        # After filtering, 50Hz should dominate
        settled = filtered.flatten()[n_samples // 4:]

        # Check that drift component is reduced
        power_0_1hz = find_power_at_frequency(settled, sample_rate, 0.1, bandwidth=0.5)
        power_50hz = find_power_at_frequency(settled, sample_rate, 50.0, bandwidth=2.0)

        # 50Hz should be much stronger than 0.1Hz (drift removed)
        assert power_50hz > power_0_1hz + 10, "Drift not sufficiently removed"


class TestLowpassFilter:
    """Tests for low-pass filter (high-frequency noise removal)."""

    def test_lowpass_attenuates_high_frequencies(self):
        """Low-pass should attenuate frequencies above cutoff."""
        sample_rate = 10000.0
        duration = 1.0
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate

        # Signal well above 500Hz cutoff
        signal_2000hz = np.sin(2 * math.pi * 2000 * t).astype(np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                lowpass_hz=500.0,
                lowpass_order=4,
            )
        )
        conditioner = SignalConditioner(settings)
        chunk = make_chunk(signal_2000hz, sample_rate)

        filtered = conditioner.process(chunk)

        input_power = find_power_at_frequency(signal_2000hz, sample_rate, 2000.0)
        output_power = find_power_at_frequency(filtered.flatten(), sample_rate, 2000.0)

        attenuation = input_power - output_power
        # 4th order Butterworth should give significant attenuation at 4x cutoff
        assert attenuation >= 20.0, f"Expected >=20dB attenuation at 2kHz, got {attenuation:.1f}dB"

    def test_lowpass_preserves_low_frequencies(self):
        """Low-pass should preserve frequencies well below cutoff."""
        sample_rate = 10000.0
        duration = 1.0
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate

        # 100Hz signal (well below 500Hz cutoff)
        signal_100hz = np.sin(2 * math.pi * 100 * t).astype(np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                lowpass_hz=500.0,
                lowpass_order=4,
            )
        )
        conditioner = SignalConditioner(settings)
        chunk = make_chunk(signal_100hz, sample_rate)

        filtered = conditioner.process(chunk)

        input_power = find_power_at_frequency(signal_100hz, sample_rate, 100.0)
        output_power = find_power_at_frequency(filtered.flatten(), sample_rate, 100.0)

        power_change = abs(input_power - output_power)
        assert power_change < 1.0, f"100Hz changed by {power_change:.1f}dB (expected <1dB)"


class TestFilterChaining:
    """Tests for combined filter configurations."""

    def test_ac_couple_plus_notch(self):
        """Combined AC coupling and notch filter."""
        sample_rate = 4000.0
        duration = 2.0
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate

        # DC + 60Hz mains + useful 200Hz signal
        dc = 3.0
        mains = 1.5 * np.sin(2 * math.pi * 60 * t)
        signal = 0.5 * np.sin(2 * math.pi * 200 * t)
        combined = (dc + mains + signal).astype(np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                ac_couple=True,
                ac_cutoff_hz=1.0,
                notch_enabled=True,
                notch_freq_hz=60.0,
                notch_q=30.0,
            )
        )
        conditioner = SignalConditioner(settings)
        chunk = make_chunk(combined, sample_rate)

        filtered = conditioner.process(chunk)
        settled = filtered.flatten()[n_samples // 4:]

        # DC should be removed
        mean_val = abs(float(np.mean(settled)))
        assert mean_val < 0.1, f"DC not removed: {mean_val:.3f}"

        # 60Hz should be attenuated
        power_60hz = find_power_at_frequency(settled, sample_rate, 60.0)
        power_200hz = find_power_at_frequency(settled, sample_rate, 200.0)

        # 200Hz should be much stronger than attenuated 60Hz
        assert power_200hz > power_60hz + 15, "60Hz not sufficiently attenuated"


class TestFilterStateContinuity:
    """Tests for filter state preservation across chunks."""

    def test_filter_state_preserved_across_chunks(self):
        """Filter state must be continuous across chunk boundaries."""
        sample_rate = 4000.0
        chunk_size = 256
        n_chunks = 5
        total_samples = chunk_size * n_chunks

        t = np.arange(total_samples) / sample_rate
        signal = np.sin(2 * math.pi * 100 * t).astype(np.float32)

        settings = FilterSettings(
            default=ChannelFilterSettings(
                lowpass_hz=200.0,
                lowpass_order=4,
            )
        )

        # Process as multiple chunks
        conditioner_chunked = SignalConditioner(settings)
        filtered_chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = make_chunk(signal[start:end], sample_rate)
            filtered = conditioner_chunked.process(chunk)
            filtered_chunks.append(filtered.flatten())

        chunked_output = np.concatenate(filtered_chunks)

        # Process as single chunk
        conditioner_single = SignalConditioner(settings)
        chunk_full = make_chunk(signal, sample_rate)
        single_output = conditioner_single.process(chunk_full).flatten()

        # Outputs should match closely
        np.testing.assert_allclose(
            chunked_output,
            single_output,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Filter state not continuous across chunks",
        )


class TestFilterValidation:
    """Tests for filter configuration validation."""

    def test_cutoff_above_nyquist_raises(self):
        """Filter cutoff above Nyquist should raise during validation."""
        sample_rate = 1000.0  # Nyquist = 500 Hz

        settings = ChannelFilterSettings(
            lowpass_hz=600.0,  # Above Nyquist
            lowpass_order=4,
        )

        with pytest.raises(ValueError):
            settings.validate(sample_rate)

    def test_negative_cutoff_raises(self):
        """Negative cutoff should raise during validation."""
        sample_rate = 1000.0

        settings = ChannelFilterSettings(
            lowpass_hz=-100.0,
            lowpass_order=4,
        )

        with pytest.raises(ValueError):
            settings.validate(sample_rate)

    def test_zero_q_factor_raises(self):
        """Zero Q factor should raise during validation."""
        sample_rate = 1000.0

        settings = ChannelFilterSettings(
            notch_enabled=True,
            notch_freq_hz=60.0,
            notch_q=0.0,
        )

        with pytest.raises(ValueError):
            settings.validate(sample_rate)
