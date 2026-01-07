"""
Property-based configuration fuzzing tests.

These tests use Hypothesis to generate random, potentially invalid
configuration values and verify the system handles them gracefully:
1. Invalid values are rejected with clear errors (not crashes)
2. Valid edge-case values work correctly
3. No silent acceptance of clearly wrong values

High-ROI fuzzing targets:
- Sample rate (zero, negative, NaN, huge values)
- Channel count (zero, negative, too many)
- Threshold values (NaN, Inf, negative when unexpected)
- Filter cutoffs (above Nyquist, negative)
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from core.conditioning import ChannelFilterSettings, FilterSettings
from shared.models import Chunk, ChunkPointer, DetectionEvent, TriggerConfig
from shared.ring_buffer import SharedRingBuffer


# Strategies for potentially invalid values
any_float = st.floats(allow_nan=True, allow_infinity=True)
any_int = st.integers()
weird_floats = st.sampled_from([0.0, -0.0, float("nan"), float("inf"), float("-inf"), 1e-300, 1e300])


class TestSampleRateValidation:
    """Fuzzing sample rate configuration."""

    @given(sample_rate=st.one_of(st.floats(max_value=0.0), st.just(float("nan")), st.just(float("inf")), st.just(float("-inf"))))
    @settings(max_examples=50, deadline=None)
    def test_invalid_sample_rate_rejected_in_filter_settings(self, sample_rate: float):
        """Non-positive sample rate should be rejected during filter validation."""
        settings = ChannelFilterSettings(
            lowpass_hz=100.0,  # Needs sample rate to validate
            lowpass_order=4,
        )

        # Validation should fail for invalid sample rates
        if sample_rate <= 0 or not np.isfinite(sample_rate):
            with pytest.raises((ValueError, ZeroDivisionError)):
                settings.validate(sample_rate)

    @given(sample_rate=st.floats(min_value=1.0, max_value=1e6))
    @settings(max_examples=30, deadline=None)
    def test_valid_sample_rate_accepted(self, sample_rate: float):
        """Valid sample rates should be accepted."""
        assume(np.isfinite(sample_rate))

        settings = ChannelFilterSettings()
        # No filters enabled, should validate successfully
        settings.validate(sample_rate)  # Should not raise


class TestFilterCutoffValidation:
    """Fuzzing filter cutoff frequencies."""

    @given(
        cutoff=st.floats(allow_nan=True, allow_infinity=True),
        sample_rate=st.floats(min_value=100.0, max_value=100000.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_invalid_cutoff_rejected(self, cutoff: float, sample_rate: float):
        """Cutoffs that are negative, NaN, Inf, or above Nyquist should be rejected."""
        assume(np.isfinite(sample_rate))
        nyquist = sample_rate / 2

        settings = ChannelFilterSettings(
            lowpass_hz=cutoff,
            lowpass_order=4,
        )

        if cutoff <= 0 or cutoff >= nyquist or not np.isfinite(cutoff):
            with pytest.raises(ValueError):
                settings.validate(sample_rate)

    @given(
        sample_rate=st.floats(min_value=1000.0, max_value=50000.0),
        cutoff_ratio=st.floats(min_value=0.01, max_value=0.4),
    )
    @settings(max_examples=30, deadline=None)
    def test_valid_cutoff_accepted(self, sample_rate: float, cutoff_ratio: float):
        """Valid cutoffs below Nyquist should be accepted."""
        assume(np.isfinite(sample_rate))
        cutoff = sample_rate * cutoff_ratio

        settings = ChannelFilterSettings(
            lowpass_hz=cutoff,
            lowpass_order=4,
        )
        settings.validate(sample_rate)  # Should not raise


class TestNotchFilterValidation:
    """Fuzzing notch filter parameters."""

    @given(
        q_factor=st.floats(allow_nan=True, allow_infinity=True),
    )
    @settings(max_examples=30, deadline=None)
    def test_invalid_q_factor_rejected(self, q_factor: float):
        """Q factor must be positive and finite."""
        settings = ChannelFilterSettings(
            notch_enabled=True,
            notch_freq_hz=60.0,
            notch_q=q_factor,
        )

        if q_factor <= 0 or not np.isfinite(q_factor):
            with pytest.raises(ValueError):
                settings.validate(10000.0)

    @given(
        notch_freq=st.floats(allow_nan=True, allow_infinity=True),
        sample_rate=st.floats(min_value=1000.0, max_value=50000.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_invalid_notch_freq_rejected(self, notch_freq: float, sample_rate: float):
        """Notch frequency must be positive and below Nyquist."""
        assume(np.isfinite(sample_rate))
        nyquist = sample_rate / 2

        settings = ChannelFilterSettings(
            notch_enabled=True,
            notch_freq_hz=notch_freq,
            notch_q=30.0,
        )

        if notch_freq <= 0 or notch_freq >= nyquist or not np.isfinite(notch_freq):
            with pytest.raises(ValueError):
                settings.validate(sample_rate)


class TestChunkValidation:
    """Fuzzing Chunk construction."""

    @given(
        dt=st.one_of(st.floats(max_value=0.0), st.just(float("nan")), st.just(float("inf")), st.just(float("-inf"))),
    )
    @settings(max_examples=30, deadline=None)
    def test_invalid_dt_rejected(self, dt: float):
        """dt must be positive."""
        samples = np.zeros((1, 10), dtype=np.float32)

        with pytest.raises(ValueError):
            Chunk(
                samples=samples,
                start_time=0.0,
                dt=dt,
                seq=0,
                channel_names=("Ch0",),
                units="V",
            )

    @given(seq=st.integers(max_value=-1))
    @settings(max_examples=20, deadline=None)
    def test_negative_seq_rejected(self, seq: int):
        """seq must be non-negative."""
        samples = np.zeros((1, 10), dtype=np.float32)

        with pytest.raises(ValueError):
            Chunk(
                samples=samples,
                start_time=0.0,
                dt=0.001,
                seq=seq,
                channel_names=("Ch0",),
                units="V",
            )

    @given(start_time=st.floats(allow_nan=True, allow_infinity=True))
    @settings(max_examples=30, deadline=None)
    def test_start_time_edge_cases(self, start_time: float):
        """start_time should handle edge cases."""
        samples = np.zeros((1, 10), dtype=np.float32)

        # NaN and Inf might be rejected or handled
        if not np.isfinite(start_time):
            # Either raises or handles gracefully - both acceptable
            try:
                Chunk(
                    samples=samples,
                    start_time=start_time,
                    dt=0.001,
                    seq=0,
                    channel_names=("Ch0",),
                    units="V",
                )
            except (ValueError, TypeError):
                pass  # Rejection is acceptable


class TestChunkPointerValidation:
    """Fuzzing ChunkPointer construction."""

    @given(
        start_index=st.integers(max_value=-1),
    )
    @settings(max_examples=20, deadline=None)
    def test_negative_start_index_rejected(self, start_index: int):
        """start_index must be non-negative."""
        with pytest.raises(ValueError):
            ChunkPointer(start_index=start_index, length=100, render_time=0.0)

    @given(
        length=st.integers(max_value=0),
    )
    @settings(max_examples=20, deadline=None)
    def test_non_positive_length_rejected(self, length: int):
        """length must be positive."""
        with pytest.raises(ValueError):
            ChunkPointer(start_index=0, length=length, render_time=0.0)

    @given(
        render_time=st.floats(max_value=-0.001),
    )
    @settings(max_examples=20, deadline=None)
    def test_negative_render_time_rejected(self, render_time: float):
        """render_time must be non-negative."""
        assume(np.isfinite(render_time))
        with pytest.raises(ValueError):
            ChunkPointer(start_index=0, length=100, render_time=render_time)


class TestRingBufferValidation:
    """Fuzzing ring buffer construction."""

    @given(
        capacity=st.integers(max_value=0),
        channels=st.integers(max_value=0),
    )
    @settings(max_examples=30, deadline=None)
    def test_invalid_dimensions_rejected(self, capacity: int, channels: int):
        """Zero or negative dimensions should be rejected."""
        with pytest.raises(ValueError):
            SharedRingBuffer((channels, capacity), dtype=np.float32)

    @given(
        capacity=st.integers(min_value=1, max_value=10000),
        channels=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_valid_dimensions_accepted(self, capacity: int, channels: int):
        """Valid dimensions should create a buffer."""
        buf = SharedRingBuffer((channels, capacity), dtype=np.float32)
        assert buf.capacity == capacity
        assert buf.shape == (channels, capacity)


class TestDetectionEventValidation:
    """Fuzzing DetectionEvent construction."""

    @given(
        t=st.one_of(st.floats(max_value=-0.001), st.just(float("nan"))),
    )
    @settings(max_examples=20, deadline=None)
    def test_negative_time_rejected(self, t: float):
        """Event time must be non-negative."""
        assume(np.isfinite(t))
        with pytest.raises(ValueError):
            DetectionEvent(
                t=t,
                chan=0,
                window=np.zeros(10, dtype=np.float32),
            )

    @given(
        chan=st.integers(max_value=-1),
    )
    @settings(max_examples=20, deadline=None)
    def test_negative_channel_rejected(self, chan: int):
        """Channel must be non-negative."""
        with pytest.raises(ValueError):
            DetectionEvent(
                t=0.0,
                chan=chan,
                window=np.zeros(10, dtype=np.float32),
            )


class TestTriggerConfigValidation:
    """Fuzzing TriggerConfig construction."""

    @given(
        channel_index=st.integers(),
        threshold=st.floats(allow_nan=True, allow_infinity=True),
        hysteresis=st.floats(allow_nan=True, allow_infinity=True),
        pretrigger_frac=st.floats(),
        window_sec=st.floats(),
    )
    @settings(max_examples=50, deadline=None)
    def test_trigger_config_edge_cases(
        self,
        channel_index: int,
        threshold: float,
        hysteresis: float,
        pretrigger_frac: float,
        window_sec: float,
    ):
        """TriggerConfig should handle or reject edge cases consistently."""
        # This tests that construction either succeeds or raises ValueError
        # (not other unexpected exceptions)
        try:
            config = TriggerConfig(
                channel_index=channel_index,
                threshold=threshold,
                hysteresis=hysteresis,
                pretrigger_frac=pretrigger_frac,
                window_sec=window_sec,
                mode="continuous",
            )
            # If construction succeeded, values should be stored
            assert config.channel_index == channel_index
        except (ValueError, TypeError):
            # Rejection is acceptable for invalid values
            pass


class TestCombinedConfigFuzzing:
    """Combined fuzzing of multiple configuration paths."""

    @given(
        ac_cutoff=st.floats(allow_nan=True, allow_infinity=True),
        notch_freq=st.floats(allow_nan=True, allow_infinity=True),
        notch_q=st.floats(allow_nan=True, allow_infinity=True),
        lowpass=st.floats(allow_nan=True, allow_infinity=True),
        highpass=st.floats(allow_nan=True, allow_infinity=True),
    )
    @settings(max_examples=100, deadline=None)
    def test_filter_settings_no_crash(
        self,
        ac_cutoff: float,
        notch_freq: float,
        notch_q: float,
        lowpass: float,
        highpass: float,
    ):
        """Filter settings construction should never crash, only raise ValueError."""
        try:
            settings = ChannelFilterSettings(
                ac_couple=True,
                ac_cutoff_hz=ac_cutoff,
                notch_enabled=True,
                notch_freq_hz=notch_freq,
                notch_q=notch_q,
                lowpass_hz=lowpass,
                highpass_hz=highpass,
            )
            # If construction succeeded, try validation
            try:
                settings.validate(10000.0)
            except ValueError:
                pass  # Expected for invalid configs
        except (ValueError, TypeError):
            pass  # Expected for invalid values
        except Exception as e:
            pytest.fail(f"Unexpected exception: {type(e).__name__}: {e}")
