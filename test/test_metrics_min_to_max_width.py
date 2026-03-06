"""Tests for the min_to_max_width metric.

min_to_max_width(samples, sr) returns the signed time (ms) between the sample
with the global minimum value and the sample with the global maximum value:

    result = (index_of_max - index_of_min) / sr * 1000.0

Sign convention:
    Positive → max comes after  min  (negative-first waveform, e.g. typical
               extracellular action potential trough → peak order)
    Negative → max comes before min  (positive-first waveform)
    Zero     → max and min are at the same index, or the waveform is flat
"""

import numpy as np
import pytest

from analysis.metrics import min_to_max_width


# ---------------------------------------------------------------------------
# Edge / guard cases
# ---------------------------------------------------------------------------

class TestGuardCases:
    def test_empty_array_returns_zero(self):
        assert min_to_max_width(np.array([]), 10_000.0) == 0.0

    def test_zero_sample_rate_returns_zero(self):
        samples = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        assert min_to_max_width(samples, 0.0) == 0.0

    def test_negative_sample_rate_returns_zero(self):
        samples = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        assert min_to_max_width(samples, -10_000.0) == 0.0

    def test_single_sample_returns_zero(self):
        """Single sample: argmax == argmin == 0, so difference is 0."""
        assert min_to_max_width(np.array([3.0], dtype=np.float32), 10_000.0) == 0.0

    def test_flat_signal_returns_zero(self):
        """All samples equal: argmax == argmin == 0 (numpy behaviour)."""
        samples = np.full(50, 2.5, dtype=np.float32)
        assert min_to_max_width(samples, 10_000.0) == 0.0

    def test_returns_float(self):
        samples = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        result = min_to_max_width(samples, 1_000.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Sign convention: negative-first waveforms → positive result
# ---------------------------------------------------------------------------

class TestNegativeFirstWaveform:
    """Max after min → positive width (typical extracellular AP trough-first shape)."""

    def test_two_sample_trough_then_peak(self):
        """min at index 0, max at index 1; sr=1000 Hz → 1 ms."""
        samples = np.array([-1.0, 1.0], dtype=np.float32)
        result = min_to_max_width(samples, 1_000.0)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_biphasic_negative_first(self):
        """Symmetric biphasic: trough at sample 25, peak at sample 75; sr=10 kHz."""
        sr = 10_000.0
        n = 100
        samples = np.zeros(n, dtype=np.float32)
        samples[25] = -2.0   # trough
        samples[75] = 2.0    # peak
        # Expected: (75 - 25) / 10000 * 1000 = 5.0 ms
        result = min_to_max_width(samples, sr)
        assert result == pytest.approx(5.0, abs=1e-6)

    def test_triphasic_trough_before_peak(self):
        """Typical extracellular triphasic: small positive → large trough → recovery peak."""
        sr = 30_000.0
        samples = np.zeros(90, dtype=np.float32)
        samples[10] = 0.5    # small initial positivity
        samples[30] = -3.0   # large trough (global min)
        samples[60] = 1.5    # recovery peak (global max)
        # Expected: (60 - 30) / 30000 * 1000 = 1.0 ms
        result = min_to_max_width(samples, sr)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_result_is_positive(self):
        """Any waveform where global max follows global min must give positive result."""
        rng = np.random.default_rng(42)
        sr = 20_000.0
        samples = rng.standard_normal(200).astype(np.float32)
        # Force min before max
        samples[40] = -10.0
        samples[160] = 10.0
        assert min_to_max_width(samples, sr) > 0.0


# ---------------------------------------------------------------------------
# Sign convention: positive-first waveforms → negative result
# ---------------------------------------------------------------------------

class TestPositiveFirstWaveform:
    """Max before min → negative width."""

    def test_two_sample_peak_then_trough(self):
        """max at index 0, min at index 1; sr=1000 Hz → −1 ms."""
        samples = np.array([1.0, -1.0], dtype=np.float32)
        result = min_to_max_width(samples, 1_000.0)
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_biphasic_positive_first(self):
        """Peak at sample 25, trough at sample 75; sr=10 kHz → −5 ms."""
        sr = 10_000.0
        n = 100
        samples = np.zeros(n, dtype=np.float32)
        samples[25] = 2.0    # peak
        samples[75] = -2.0   # trough
        result = min_to_max_width(samples, sr)
        assert result == pytest.approx(-5.0, abs=1e-6)

    def test_result_is_negative(self):
        """Any waveform where global max precedes global min gives negative result."""
        rng = np.random.default_rng(99)
        sr = 20_000.0
        samples = rng.standard_normal(200).astype(np.float32)
        samples[50] = 10.0   # global max early
        samples[170] = -10.0  # global min late
        assert min_to_max_width(samples, sr) < 0.0


# ---------------------------------------------------------------------------
# Numerical accuracy
# ---------------------------------------------------------------------------

class TestNumericalAccuracy:
    def test_exact_value_known_indices(self):
        """Verify arithmetic: (idx_max - idx_min) / sr * 1000."""
        sr = 25_000.0
        samples = np.zeros(250, dtype=np.float32)
        idx_min, idx_max = 37, 112
        samples[idx_min] = -5.0
        samples[idx_max] = 5.0
        expected = (idx_max - idx_min) / sr * 1000.0
        result = min_to_max_width(samples, sr)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_sample_rate_scaling(self):
        """Doubling the sample rate halves the reported width in ms."""
        samples = np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        # idx_min=1, idx_max=3, gap=2 samples
        w1 = min_to_max_width(samples, 1_000.0)   # 2/1000*1000 = 2.0 ms
        w2 = min_to_max_width(samples, 2_000.0)   # 2/2000*1000 = 1.0 ms
        assert w1 == pytest.approx(2.0, abs=1e-6)
        assert w2 == pytest.approx(1.0, abs=1e-6)
        assert w1 == pytest.approx(2 * w2, rel=1e-5)

    def test_antisymmetry(self):
        """Negating all samples swaps min and max, flipping the sign of the result."""
        rng = np.random.default_rng(7)
        sr = 10_000.0
        samples = rng.standard_normal(100).astype(np.float32)
        # Only antisymmetric if argmax != argmin (always true for random data)
        w_pos = min_to_max_width(samples, sr)
        w_neg = min_to_max_width(-samples, sr)
        assert w_pos == pytest.approx(-w_neg, rel=1e-5)

    def test_integer_array_input(self):
        """Function should handle integer dtype arrays without error."""
        samples = np.array([0, -3, 0, 5, 0], dtype=np.int16)
        result = min_to_max_width(samples, 1_000.0)
        # idx_min=1, idx_max=3 → (3-1)/1000*1000 = 2.0 ms
        assert result == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Integration: metric present in AnalysisEvent.properties
# ---------------------------------------------------------------------------

class TestIntegration:
    """Smoke-test that min_to_max_width_ms appears in event properties after
    detection_to_analysis_event()."""

    def test_property_present_and_finite(self):
        from shared.types import AnalysisEvent
        from shared.models import Chunk, DetectionEvent
        from analysis.analysis_worker import detection_to_analysis_event

        sr = 10_000.0
        dt = 1.0 / sr
        n = 100
        wf = np.zeros(n, dtype=np.float32)
        wf[20] = -2.0   # trough first
        wf[70] = 3.0    # peak second

        de = DetectionEvent(
            t=0.05,
            chan=0,
            window=wf,
            properties={},
            params={"threshold": 1.0, "pre_samples": 30},
        )

        # Build a minimal Chunk
        chunk = Chunk(
            samples=wf.reshape(1, -1),
            start_time=0.0,
            dt=dt,
            seq=0,
            channel_names=("ch0",),
            units="V",
            meta={"start_sample": 0},
        )

        event, _, _ = detection_to_analysis_event(
            de=de,
            chunk=chunk,
            event_id=1,
            sample_rate=sr,
            window_ms=10.0,
            last_crossing_time=None,
            noise_mad=0.1,
            noise_initialized=True,
        )

        props = event.properties
        assert "min_to_max_width_ms" in props, "min_to_max_width_ms missing from properties"
        val = props["min_to_max_width_ms"]
        assert isinstance(val, float)
        assert np.isfinite(val)
        # trough at 20, peak at 70 → (70-20)/10000*1000 = 5.0 ms
        assert val == pytest.approx(5.0, abs=1e-4)

    def test_property_negative_for_positive_first(self):
        """Positive-first waveform → negative min_to_max_width_ms."""
        from shared.types import AnalysisEvent
        from shared.models import Chunk, DetectionEvent
        from analysis.analysis_worker import detection_to_analysis_event

        sr = 10_000.0
        dt = 1.0 / sr
        n = 100
        wf = np.zeros(n, dtype=np.float32)
        wf[30] = 3.0    # peak first
        wf[70] = -2.0   # trough second

        de = DetectionEvent(
            t=0.05,
            chan=0,
            window=wf,
            properties={},
            params={"threshold": 1.0, "pre_samples": 30},
        )

        chunk = Chunk(
            samples=wf.reshape(1, -1),
            start_time=0.0,
            dt=dt,
            seq=0,
            channel_names=("ch0",),
            units="V",
            meta={"start_sample": 0},
        )

        event, _, _ = detection_to_analysis_event(
            de=de,
            chunk=chunk,
            event_id=2,
            sample_rate=sr,
            window_ms=10.0,
            last_crossing_time=None,
            noise_mad=0.1,
            noise_initialized=True,
        )

        val = event.properties["min_to_max_width_ms"]
        assert val < 0.0, "positive-first waveform should yield negative min_to_max_width_ms"
