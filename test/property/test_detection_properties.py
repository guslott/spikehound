"""
Property-based tests for spike detection using Hypothesis.

These tests verify detection invariants on randomly generated spike trains:
1. All detections must be above threshold
2. No detections can occur within refractory period of each other
3. Detection times are monotonically increasing
4. Detected spikes are within tolerance of ground-truth planted spikes

The key technique is generating spike trains with KNOWN spike times and
comparing detection output against ground truth.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from core.detection.threshold import AmpThresholdDetector
from shared.models import Chunk
from test.fixtures.signal_generators import (
    make_triphasic_spike,
    make_spike_train,
    add_gaussian_noise,
)
from test.fixtures.reference_models import (
    ReferenceThresholdDetector,
    create_detection_oracle,
)


def make_chunk(
    samples: np.ndarray,
    sample_rate: float,
    start_time: float = 0.0,
    seq: int = 0,
) -> Chunk:
    """Create a test chunk from samples."""
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    return Chunk(
        samples=samples.astype(np.float32),
        start_time=start_time,
        dt=1.0 / sample_rate,
        seq=seq,
        channel_names=tuple(f"Ch{i}" for i in range(samples.shape[0])),
        units="V",
    )


class TestDetectionInvariants:
    """Property tests for detection invariants."""

    @given(
        n_spikes=st.integers(min_value=1, max_value=20),
        snr_db=st.floats(min_value=10.0, max_value=40.0),
        refractory_ms=st.floats(min_value=0.5, max_value=5.0),
        threshold_factor=st.floats(min_value=3.0, max_value=8.0),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, deadline=None)
    def test_all_detections_above_threshold(
        self,
        n_spikes: int,
        snr_db: float,
        refractory_ms: float,
        threshold_factor: float,
        seed: int,
    ):
        """Every detection must have amplitude exceeding threshold."""
        sample_rate = 10000.0
        duration = 1.0

        # Generate random spike times with minimum spacing
        rng = np.random.default_rng(seed)
        min_spacing = refractory_ms / 1000.0 * 1.5  # Ensure spikes are spaced

        spike_times = []
        current_time = 0.02  # Start after 20ms
        for _ in range(n_spikes):
            current_time += rng.uniform(min_spacing, 0.08)
            if current_time < duration - 0.02:
                spike_times.append(current_time)
            else:
                break

        assume(len(spike_times) >= 1)

        # Create signal
        template = make_triphasic_spike(1.5, 1.0, sample_rate)
        signal, _ = make_spike_train(spike_times, template, duration, sample_rate)
        signal = add_gaussian_noise(signal, snr_db, seed=seed)

        # Estimate noise and set threshold
        noise_estimate = float(np.median(np.abs(signal)))
        threshold = -noise_estimate * threshold_factor  # Negative for negative-going spikes

        # Configure detector
        detector = AmpThresholdDetector()
        detector.configure(
            factor=threshold_factor,
            sign=-1,
            refractory_ms=refractory_ms,
        )
        detector.reset(sample_rate, n_channels=1)
        # Mocking noise levels to ensure deterministic threshold
        detector._noise_levels = np.array([noise_estimate], dtype=np.float32)

        chunk = make_chunk(signal, sample_rate)
        events = detector.process_chunk(chunk)

        # Verify all detections are below threshold (negative-going)
        for event in events:
            # Event window should have a sample below threshold
            min_val = float(np.min(event.window))
            assert min_val <= threshold * 0.8, f"Detection at {event.t}s has min={min_val}, threshold={threshold}"

    @given(
        n_spikes=st.integers(min_value=3, max_value=15),
        refractory_ms=st.floats(min_value=1.0, max_value=5.0),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, deadline=None)
    def test_refractory_period_respected(
        self,
        n_spikes: int,
        refractory_ms: float,
        seed: int,
    ):
        """No two detections can occur within refractory period."""
        sample_rate = 10000.0
        duration = 1.0

        # Generate spike train with enough spacing for detection
        rng = np.random.default_rng(seed)
        min_spacing = refractory_ms / 1000.0 * 0.8  # Allow some close spikes

        spike_times = []
        current_time = 0.02
        for _ in range(n_spikes):
            current_time += rng.uniform(min_spacing, 0.1)
            if current_time < duration - 0.02:
                spike_times.append(current_time)

        assume(len(spike_times) >= 2)

        template = make_triphasic_spike(1.5, 1.0, sample_rate)
        signal, _ = make_spike_train(spike_times, template, duration, sample_rate)

        detector = AmpThresholdDetector()
        detector.configure(
            factor=4.0,
            sign=-1,
            refractory_ms=refractory_ms,
        )
        detector.reset(sample_rate, n_channels=1)
        # Force a predictable threshold (e.g., 0.5)
        detector._noise_levels = np.array([0.1], dtype=np.float32)
        detector._factor = 5.0  # 0.1 * 5.0 = 0.5 threshold

        chunk = make_chunk(signal, sample_rate)
        events = detector.process_chunk(chunk)

        # Check spacing between consecutive detections
        refractory_sec = refractory_ms / 1000.0
        event_times = [e.t for e in events]

        for i in range(1, len(event_times)):
            interval = event_times[i] - event_times[i - 1]
            # Allow small tolerance for timing uncertainty
            assert interval >= refractory_sec * 0.95, \
                f"Refractory violated: {interval*1000:.2f}ms < {refractory_ms}ms"

    @given(
        n_spikes=st.integers(min_value=5, max_value=20),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_detection_times_monotonic(
        self,
        n_spikes: int,
        seed: int,
    ):
        """Detection times must be strictly increasing."""
        sample_rate = 10000.0
        duration = 1.5

        rng = np.random.default_rng(seed)
        spike_times = sorted(rng.uniform(0.02, duration - 0.02, n_spikes))

        template = make_triphasic_spike(1.5, 1.0, sample_rate)
        signal, _ = make_spike_train(spike_times, template, duration, sample_rate)

        detector = AmpThresholdDetector()
        detector.configure(factor=4.0, sign=-1, refractory_ms=1.0)
        detector.reset(sample_rate, n_channels=1)
        detector._noise_levels = np.array([0.1], dtype=np.float32)

        chunk = make_chunk(signal, sample_rate)
        events = detector.process_chunk(chunk)

        event_times = [e.t for e in events]
        for i in range(1, len(event_times)):
            assert event_times[i] > event_times[i - 1], "Detection times not monotonic"


class TestDetectionAccuracy:
    """Tests for detection accuracy against ground truth."""

    @given(
        n_spikes=st.integers(min_value=5, max_value=15),
        snr_db=st.floats(min_value=15.0, max_value=30.0),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_detection_finds_planted_spikes(
        self,
        n_spikes: int,
        snr_db: float,
        seed: int,
    ):
        """Detector should find most planted spikes at reasonable SNR."""
        sample_rate = 10000.0
        duration = 2.0

        # Generate well-spaced spike train
        rng = np.random.default_rng(seed)
        spike_times = sorted(rng.uniform(0.05, duration - 0.05, n_spikes))

        # Ensure minimum spacing
        filtered_times = [spike_times[0]]
        for t in spike_times[1:]:
            if t - filtered_times[-1] >= 0.02:  # 20ms minimum
                filtered_times.append(t)
        spike_times = filtered_times

        assume(len(spike_times) >= 3)

        template = make_triphasic_spike(1.5, 1.0, sample_rate)
        signal, ground_truth_samples = make_spike_train(
            spike_times, template, duration, sample_rate
        )
        signal = add_gaussian_noise(signal, snr_db, seed=seed)

        detector = AmpThresholdDetector()
        detector.configure(factor=4.0, sign=-1, refractory_ms=2.0)
        detector.reset(sample_rate, n_channels=1)
        detector._noise_levels = np.array([0.1], dtype=np.float32)

        chunk = make_chunk(signal, sample_rate)
        events = detector.process_chunk(chunk)

        # Convert detections to sample indices
        detected_samples = np.array([int(e.t * sample_rate) for e in events])

        # Use oracle to compare
        tolerance = int(sample_rate * 0.002)  # 2ms tolerance
        tp, fp, fn = create_detection_oracle(
            ground_truth_samples,
            detected_samples,
            tolerance,
        )

        # At SNR >= 15dB, we expect reasonable detection
        # Allow some false negatives but should find majority
        n_truth = len(spike_times)
        if n_truth > 0:
            recall = tp / n_truth
            # Expect at least 50% recall at this SNR
            assert recall >= 0.5, f"Low recall: {recall:.2%} (found {tp}/{n_truth})"

    @given(
        duration=st.floats(min_value=0.5, max_value=2.0),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_no_false_positives_on_noise_only(
        self,
        duration: float,
        seed: int,
    ):
        """Detector should not produce false positives on pure noise."""
        sample_rate = 10000.0
        n_samples = int(duration * sample_rate)

        rng = np.random.default_rng(seed)
        # Low-amplitude noise (well below typical thresholds)
        noise = rng.normal(0, 0.01, n_samples).astype(np.float32)

        detector = AmpThresholdDetector()
        detector.configure(factor=6.0, sign=-1, refractory_ms=2.0)
        detector.reset(sample_rate, n_channels=1)
        detector._noise_levels = np.array([0.1], dtype=np.float32)

        chunk = make_chunk(noise, sample_rate)
        events = detector.process_chunk(chunk)

        # Should have no or very few false positives
        assert len(events) <= 2, f"Too many false positives on noise: {len(events)}"


class TestCrossChunkDetection:
    """Tests for detection continuity across chunk boundaries."""

    @given(
        spike_position=st.floats(min_value=0.4, max_value=0.6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_spike_at_chunk_boundary_detected(
        self,
        spike_position: float,
        seed: int,
    ):
        """Spike straddling chunk boundary should still be detected."""
        sample_rate = 10000.0
        duration = 0.2
        n_samples = int(duration * sample_rate)
        chunk_boundary = n_samples // 2

        # Place spike near boundary
        spike_time = chunk_boundary / sample_rate * spike_position * 2

        template = make_triphasic_spike(1.5, 1.0, sample_rate)
        signal, _ = make_spike_train([spike_time], template, duration, sample_rate)

        detector = AmpThresholdDetector()
        detector.configure(factor=4.0, sign=-1, refractory_ms=1.0)
        detector.reset(sample_rate, n_channels=1)
        detector._noise_levels = np.array([0.1], dtype=np.float32)

        # Process as two chunks
        chunk1 = make_chunk(
            signal[:chunk_boundary],
            sample_rate,
            start_time=0.0,
            seq=0,
        )
        chunk2 = make_chunk(
            signal[chunk_boundary:],
            sample_rate,
            start_time=chunk_boundary / sample_rate,
            seq=1,
        )

        events1 = detector.process_chunk(chunk1)
        events2 = detector.process_chunk(chunk2)

        total_events = len(events1) + len(events2)

        # Should detect the spike (either in chunk1 or chunk2)
        assert total_events >= 1, "Spike at boundary not detected"


class TestReferenceModelComparison:
    """Compare production detector against reference implementation."""

    @given(
        n_spikes=st.integers(min_value=3, max_value=10),
        threshold=st.floats(min_value=-0.5, max_value=-0.1),
        refractory=st.integers(min_value=10, max_value=50),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_matches_reference_detector(
        self,
        n_spikes: int,
        threshold: float,
        refractory: int,
        seed: int,
    ):
        """Production detector should match reference on clean signals."""
        sample_rate = 10000.0
        duration = 1.0

        # Generate clean spike train
        rng = np.random.default_rng(seed)
        spike_times = sorted(
            rng.uniform(0.02, duration - 0.02, n_spikes)
        )

        # Ensure spacing
        filtered = [spike_times[0]]
        min_spacing = (refractory * 2) / sample_rate
        for t in spike_times[1:]:
            if t - filtered[-1] >= min_spacing:
                filtered.append(t)

        assume(len(filtered) >= 2)

        template = make_triphasic_spike(1.5, 1.0, sample_rate)
        signal, _ = make_spike_train(filtered, template, duration, sample_rate)

        # Reference detector
        ref_detector = ReferenceThresholdDetector(
            threshold=threshold,
            refractory_samples=refractory,
            polarity=-1,
        )
        ref_detections = ref_detector.detect(signal)

        # Production detector with equivalent settings
        refractory_ms = refractory / sample_rate * 1000

        prod_detector = AmpThresholdDetector()
        prod_detector.reset(sample_rate, n_channels=1)
        # Force specific threshold by setting noise=1.0 and factor=abs(threshold)
        prod_detector._noise_levels = np.array([1.0], dtype=np.float32)
        prod_detector._factor = float(abs(threshold))
        prod_detector._sign = -1
        prod_detector._refractory_ms = refractory_ms

        chunk = make_chunk(signal, sample_rate)
        prod_events = prod_detector.process_chunk(chunk)
        prod_detections = [int(round(e.t * sample_rate)) for e in prod_events]

        # Compare detection counts (should be similar, not necessarily identical
        # due to implementation differences in peak finding)
        if len(ref_detections) > 0:
            ratio = len(prod_detections) / len(ref_detections)
            assert 0.5 <= ratio <= 2.0, \
                f"Detection count mismatch: ref={len(ref_detections)}, prod={len(prod_detections)}"
