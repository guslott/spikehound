"""
Tests for EWMA noise adaptation in AmpThresholdDetector.

The original detector computed noise level from the *first* chunk only.  In a
classroom lab the user often plugs in an electrode while the scope is already
running, so the first chunk is dominated by the insertion transient.  The MAD
of that chunk is ~10× larger than the true background noise, and the threshold
was permanently set 10× too high for the rest of the session.

The fix: replace the one-shot ``if _noise_levels is None`` guard with an
Exponential Weighted Moving Average (EWMA) update on every chunk.  The time
constant (alpha=0.1) matches RealTimeAnalyzer._maybe_update_auto_thresholds().

These tests verify:

  1. The threshold is no longer frozen after the first chunk.
  2. After a large startup transient, the threshold adapts back down toward the
     true noise floor over subsequent normal chunks.
  3. After the threshold recovers, actual spikes are detected again.
  4. Pure-noise input converges to the correct σ estimate.
  5. Flat-line input (disconnected electrode) never produces a zero threshold.
  6. Multi-channel noise levels are estimated independently per channel.
  7. `reset()` clears the EWMA state, starting fresh on the next chunk.
  8. Existing single-chunk behaviour is unchanged (regression guard).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.detection import AmpThresholdDetector
from shared.models import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FS = 10_000.0    # 10 kHz — typical electrophysiology sample rate
_DT = 1.0 / _FS


def _make_chunk(
    samples: np.ndarray,
    *,
    start_time: float = 0.0,
    seq: int = 0,
) -> Chunk:
    """Wrap a (n_channels, n_samples) array in a Chunk."""
    n_ch = samples.shape[0]
    return Chunk(
        samples=samples.astype(np.float32),
        start_time=start_time,
        dt=_DT,
        seq=seq,
        channel_names=tuple(f"ch{i}" for i in range(n_ch)),
        units="V",
        meta={"start_sample": seq * samples.shape[1]},
    )


def _noise_chunk(sigma: float, n_samples: int = 1_000, n_ch: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(n_ch, n_samples)).astype(np.float32)


def _spike_chunk(sigma: float, spike_amp: float, spike_pos: int = 500,
                 n_samples: int = 1_000, n_ch: int = 1, seed: int = 0) -> np.ndarray:
    data = _noise_chunk(sigma, n_samples, n_ch, seed)
    data[0, spike_pos] = spike_amp
    return data


def _make_detector(factor: float = 5.0, sign: int = -1) -> AmpThresholdDetector:
    d = AmpThresholdDetector()
    d.configure(factor=factor, sign=sign, refractory_ms=1.0, window_ms=2.0)
    d.reset(_FS, 1)
    return d


# ---------------------------------------------------------------------------
# 1. Threshold is not frozen after the first chunk
# ---------------------------------------------------------------------------

class TestThresholdNotFrozen:

    def test_noise_levels_update_on_second_chunk(self):
        """_noise_levels must change between chunk 1 and chunk 2."""
        d = _make_detector()
        chunk1 = _make_chunk(_noise_chunk(0.1, seed=1), seq=0)
        d.process_chunk(chunk1)
        level_after_chunk1 = d._noise_levels.copy()

        chunk2 = _make_chunk(_noise_chunk(0.5, seed=2), seq=1)  # different σ
        d.process_chunk(chunk2)
        level_after_chunk2 = d._noise_levels.copy()

        assert not np.allclose(level_after_chunk1, level_after_chunk2), (
            "Noise level did not change after chunk 2 — threshold is frozen (the bug)"
        )

    def test_noise_levels_change_every_chunk(self):
        """Each chunk must nudge the EWMA; 10 consecutive chunks must all produce
        distinct _noise_levels values."""
        d = _make_detector()
        rng = np.random.default_rng(42)
        levels = []
        for i in range(10):
            sigma = rng.uniform(0.05, 0.5)
            chunk = _make_chunk(rng.normal(0, sigma, (1, 1000)).astype(np.float32), seq=i)
            d.process_chunk(chunk)
            levels.append(d._noise_levels[0])
        # All 10 levels must be different
        assert len(set(np.round(levels, 8))) == 10, (
            "Some _noise_levels are identical across consecutive chunks — EWMA not running"
        )


# ---------------------------------------------------------------------------
# 2. Threshold recovers from a startup transient
# ---------------------------------------------------------------------------

class TestTransientRecovery:
    """The critical classroom scenario: electrode plugged in while scope runs."""

    def _converged_level(self, sigma: float, n_chunks: int = 200) -> float:
        """Return the converged noise level for pure noise with given sigma."""
        d = _make_detector()
        for i in range(n_chunks):
            d.process_chunk(_make_chunk(_noise_chunk(sigma, seed=i), seq=i))
        return float(d._noise_levels[0])

    def test_threshold_decays_after_transient(self):
        """
        After a 10× transient on chunk 0, feeding 60 subsequent normal-noise
        chunks must bring _noise_levels below 3× the converged normal level.
        """
        sigma = 0.1
        transient_sigma = sigma * 10      # 10× insertion artifact

        d = _make_detector()
        # Chunk 0: transient
        d.process_chunk(_make_chunk(_noise_chunk(transient_sigma, seed=99), seq=0))
        level_after_transient = float(d._noise_levels[0])

        # Chunks 1..60: normal noise
        for i in range(1, 61):
            d.process_chunk(_make_chunk(_noise_chunk(sigma, seed=i), seq=i))
        level_after_recovery = float(d._noise_levels[0])

        converged = self._converged_level(sigma)

        assert level_after_transient > converged * 5, (
            "Transient did not raise threshold as expected (test precondition)"
        )
        assert level_after_recovery < converged * 3, (
            f"After 60 normal chunks the threshold ({level_after_recovery:.4f}) is still "
            f"more than 3× the converged level ({converged:.4f}). "
            "EWMA is not adapting fast enough or not running."
        )

    def test_threshold_continues_to_fall_over_time(self):
        """The threshold must strictly decrease toward the noise floor as more
        normal chunks arrive after a transient."""
        sigma = 0.1
        d = _make_detector()
        d.process_chunk(_make_chunk(_noise_chunk(sigma * 8, seed=0), seq=0))

        levels = []
        for i in range(1, 30):
            d.process_chunk(_make_chunk(_noise_chunk(sigma, seed=i), seq=i))
            levels.append(float(d._noise_levels[0]))

        # Each level must be less than or equal to the previous one.
        for j in range(1, len(levels)):
            assert levels[j] <= levels[j - 1] + 1e-7, (
                f"Threshold rose at step {j}: {levels[j-1]:.5f} → {levels[j]:.5f}"
            )


# ---------------------------------------------------------------------------
# 3. Spikes detected after threshold recovers
# ---------------------------------------------------------------------------

class TestSpikeDetectionAfterRecovery:
    """End-to-end check: spikes must be found once the EWMA has settled."""

    def test_spikes_detected_after_transient_decays(self):
        """
        Simulate the classroom scenario:
          - Chunks 0–4: insertion transient (no spikes injected)
          - Chunks 5–50: normal noise + clear spikes
        Spikes should be detected in at least the last several chunks.
        """
        sigma = 0.1
        spike_amp = -sigma * 15    # spike is 15× σ below baseline (very clear)
        transient_sigma = sigma * 10

        d = _make_detector(factor=5.0, sign=-1)

        # Transient chunks
        for i in range(5):
            chunk = _make_chunk(_noise_chunk(transient_sigma, seed=i), seq=i)
            d.process_chunk(chunk)

        # Normal chunks with spikes — count detections in the second half only
        # (the first few chunks after the transient may still have a high threshold)
        total_late_spikes = 0
        rng = np.random.default_rng(7)
        for i in range(5, 51):
            t_start = i * 1000 * _DT
            data = rng.normal(0, sigma, (1, 1000)).astype(np.float32)
            data[0, 500] = spike_amp
            chunk = _make_chunk(data, start_time=t_start, seq=i)
            events = list(d.process_chunk(chunk))
            if i >= 35:
                total_late_spikes += len(events)

        assert total_late_spikes >= 10, (
            f"Only {total_late_spikes} spikes detected in chunks 35–50; "
            "threshold has not recovered from the startup transient."
        )

    def test_spikes_detected_from_start_without_transient(self):
        """Without a transient, spikes must be detected from the very first chunk
        (regression check: EWMA must not break detection on clean input)."""
        sigma = 0.1
        spike_amp = -sigma * 15
        d = _make_detector(factor=5.0, sign=-1)

        detected = 0
        rng = np.random.default_rng(3)
        for i in range(10):
            t_start = i * 1000 * _DT
            data = rng.normal(0, sigma, (1, 1000)).astype(np.float32)
            data[0, 500] = spike_amp
            chunk = _make_chunk(data, start_time=t_start, seq=i)
            events = list(d.process_chunk(chunk))
            detected += len(events)

        assert detected == 10, (
            f"Expected 10 spikes (1 per chunk), got {detected}. "
            "EWMA may have broken normal detection."
        )


# ---------------------------------------------------------------------------
# 4. Convergence to the true noise floor
# ---------------------------------------------------------------------------

class TestEWMAConvergence:
    """After many chunks of stationary noise, _noise_levels must converge close
    to the true σ (within 20%)."""

    @pytest.mark.parametrize("sigma", [0.05, 0.1, 0.5])
    def test_converges_to_true_sigma(self, sigma: float):
        d = _make_detector()
        for i in range(200):
            d.process_chunk(_make_chunk(_noise_chunk(sigma, seed=i), seq=i))
        estimated = float(d._noise_levels[0])
        assert abs(estimated - sigma) / sigma < 0.20, (
            f"Converged noise estimate {estimated:.4f} deviates more than 20% "
            f"from true σ={sigma:.4f}"
        )

    def test_estimate_is_monotonically_improving(self):
        """The EWMA error relative to the true σ must shrink over time."""
        sigma = 0.1
        d = _make_detector()
        errors = []
        for i in range(100):
            d.process_chunk(_make_chunk(_noise_chunk(sigma, seed=i), seq=i))
            errors.append(abs(float(d._noise_levels[0]) - sigma))
        # Error in the second half must be less than error in the first half
        first_half = np.mean(errors[:50])
        second_half = np.mean(errors[50:])
        assert second_half < first_half, (
            f"EWMA error did not improve over time: "
            f"first-half mean={first_half:.4f}, second-half mean={second_half:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Zero guard — flat-line input
# ---------------------------------------------------------------------------

class TestZeroGuard:

    def test_flat_line_does_not_produce_zero_threshold(self):
        """A fully silent (disconnected electrode) input must not produce a
        zero threshold that trips on every subsequent sample."""
        d = _make_detector()
        flat = np.zeros((1, 1000), dtype=np.float32)
        for i in range(10):
            d.process_chunk(_make_chunk(flat, seq=i))
        assert d._noise_levels[0] >= 1e-10, (
            f"Zero threshold after flat-line input: {d._noise_levels[0]}"
        )

    def test_flat_line_then_normal_noise_still_detects_spikes(self):
        """After some flat-line chunks, switching to normal noise + spikes
        should eventually produce detections."""
        sigma = 0.1
        spike_amp = -sigma * 15
        d = _make_detector(factor=5.0, sign=-1)

        for i in range(5):
            d.process_chunk(_make_chunk(np.zeros((1, 1000), dtype=np.float32), seq=i))

        # Feed normal noise + spikes; count detections in last 20 chunks
        rng = np.random.default_rng(55)
        late_detections = 0
        for i in range(5, 55):
            t_start = i * 1000 * _DT
            data = rng.normal(0, sigma, (1, 1000)).astype(np.float32)
            data[0, 500] = spike_amp
            chunk = _make_chunk(data, start_time=t_start, seq=i)
            events = list(d.process_chunk(chunk))
            if i >= 35:
                late_detections += len(events)

        assert late_detections >= 10, (
            f"Only {late_detections} spikes after flat-line warmup; "
            "threshold may have been stuck at the fallback value."
        )


# ---------------------------------------------------------------------------
# 6. Multi-channel independence
# ---------------------------------------------------------------------------

class TestMultiChannel:

    def test_each_channel_has_independent_noise_estimate(self):
        """Two channels with different σ values must produce different
        _noise_levels entries after convergence."""
        d = AmpThresholdDetector()
        d.configure(factor=5.0, sign=-1, refractory_ms=1.0, window_ms=2.0)
        d.reset(_FS, 2)

        rng = np.random.default_rng(0)
        sigma_ch0, sigma_ch1 = 0.1, 0.5
        for i in range(150):
            data = np.vstack([
                rng.normal(0, sigma_ch0, (1, 1000)),
                rng.normal(0, sigma_ch1, (1, 1000)),
            ]).astype(np.float32)
            d.process_chunk(_make_chunk(data, seq=i))

        est0 = float(d._noise_levels[0])
        est1 = float(d._noise_levels[1])
        # Channel 1 must have a clearly higher estimate than channel 0
        assert est1 > est0 * 2, (
            f"Channel estimates are not independent: ch0={est0:.4f}, ch1={est1:.4f}"
        )
        # Both must be reasonably close to the true σ
        assert abs(est0 - sigma_ch0) / sigma_ch0 < 0.25
        assert abs(est1 - sigma_ch1) / sigma_ch1 < 0.25


# ---------------------------------------------------------------------------
# 7. reset() clears EWMA state
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_clears_noise_levels(self):
        """reset() must set _noise_levels back to None so the next chunk
        initialises fresh rather than blending into stale history."""
        d = _make_detector()
        d.process_chunk(_make_chunk(_noise_chunk(0.5, seed=0), seq=0))
        assert d._noise_levels is not None

        d.reset(_FS, 1)
        assert d._noise_levels is None, (
            "reset() did not clear _noise_levels; stale transient data persists"
        )

    def test_after_reset_initialises_from_first_chunk(self):
        """After reset(), the first new chunk must set _noise_levels directly
        (no EWMA blend with previous history)."""
        d = _make_detector()
        # Prime with high-noise history
        for i in range(20):
            d.process_chunk(_make_chunk(_noise_chunk(1.0, seed=i), seq=i))

        d.reset(_FS, 1)

        # Now feed a low-noise chunk
        low_sigma = 0.01
        chunk = _make_chunk(_noise_chunk(low_sigma, seed=99), seq=0)
        d.process_chunk(chunk)

        # If reset worked, the level should be close to low_sigma, not blended
        # with the high-noise history
        assert float(d._noise_levels[0]) < 0.1, (
            f"After reset(), _noise_levels={d._noise_levels[0]:.4f} suggests "
            "stale history was not cleared."
        )


# ---------------------------------------------------------------------------
# 8. Regression: single-chunk behaviour unchanged
# ---------------------------------------------------------------------------

class TestSingleChunkRegression:
    """Verify that the EWMA change does not break existing single-chunk tests.
    On the first chunk, the EWMA initialises directly (no blend), so
    behaviour is identical to the old one-shot code."""

    def test_spike_detected_in_first_chunk(self):
        sigma = 0.1
        spike_amp = -sigma * 15
        rng = np.random.default_rng(42)
        data = rng.normal(0, sigma, (1, 1000)).astype(np.float32)
        data[0, 500] = spike_amp

        d = _make_detector(factor=5.0, sign=-1)
        events = list(d.process_chunk(_make_chunk(data, seq=0)))
        assert len(events) >= 1

    def test_noise_level_on_first_chunk_equals_mad_estimate(self):
        """On the very first chunk, _noise_levels must equal MAD/0.6745
        (no EWMA blending yet)."""
        sigma = 0.1
        rng = np.random.default_rng(7)
        data = rng.normal(0, sigma, (1, 2000)).astype(np.float32)

        d = _make_detector()
        d.process_chunk(_make_chunk(data, seq=0))

        med = np.median(data, axis=1, keepdims=True)
        mad = np.median(np.abs(data - med), axis=1)
        expected_sigma = float(mad[0] / 0.6745)

        assert abs(float(d._noise_levels[0]) - expected_sigma) < 1e-6, (
            "First-chunk initialisation does not match MAD/0.6745 directly"
        )
