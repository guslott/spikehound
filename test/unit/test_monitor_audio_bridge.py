"""
Unit tests for MonitorAudioBridge.

These tests verify the bridge's core contracts without requiring a live
AudioPlayer, real hardware, or the Qt GUI:

  - Filter parity: bridge produces the same output as a standalone
    SignalConditioner given the same settings (uniform and named-override paths)
  - Gain: set_gain() scales output proportionally
  - Channel selection: listen_channel_idx determines which channel is extracted
  - Out-of-range / None channel: bridge writes nothing (no crash)
  - update_filter_settings(): new settings take effect on the next chunk
  - update_channel_names(): new names are visible to per-channel overrides
  - Underrun behaviour: on_chunk() silently skips when listen_idx is None
  - Thread safety: concurrent set_gain + on_chunk does not corrupt output
"""
from __future__ import annotations

import threading
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.conditioning import ChannelFilterSettings, FilterSettings, SignalConditioner
from core.monitor_audio_bridge import MonitorAudioBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44_100.0
N_FRAMES = 256


def _make_player_mock() -> tuple[MagicMock, list[np.ndarray]]:
    """Return (mock_player, written_chunks_list).

    ``written_chunks_list`` accumulates every array passed to _ring_write.
    """
    written: list[np.ndarray] = []

    def _capture(samples: np.ndarray) -> None:
        written.append(samples.copy())

    mock = MagicMock()
    mock._ring_write.side_effect = _capture
    return mock, written


def _raw_chunk(
    n_channels: int,
    n_frames: int,
    *,
    rng: np.random.Generator,
    scale: float = 0.5,
) -> np.ndarray:
    """Return a random float32 array shaped (n_frames, n_channels).

    ``scale`` is set to 0.5 by default so that max|sample| < 1.0, which keeps
    values well below the bridge's ±1.5 hard-clip and allows bit-exact
    comparisons between bridge output and expected values.
    """
    data = rng.standard_normal((n_frames, n_channels)).astype(np.float32)
    max_abs = float(np.abs(data).max())
    if max_abs > 0:
        data *= scale / max_abs
    return data


def _bridge(
    n_channels: int,
    filter_settings: Optional[FilterSettings] = None,
    listen_channel_idx: int = 0,
    gain: float = 1.0,
    channel_names: Optional[list[str]] = None,
) -> tuple[MonitorAudioBridge, list[np.ndarray]]:
    """Create a bridge wired to a mock player and return (bridge, written_list)."""
    player, written = _make_player_mock()
    if filter_settings is None:
        filter_settings = FilterSettings()
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]
    bridge = MonitorAudioBridge(
        player=player,
        filter_settings=filter_settings,
        sample_rate=SAMPLE_RATE,
        n_channels=n_channels,
        channel_names=channel_names,
        listen_channel_idx=listen_channel_idx,
        gain=gain,
    )
    return bridge, written


def _sine_chunk(
    freq_hz: float,
    n_frames: int,
    offset: int,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Return a continuous sine chunk at given offset (sample count from t=0),
    shaped (n_frames, 1), with max amplitude << 1.5 to avoid hard-clip."""
    t = (offset + np.arange(n_frames, dtype=np.float64)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32).reshape(n_frames, 1)


# ---------------------------------------------------------------------------
# Filter parity
# ---------------------------------------------------------------------------

class TestFilterParity:
    """Bridge output must match a standalone SignalConditioner for the same spec."""

    def _reference_output(
        self,
        settings: FilterSettings,
        samples_nch_nfr: np.ndarray,
        channel_names: list[str],
    ) -> np.ndarray:
        """Run SignalConditioner directly and return (n_channels, n_frames) result."""
        from shared.models import Chunk

        chunk = Chunk(
            samples=samples_nch_nfr,
            start_time=0.0,
            dt=1.0 / SAMPLE_RATE,
            seq=0,
            channel_names=tuple(channel_names),
            units="V",
        )
        return SignalConditioner(settings).process(chunk)

    def test_bypass_parity(self) -> None:
        """With no filters, bridge writes input channel unchanged."""
        rng = np.random.default_rng(0)
        n_ch, listen = 3, 1
        # scale=0.5 → max |sample| = 0.5, safely below the ±1.5 hard-clip
        raw = _raw_chunk(n_ch, N_FRAMES, rng=rng, scale=0.5)
        names = [f"ch{i}" for i in range(n_ch)]

        bridge, written = _bridge(n_ch, FilterSettings(), listen_channel_idx=listen,
                                   channel_names=names)
        bridge.on_chunk(raw)

        assert len(written) == 1
        np.testing.assert_allclose(
            written[0], raw[:, listen],
            atol=1e-6,
            err_msg="Bypass bridge output does not match raw channel",
        )

    def test_notch_filter_parity(self) -> None:
        """Bridge with a notch filter produces the same output as SignalConditioner."""
        rng = np.random.default_rng(1)
        n_ch, listen = 2, 0
        settings = FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True, notch_freq_hz=60.0, notch_q=30.0
            )
        )
        names = [f"ch{i}" for i in range(n_ch)]
        raw = _raw_chunk(n_ch, N_FRAMES, rng=rng, scale=0.5)

        # Bridge path
        bridge, written = _bridge(n_ch, settings, listen_channel_idx=listen,
                                   channel_names=names)
        bridge.on_chunk(raw)

        # Reference path (raw is (n_frames, n_ch); conditioner expects (n_ch, n_frames))
        ref_out = self._reference_output(settings, raw.T.copy(), names)

        np.testing.assert_allclose(
            written[0], ref_out[listen],
            atol=1e-5,
            err_msg="Notch bridge output diverges from SignalConditioner reference",
        )

    def test_named_override_parity(self) -> None:
        """Per-channel overrides keyed by *name* resolve correctly in the bridge."""
        rng = np.random.default_rng(2)
        n_ch = 2
        names = ["electrode_a", "electrode_b"]

        base = ChannelFilterSettings(
            notch_enabled=True, notch_freq_hz=60.0, notch_q=30.0
        )
        override_b = ChannelFilterSettings(
            lowpass_hz=300.0, lowpass_order=4
        )
        settings = FilterSettings(default=base, overrides={"electrode_b": override_b})

        raw = _raw_chunk(n_ch, N_FRAMES, rng=rng, scale=0.5)

        # Bridge path (listen on ch1 = "electrode_b")
        bridge, written = _bridge(n_ch, settings, listen_channel_idx=1,
                                   channel_names=names)
        bridge.on_chunk(raw)

        # Reference path
        ref_out = self._reference_output(settings, raw.T.copy(), names)

        np.testing.assert_allclose(
            written[0], ref_out[1],
            atol=1e-5,
            err_msg="Named per-channel override output diverges from reference",
        )


# ---------------------------------------------------------------------------
# Gain
# ---------------------------------------------------------------------------

class TestGain:

    def test_gain_scales_output(self) -> None:
        """Output amplitude must be proportional to gain."""
        rng = np.random.default_rng(10)
        # scale=0.5 keeps |sample| << 1.5, so clipping is not a factor
        raw = _raw_chunk(1, N_FRAMES, rng=rng, scale=0.5)

        bridge_1x, w1 = _bridge(1, gain=1.0)
        bridge_halfx, w2 = _bridge(1, gain=0.5)

        bridge_1x.on_chunk(raw)
        bridge_halfx.on_chunk(raw)

        np.testing.assert_allclose(
            w2[0], w1[0] * 0.5,
            atol=1e-6,
            err_msg="0.5× gain did not halve the output",
        )

    def test_set_gain_takes_effect(self) -> None:
        """set_gain() must change the amplitude used on the *next* chunk."""
        rng = np.random.default_rng(11)
        raw = _raw_chunk(1, N_FRAMES, rng=rng, scale=0.5)

        bridge, written = _bridge(1, gain=1.0)
        bridge.on_chunk(raw)          # chunk 0 at gain=1.0
        bridge.set_gain(0.25)
        bridge.on_chunk(raw)          # chunk 1 at gain=0.25

        np.testing.assert_allclose(
            written[1], written[0] * 0.25,
            atol=1e-6,
            err_msg="set_gain() did not update output gain",
        )

    def test_zero_gain_silences_output(self) -> None:
        """gain=0 must produce all-zeros output."""
        rng = np.random.default_rng(12)
        raw = _raw_chunk(2, N_FRAMES, rng=rng, scale=0.5)

        bridge, written = _bridge(2, gain=0.0)
        bridge.on_chunk(raw)

        np.testing.assert_array_equal(
            written[0], np.zeros(N_FRAMES),
            err_msg="gain=0 did not silence output",
        )


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------

class TestChannelSelection:

    def test_selects_correct_channel(self) -> None:
        """Each channel index produces a distinct output."""
        n_ch = 4
        rng = np.random.default_rng(20)
        raw = _raw_chunk(n_ch, N_FRAMES, rng=rng, scale=0.5)

        for listen in range(n_ch):
            bridge, written = _bridge(n_ch, listen_channel_idx=listen)
            bridge.on_chunk(raw)
            # With bypass and safe amplitude, output must equal the selected column
            np.testing.assert_allclose(
                written[0], raw[:, listen],
                atol=1e-6,
                err_msg=f"Channel {listen}: extracted wrong samples",
            )

    def test_out_of_range_idx_writes_nothing(self) -> None:
        """A listen_idx that exceeds n_channels must write nothing (no crash)."""
        rng = np.random.default_rng(21)
        raw = _raw_chunk(2, N_FRAMES, rng=rng)

        bridge, written = _bridge(2, listen_channel_idx=99)
        bridge.on_chunk(raw)   # must not raise

        assert len(written) == 0, "Out-of-range idx wrote data unexpectedly"

    def test_none_idx_writes_nothing(self) -> None:
        """listen_channel_idx=None must write nothing (no crash)."""
        rng = np.random.default_rng(22)
        raw = _raw_chunk(2, N_FRAMES, rng=rng)

        bridge, written = _bridge(2, listen_channel_idx=None)
        bridge.on_chunk(raw)

        assert len(written) == 0, "None idx wrote data unexpectedly"

    def test_set_listen_channel_idx_updates(self) -> None:
        """set_listen_channel_idx() must change which channel is written."""
        n_ch = 3
        rng = np.random.default_rng(23)
        raw = _raw_chunk(n_ch, N_FRAMES, rng=rng, scale=0.5)

        bridge, written = _bridge(n_ch, listen_channel_idx=0)
        bridge.on_chunk(raw)           # ch0
        bridge.set_listen_channel_idx(2)
        bridge.on_chunk(raw)           # ch2

        np.testing.assert_allclose(written[0], raw[:, 0], atol=1e-6)
        np.testing.assert_allclose(
            written[1], raw[:, 2], atol=1e-6,
            err_msg="set_listen_channel_idx(2) did not switch channel",
        )


# ---------------------------------------------------------------------------
# Settings updates
# ---------------------------------------------------------------------------

class TestSettingsUpdates:

    def test_update_filter_settings_takes_effect(self) -> None:
        """update_filter_settings() must change filtering on the next chunk.

        Strategy: feed a continuous 60 Hz sine (amplitude 0.5, safely below
        the ±1.5 hard-clip) long enough for a Q=30 notch to fully settle.
        A Q=30 notch at 60 Hz has a time-constant of Q/(pi*f0) ≈ 0.16 s.
        We feed 90 chunks × 256 frames at 44100 Hz ≈ 520 ms >> 0.37 s needed
        to reach <10% of original amplitude.
        """
        n_ch, listen = 1, 0

        # Start with bypass — 60 Hz passes through
        bridge, written = _bridge(n_ch, FilterSettings(), listen_channel_idx=listen)
        bridge.on_chunk(_sine_chunk(60.0, N_FRAMES, offset=0))
        amplitude_before = float(np.abs(written[0]).max())

        # Switch to notch-60Hz filter
        notch_settings = FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True, notch_freq_hz=60.0, notch_q=30.0
            )
        )
        bridge.update_filter_settings(notch_settings)

        # Feed continuous chunks (advancing time) so the filter state can evolve
        n_settle = 90
        for i in range(n_settle):
            bridge.on_chunk(_sine_chunk(60.0, N_FRAMES, offset=(i + 1) * N_FRAMES))

        amplitude_after = float(np.abs(written[-1]).max())

        assert amplitude_before > 0.4, "Bypass should pass 60 Hz at ~0.5 amplitude"
        assert amplitude_after < amplitude_before * 0.1, (
            f"After {n_settle} chunks of notch settling: amplitude {amplitude_after:.4f} "
            f"is still too large (was {amplitude_before:.4f})"
        )

    def test_update_channel_names_propagates(self) -> None:
        """update_channel_names() must update the names used for override resolution."""
        n_ch = 2
        old_names = ["old_a", "old_b"]
        new_names = ["new_a", "new_b"]

        # Override keyed on new name
        override = ChannelFilterSettings(lowpass_hz=200.0, lowpass_order=4)
        settings = FilterSettings(
            default=ChannelFilterSettings(),
            overrides={"new_b": override},
        )

        bridge, _ = _bridge(n_ch, settings, listen_channel_idx=1,
                             channel_names=old_names)
        bridge.update_channel_names(new_names)

        # Verify the internal list was updated
        with bridge._lock:
            assert bridge._channel_names == new_names, (
                f"update_channel_names did not update: {bridge._channel_names}"
            )

    def test_update_filter_settings_resets_state(self) -> None:
        """Switching to bypass via update_filter_settings must clear filter state."""
        rng = np.random.default_rng(32)
        n_ch, listen = 1, 0
        # scale=0.5 keeps values safely below the ±1.5 hard-clip
        raw = _raw_chunk(n_ch, N_FRAMES, rng=rng, scale=0.5)

        # Start with notch and warm up state
        notch = FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True, notch_freq_hz=60.0, notch_q=30.0
            )
        )
        bridge, written = _bridge(n_ch, notch, listen_channel_idx=listen)
        bridge.on_chunk(raw)

        # Reset to bypass
        bridge.update_filter_settings(FilterSettings())
        bridge.on_chunk(raw)

        # With bypass and sub-clipping amplitude, output must equal the raw input
        np.testing.assert_allclose(
            written[-1], raw[:, listen],
            atol=1e-6,
            err_msg="After resetting to bypass, output does not match raw input",
        )


# ---------------------------------------------------------------------------
# Thread safety smoke test
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_set_gain_and_on_chunk(self) -> None:
        """Concurrent set_gain() and on_chunk() must not raise or corrupt output."""
        n_ch = 2
        bridge, written = _bridge(n_ch, listen_channel_idx=0)

        errors: list[Exception] = []

        def writer() -> None:
            r = np.random.default_rng(99)
            for _ in range(200):
                raw = _raw_chunk(n_ch, 64, rng=r)
                try:
                    bridge.on_chunk(raw)
                except Exception as exc:
                    errors.append(exc)

        def updater() -> None:
            for i in range(50):
                try:
                    bridge.set_gain(float(i % 10) / 10)
                    bridge.set_listen_channel_idx(i % n_ch)
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=updater)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"Concurrent access raised: {errors}"
