"""
core/monitor_audio_bridge.py — Low-latency monitor audio bridge.

Receives raw chunks directly from the SoundCardSource emitter thread,
applies filtering via a dedicated SignalConditioner instance, selects the
listen channel, applies gain, and writes mono float32 samples straight
into the AudioPlayer's software ring — bypassing the dispatcher, the
audio_queue, and the AudioManager router thread.

Typical latency saving vs. the old path:
  Dispatcher tick period (~16 ms) + AudioManager router hop (~0–10 ms)
  → removed from the monitor critical path.

Thread safety
-------------
``on_chunk()`` is called from the SoundCardSource emitter thread.
All ``set_*`` / ``update_*`` methods may be called from any thread
(GUI/control thread).  Mutable state is protected by ``_lock``.
"""

from __future__ import annotations

import threading
import numpy as np
from typing import Optional, TYPE_CHECKING

from .conditioning import FilterSettings, SignalConditioner

if TYPE_CHECKING:  # pragma: no cover
    from audio.player import AudioPlayer


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

class _FakeChunk:
    """Minimal chunk-like object accepted by ``SignalConditioner.process()``."""
    __slots__ = ("samples", "dt", "channel_names")

    def __init__(
        self,
        samples: np.ndarray,
        dt: float,
        channel_names: list[str],
    ) -> None:
        self.samples = samples
        self.dt = dt
        self.channel_names = channel_names


# ---------------------------------------------------------------------------
# MonitorAudioBridge
# ---------------------------------------------------------------------------

class MonitorAudioBridge:
    """
    Low-latency audio monitor bridge.

    Life-cycle
    ----------
    1. Create with a reference to the active ``AudioPlayer``.
    2. Register with ``SoundCardSource`` via ``source.register_monitor_bridge(bridge)``.
    3. ``on_chunk()`` is called automatically for each captured chunk.
    4. Deregister with ``source.register_monitor_bridge(None)`` before destroying.
    """

    def __init__(
        self,
        player: "AudioPlayer",
        filter_settings: FilterSettings,
        sample_rate: float,
        n_channels: int,
        channel_names: list[str],
        listen_channel_idx: Optional[int],
        gain: float,
    ) -> None:
        self._player = player
        self._sample_rate = float(sample_rate)
        self._gain = float(gain)
        self._lock = threading.Lock()
        self._listen_idx: Optional[int] = listen_channel_idx
        self._channel_names: list[str] = list(channel_names)

        # Dedicated conditioner instance — independent from dispatcher state.
        self._conditioner = SignalConditioner(filter_settings)

    # ------------------------------------------------------------------
    # Hot path — called from emitter thread
    # ------------------------------------------------------------------

    def on_chunk(self, raw_chunk: np.ndarray) -> None:
        """Process one captured chunk on the emitter thread.

        Args:
            raw_chunk: float32 array shaped ``(n_frames, n_active_channels)``,
                       already restricted to the active channel selection.
        """
        with self._lock:
            listen_idx = self._listen_idx
            gain = self._gain
            channel_names = self._channel_names

        if listen_idx is None:
            return

        # Transpose to (n_channels, n_frames) expected by SignalConditioner.
        samples = np.ascontiguousarray(raw_chunk.T, dtype=np.float32)
        n_ch = samples.shape[0]

        if listen_idx >= n_ch:
            return  # Channel index out of range for this chunk — skip silently.

        fake_chunk = _FakeChunk(samples, 1.0 / self._sample_rate, channel_names[:n_ch])
        filtered = self._conditioner.process(fake_chunk)  # (n_channels, n_frames)

        # Extract the chosen channel, apply gain, hard-clip, write to ring.
        mono = filtered[listen_idx] * gain
        np.clip(mono, -1.5, 1.5, out=mono)
        self._player._ring_write(mono)

    # ------------------------------------------------------------------
    # Control interface — called from GUI / control thread
    # ------------------------------------------------------------------

    def set_listen_channel_idx(self, idx: Optional[int]) -> None:
        """Set which channel index within active channels to monitor."""
        with self._lock:
            self._listen_idx = None if idx is None else int(idx)

    def set_gain(self, gain: float) -> None:
        """Set playback gain (0.0 – 1.0 typical range, hard-clipped at ±1.5)."""
        with self._lock:
            self._gain = max(0.0, float(gain))

    def update_filter_settings(self, settings: FilterSettings) -> None:
        """Apply new filter settings to the monitor conditioner.

        The conditioner rebuilds its filter chain on the next chunk so that
        both the monitor path and the dispatcher path stay in sync.
        """
        with self._lock:
            self._conditioner.update_settings(settings)

    def update_channel_names(self, names: list[str]) -> None:
        """Sync channel name list when the active channel set changes."""
        with self._lock:
            self._channel_names = list(names)
