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

Latency measurement
-------------------
``on_chunk()`` timestamps itself from entry to the completion of
``_ring_write()``.  These durations are kept in a 64-slot rolling deque
and exposed via ``chunk_latency_stats_ms()``.  The AudioManager combines
this measured bridge time with the known capture-device hardware buffer
and the live player-ring fill level to produce a fully measured end-to-end
latency estimate (no more fixed upstream constant).

Thread safety
-------------
``on_chunk()`` is called from the SoundCardSource emitter thread.
All ``set_*`` / ``update_*`` methods may be called from any thread
(GUI/control thread).  Mutable state is protected by ``_lock``.
The latency deque has its own ``_latency_lock`` to avoid coupling
the hot-path lock to the health-snapshot reader thread.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from .conditioning import FilterSettings, SignalConditioner

if TYPE_CHECKING:  # pragma: no cover
    from audio.player import AudioPlayer

# Number of recent chunk timings retained for rolling statistics.
_LATENCY_WINDOW = 64


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

        # Rolling latency measurement: time from on_chunk() entry to
        # _ring_write() completion, in milliseconds.
        self._latency_deque: deque[float] = deque(maxlen=_LATENCY_WINDOW)
        self._latency_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Hot path — called from emitter thread
    # ------------------------------------------------------------------

    def on_chunk(self, raw_chunk: np.ndarray) -> None:
        """Process one captured chunk on the emitter thread.

        Args:
            raw_chunk: float32 array shaped ``(n_frames, n_active_channels)``,
                       already restricted to the active channel selection.
        """
        # Timestamp at entry — before any other work — so we capture the full
        # bridge contribution including lock acquisition and filter computation.
        t_start = time.perf_counter()

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

        # Record how long this chunk took from entry to ring-write completion.
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        with self._latency_lock:
            self._latency_deque.append(elapsed_ms)

    # ------------------------------------------------------------------
    # Latency statistics — called from health-snapshot / GUI thread
    # ------------------------------------------------------------------

    def chunk_latency_stats_ms(self) -> Optional[Tuple[float, float]]:
        """Return rolling ``(mean_ms, p95_ms)`` of bridge processing time.

        This measures the time from ``on_chunk()`` entry to ``_ring_write()``
        completion — i.e. the bridge's contribution to end-to-end latency.
        Returns ``None`` until at least one chunk has been processed.
        """
        with self._latency_lock:
            if not self._latency_deque:
                return None
            samples = list(self._latency_deque)

        arr = np.array(samples, dtype=np.float64)
        mean_ms = float(arr.mean())
        p95_ms = float(np.percentile(arr, 95))
        return mean_ms, p95_ms

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
