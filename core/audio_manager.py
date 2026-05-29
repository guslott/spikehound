"""AudioManager - Centralized monitor-audio playback management.

Owns the AudioPlayer + MonitorAudioBridge lifecycle for channel monitoring
("hear the neuron").  The bridge writes filtered mono samples straight into
the player's ring from the source's capture/emitter thread, so there is no
router thread and no audio fan-out queue — the bridge is the only path.
"""

from __future__ import annotations

import threading
import logging
from typing import TYPE_CHECKING, Optional

from audio.player import AudioPlayer, AudioConfig

if TYPE_CHECKING:  # pragma: no cover
    from core.runtime import Runtime
    from core.conditioning import FilterSettings

logger = logging.getLogger(__name__)


class AudioManager:
    """Manages monitor-audio playback for channel listening."""

    def __init__(self, runtime: Runtime) -> None:
        """Initialize audio manager with runtime reference.

        Args:
            runtime: Runtime instance providing access to the controller
        """
        self.runtime = runtime

        # Listen state
        self._listen_channel_id: Optional[int] = None
        self._channel_ids_current: list[int] = []

        # AudioPlayer state
        self._audio_player: Optional[AudioPlayer] = None
        self._audio_input_samplerate: float = 0.0
        self._audio_current_device: Optional[object] = None
        self._audio_gain: float = 0.7

        # MonitorAudioBridge — writes filtered mono samples straight into the
        # player ring from the source's capture/emitter thread.  This is the
        # one and only monitor audio path.
        self._monitor_bridge: Optional[object] = None

        self._audio_lock = threading.Lock()

        # Running state
        self._running = False

    def start(self) -> None:
        """Mark the audio manager active.

        The player + bridge are created lazily when a listen channel is
        selected (see :meth:`set_listen_channel`).
        """
        if self._running:
            return
        self._running = True
        logger.info("AudioManager started")

    def stop(self) -> None:
        """Tear down any active player/bridge."""
        if not self._running:
            return
        self._stop_audio_player()
        self._running = False
        logger.info("AudioManager stopped")

    def set_listen_channel(self, channel_id: Optional[int]) -> None:
        """Set which channel to monitor for audio playback.

        Passing ``None`` stops monitoring and releases the playback device.
        Otherwise the player + bridge are created (if needed) for the current
        source sample rate and output device.

        Args:
            channel_id: Channel ID to monitor, or None to stop monitoring
        """
        with self._audio_lock:
            self._listen_channel_id = channel_id
            channel_ids = self._channel_ids_current

        if channel_id is None:
            self._stop_audio_player()
            logger.debug("Audio monitoring stopped")
            return

        # Ensure the player + bridge exist for the current source/rate/device.
        if not self._ensure_audio_player():
            logger.debug(
                "Audio monitoring requested but player not ready "
                "(no active stream yet?)"
            )
            return

        # Push the selected channel index to the (possibly pre-existing) bridge.
        with self._audio_lock:
            bridge = self._monitor_bridge
        if bridge is not None:
            try:
                idx = channel_ids.index(channel_id)
            except ValueError:
                idx = None
            bridge.set_listen_channel_idx(idx)
        logger.debug("Audio monitoring channel %s", channel_id)

    def set_output_device(self, device_id: Optional[object]) -> None:
        """Set audio output device.

        Recreates the player on the new device if monitoring is active.

        Args:
            device_id: Device ID for audio output (int or object), or None for default
        """
        with self._audio_lock:
            changed = self._audio_current_device != device_id
            self._audio_current_device = device_id
            listen_id = self._listen_channel_id
            have_player = self._audio_player is not None

        if changed and have_player:
            self._stop_audio_player()
            if listen_id is not None:
                # Recreate the player/bridge on the new output device.
                self.set_listen_channel(listen_id)

    def set_gain(self, gain: float) -> None:
        """Set audio output gain.

        Also forwards the new gain to the live MonitorAudioBridge so the
        bridge stays in sync whenever the GUI slider moves.

        Args:
            gain: Gain value (0.0 to 1.0)
        """
        gain = max(0.0, min(1.0, float(gain)))
        self._audio_gain = gain
        with self._audio_lock:
            bridge = self._monitor_bridge
        if bridge is not None:
            bridge.set_gain(gain)

    def update_filter_settings(self, settings: "FilterSettings") -> None:  # noqa: F821
        """Forward updated filter settings to the live MonitorAudioBridge.

        Called by PipelineController.update_filter_settings() whenever the GUI
        pushes a new FilterSettings object so the bridge conditioner stays in
        parity with the dispatcher conditioner.

        Args:
            settings: New FilterSettings to apply.
        """
        with self._audio_lock:
            bridge = self._monitor_bridge
        if bridge is not None:
            bridge.update_filter_settings(settings)

    def update_active_channels(self, channel_ids: list[int]) -> None:
        """Update the list of active channels (used to resolve listen index).

        Args:
            channel_ids: List of currently active channel IDs
        """
        with self._audio_lock:
            self._channel_ids_current = list(channel_ids)

    def is_monitoring(self) -> bool:
        """Check if currently monitoring a channel.

        Returns:
            True if monitoring is active
        """
        with self._audio_lock:
            return self._listen_channel_id is not None

    # Hardware capture-device buffer latency (ms).
    # Set in daq/soundcard_source.py (buffersize_msec=10).  This is the only
    # upstream contribution that cannot be measured in software because it
    # represents the time audio spends inside the hardware before the first
    # callback fires.  Everything downstream of the callback is now measured
    # directly by MonitorAudioBridge.chunk_latency_stats_ms().
    _CAPTURE_DEVICE_MS: float = 10.0

    def monitor_latency_ms(self) -> Optional[float]:
        """Return the measured mean end-to-end monitor latency in milliseconds.

        Components:
          - Capture device hardware buffer (known constant: 10 ms).
          - Input batching: the source emitter accumulates a full chunk before
            delivering it, so a sample waits ~half a chunk duration to be
            assembled.  Measured from the observed chunk size via
            ``MonitorAudioBridge.input_batching_ms()``.
          - Bridge processing time: measured rolling mean from
            ``MonitorAudioBridge.chunk_latency_stats_ms()``.  Falls back to
            0.1 ms (typical filter + ring-write overhead) until the first
            chunk has been processed.
          - Player ring fill + playback device buffer: from
            ``AudioPlayer.estimated_latency_ms()``.

        Returns ``None`` when monitoring is not active or the player is not running.
        """
        with self._audio_lock:
            if self._listen_channel_id is None:
                return None
            player = self._audio_player
            bridge = self._monitor_bridge
        if player is None:
            return None
        stats = bridge.chunk_latency_stats_ms() if bridge is not None else None
        bridge_ms = stats[0] if stats is not None else 0.1
        batching_ms = (bridge.input_batching_ms() if bridge is not None else None) or 0.0
        return (
            self._CAPTURE_DEVICE_MS
            + batching_ms
            + bridge_ms
            + player.estimated_latency_ms()
        )

    def monitor_latency_p95_ms(self) -> Optional[float]:
        """Return the measured p95 end-to-end monitor latency in milliseconds.

        Uses the p95 bridge processing time instead of the mean, giving a
        conservative worst-case-ish estimate suitable for jitter analysis.
        Returns ``None`` when monitoring is not active or no data yet.
        """
        with self._audio_lock:
            if self._listen_channel_id is None:
                return None
            player = self._audio_player
            bridge = self._monitor_bridge
        if player is None:
            return None
        stats = bridge.chunk_latency_stats_ms() if bridge is not None else None
        if stats is None:
            return None
        bridge_p95_ms = stats[1]
        batching_ms = bridge.input_batching_ms() or 0.0
        return (
            self._CAPTURE_DEVICE_MS
            + batching_ms
            + bridge_p95_ms
            + player.estimated_latency_ms()
        )

    # Internal implementation

    def _ensure_audio_player(self) -> bool:
        """Create or reconfigure the AudioPlayer + MonitorAudioBridge to match
        the current source sample rate and output device.

        Returns:
            True if a running player is available, False otherwise.
        """
        controller = self.runtime.controller
        if controller is None:
            return False

        sample_rate = controller.sample_rate
        if not sample_rate or sample_rate <= 0:
            return False

        device_id = self._audio_current_device

        with self._audio_lock:
            # Reuse the existing player when sample rate and device are unchanged.
            if (
                self._audio_player is not None
                and abs(self._audio_input_samplerate - sample_rate) < 1e-6
                and self._audio_current_device == device_id
            ):
                return True

            player_to_stop = self._audio_player
            self._audio_player = None
            self._audio_input_samplerate = 0.0
            self._monitor_bridge = None

        # Tear down any previous player/bridge outside the lock.
        if player_to_stop is not None:
            try:
                controller.set_monitor_bridge(None)
            except Exception as exc:
                logger.warning("Failed to deregister previous monitor bridge: %s", exc)
            try:
                player_to_stop.stop()
                player_to_stop.join(timeout=1.0)
            except Exception as exc:
                logger.warning("Failed to stop previous AudioPlayer cleanly: %s", exc)

        # Size the player ring from the actual source chunk size so one full
        # chunk always fits; otherwise _ring_write clamps oversized writes and
        # silently drops the head of every chunk.
        frames_per_write = int(controller.chunk_size or 0)

        # Opt-in low-latency monitor mode shrinks the playback device buffer
        # (10 -> 5 ms).  Read live so it applies on the next listen.
        low_latency = False
        try:
            low_latency = bool(self.runtime.app_settings.monitor_low_latency)
        except Exception:
            low_latency = False
        playback_buffer_msec = 5 if low_latency else 10

        config = AudioConfig(
            out_channels=1,
            device=device_id,
            blocksize=64,
            # The ring only needs to absorb one source chunk + scheduler jitter
            # between the MonitorAudioBridge write and the playback callback read.
            ring_seconds=0.015,
            frames_per_write=frames_per_write,
            playback_buffer_msec=playback_buffer_msec,
        )

        try:
            player = AudioPlayer(input_samplerate=int(sample_rate), config=config)
        except Exception as exc:
            logger.error("Failed to create AudioPlayer: %s", exc)
            return False

        with self._audio_lock:
            self._audio_player = player
            self._audio_input_samplerate = float(sample_rate)
            listen_id = self._listen_channel_id
            channel_ids = self._channel_ids_current
            gain = self._audio_gain

        player.start()
        logger.info("AudioPlayer started: %sHz", sample_rate)

        # Create the low-latency monitor bridge and register it with the source.
        if listen_id is not None:
            try:
                from .monitor_audio_bridge import MonitorAudioBridge

                # Seed the bridge with the current filter state so it is in
                # parity with the dispatcher conditioner from the first chunk.
                current_filter_settings = controller.filter_settings

                # Use real channel names so that per-channel filter overrides
                # (keyed by name) resolve correctly inside the bridge conditioner.
                channel_infos = controller.active_channels()
                id_to_name = {info.id: info.name for info in channel_infos}
                channel_names = [id_to_name.get(cid, str(cid)) for cid in channel_ids]

                listen_idx: Optional[int] = None
                try:
                    listen_idx = channel_ids.index(listen_id)
                except ValueError:
                    pass

                bridge = MonitorAudioBridge(
                    player=player,
                    filter_settings=current_filter_settings,
                    sample_rate=sample_rate,
                    n_channels=len(channel_ids),
                    channel_names=channel_names,
                    listen_channel_idx=listen_idx,
                    gain=gain,
                )
                controller.set_monitor_bridge(bridge)
                with self._audio_lock:
                    self._monitor_bridge = bridge
                logger.info("MonitorAudioBridge registered — low-latency monitor path active")
            except Exception as exc:
                logger.warning("Failed to create MonitorAudioBridge: %s", exc)

        return True

    def _stop_audio_player(self) -> None:
        """Stop and clean up the audio player and deregister the bridge."""
        with self._audio_lock:
            player = self._audio_player
            self._audio_player = None
            self._audio_input_samplerate = 0.0
            self._monitor_bridge = None

        # Deregister the bridge from the source before stopping the player so
        # the emitter thread cannot call on_chunk() after the ring is gone.
        controller = self.runtime.controller
        if controller is not None:
            try:
                controller.set_monitor_bridge(None)
            except Exception as exc:
                logger.warning("Failed to deregister monitor bridge: %s", exc)

        if player is not None:
            try:
                player.stop()
                player.join(timeout=1.0)
                logger.debug("AudioPlayer stopped")
            except Exception as exc:
                logger.warning("Error stopping AudioPlayer: %s", exc)
