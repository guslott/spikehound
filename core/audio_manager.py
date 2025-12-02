"""AudioManager - Centralized audio routing and playback management.

Extracted from MainWindow to enable headless audio monitoring support.
Manages the AudioPlayer lifecycle and routes audio from dispatcher to player.
"""

from __future__ import annotations

import queue
import threading
import logging
from typing import TYPE_CHECKING, Optional

from shared.models import ChunkPointer, EndOfStream
from audio.player import AudioPlayer, AudioConfig

if TYPE_CHECKING:  # pragma: no cover
    from core.runtime import Runtime
    from shared.ring_buffer import SharedRingBuffer

logger = logging.getLogger(__name__)


class AudioManager:
    """Manages audio routing and playback for channel monitoring."""

    def __init__(self, runtime: Runtime) -> None:
        """Initialize audio manager with runtime reference.
        
        Args:
            runtime: Runtime instance providing access to audio_queue and controller
        """
        self.runtime = runtime
        
        # Audio routing state
        self._listen_channel_id: Optional[int] = None
        self._channel_ids_current: list[int] = []
        
        # AudioPlayer state
        self._audio_player: Optional[AudioPlayer] = None
        self._audio_player_queue: Optional[queue.Queue] = None
        self._audio_input_samplerate: float = 0.0
        self._audio_current_device: Optional[object] = None
        self._audio_player_buffer: Optional[SharedRingBuffer] = None
        self._audio_gain: float = 0.7
        
        # Thread control
        self._audio_router_thread: Optional[threading.Thread] = None
        self._audio_router_stop = threading.Event()
        self._audio_lock = threading.Lock()
        
        # Running state
        self._running = False

    def start(self) -> None:
        """Start the audio routing thread."""
        if self._running:
            return
        
        self._audio_router_stop.clear()
        self._audio_router_thread = threading.Thread(
            target=self._audio_router_loop,
            name="AudioRouter",
            daemon=True
        )
        self._audio_router_thread.start()
        self._running = True
        logger.info("AudioManager started")

    def stop(self) -> None:
        """Stop the audio routing thread and cleanup."""
        if not self._running:
            return
        
        self._audio_router_stop.set()
        
        # Stop router thread
        if self._audio_router_thread is not None:
            self._audio_router_thread.join(timeout=2.0)
            self._audio_router_thread = None
        
        # Stop audio player
        self._stop_audio_player()
        
        self._running = False
        logger.info("AudioManager stopped")

    def set_listen_channel(self, channel_id: Optional[int]) -> None:
        """Set which channel to monitor for audio playback.
        
        Args:
            channel_id: Channel ID to monitor, or None to stop monitoring
        """
        with self._audio_lock:
            self._listen_channel_id = channel_id
        
        if channel_id is None:
            self._stop_audio_player()
            logger.debug("Audio monitoring stopped")
        else:
            logger.debug(f"Audio monitoring channel {channel_id}")

    def set_output_device(self, device_id: Optional[object]) -> None:
        """Set audio output device.
        
        Args:
            device_id: Device ID for audio output (int or object), or None for default
        """
        with self._audio_lock:
            if self._audio_current_device != device_id:
                self._audio_current_device = device_id
                # Force player recreation on next ensure
                if self._audio_player is not None:
                    self._stop_audio_player()

    def set_gain(self, gain: float) -> None:
        """Set audio output gain.
        
        Args:
            gain: Gain value (0.0 to 1.0)
        """
        self._audio_gain = max(0.0, min(1.0, float(gain)))

    def update_active_channels(self, channel_ids: list[int]) -> None:
        """Update the list of active channels.
        
        Args:
            channel_ids: List of currently active channel IDs
        """
        self._channel_ids_current = list(channel_ids)

    def is_monitoring(self) -> bool:
        """Check if currently monitoring a channel.
        
        Returns:
            True if monitoring is active
        """
        with self._audio_lock:
            return self._listen_channel_id is not None

    # Internal implementation

    def _audio_router_loop(self) -> None:
        """Audio routing thread main loop.
        
        Pulls audio chunks from runtime.audio_queue and routes them to AudioPlayer
        based on selected listen channel.
        """
        while not self._audio_router_stop.is_set():
            try:
                controller = self.runtime.controller
                if controller is None:
                    self._audio_router_stop.wait(0.1)
                    continue
                
                # Get audio queue from runtime
                try:
                    aq = getattr(self.runtime, "audio_queue", None)
                    if aq is None:
                        self._audio_router_stop.wait(0.1)
                        continue
                    item = aq.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Skip end-of-stream markers
                if item is EndOfStream:
                    continue
                
                # Check if we have a listen channel selected
                with self._audio_lock:
                    listen_id = self._listen_channel_id
                
                if listen_id is None or listen_id not in self._channel_ids_current:
                    continue
                
                # Get viz buffer for chunk pointer resolution
                viz_buffer = controller.viz_buffer() if controller else None
                if isinstance(item, ChunkPointer) and viz_buffer is None:
                    continue
                
                # Ensure audio player is running
                if not self._ensure_audio_player():
                    continue
                
                # Find channel index
                try:
                    idx = self._channel_ids_current.index(listen_id)
                except ValueError:
                    continue
                
                # Route to audio player
                with self._audio_lock:
                    player = self._audio_player
                    player_queue = self._audio_player_queue
                
                if player is not None and player_queue is not None:
                    player.set_selected_channel(idx)
                    try:
                        player_queue.put_nowait(item)
                    except queue.Full:
                        pass  # Drop if queue full
            
            except Exception as exc:
                logger.error(f"Error in audio router loop: {exc}", exc_info=True)
                continue

    def _ensure_audio_player(self) -> bool:
        """Create or reconfigure AudioPlayer to match current state.
        
        Returns:
            True if player is ready, False otherwise
        """
        controller = self.runtime.controller
        if controller is None:
            return False
        
        sample_rate = controller.sample_rate
        if sample_rate <= 0:
            return False
        
        device_id = self._audio_current_device
        viz_buffer = controller.viz_buffer()
        
        with self._audio_lock:
            # Check if current player matches requirements
            if (
                self._audio_player is not None
                and abs(self._audio_input_samplerate - sample_rate) < 1e-6
                and self._audio_current_device == device_id
                and self._audio_player_buffer is viz_buffer
            ):
                return True
            
            # Need to recreate player
            player_to_stop = self._audio_player
            self._audio_player = None
            self._audio_player_queue = None
            self._audio_input_samplerate = 0.0
            self._audio_player_buffer = None
        
        # Stop old player outside lock
        if player_to_stop is not None:
            try:
                player_to_stop.stop()
                player_to_stop.join(timeout=1.0)
            except Exception:
                pass
        
        # Create new player
        queue_obj: queue.Queue = queue.Queue(maxsize=4)
        config = AudioConfig(
            out_samplerate=44_100,
            out_channels=1,
            device=device_id,
            gain=self._audio_gain,
            blocksize=128,
            ring_seconds=0.1,
        )
        
        try:
            player = AudioPlayer(
                audio_queue=queue_obj,
                input_samplerate=int(sample_rate),
                config=config,
                selected_channel=0,
                ring_buffer=viz_buffer,
            )
        except Exception as exc:
            logger.error(f"Failed to create AudioPlayer: {exc}")
            return False
        
        # Update state
        with self._audio_lock:
            self._audio_player = player
            self._audio_player_queue = queue_obj
            self._audio_input_samplerate = float(sample_rate)
            self._audio_player_buffer = viz_buffer
        
        # Start player
        player.start()
        logger.info(f"AudioPlayer started: {sample_rate}Hz -> 44.1kHz")
        return True

    def _stop_audio_player(self) -> None:
        """Stop and cleanup the audio player."""
        with self._audio_lock:
            player = self._audio_player
            self._audio_player = None
            self._audio_player_queue = None
            self._audio_input_samplerate = 0.0
            self._audio_player_buffer = None
        
        if player is not None:
            try:
                player.stop()
                player.join(timeout=1.0)
                logger.debug("AudioPlayer stopped")
            except Exception as exc:
                logger.warning(f"Error stopping AudioPlayer: {exc}")
