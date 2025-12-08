# daq/file_source.py
"""File-based playback device for replaying recorded WAV files as streaming data."""

from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path
from typing import List, Optional, Sequence, Any

import numpy as np

from PySide6 import QtCore, QtWidgets

from .base_device import (
    BaseDevice,
    DeviceInfo,
    ChannelInfo,
    Capabilities,
    ActualConfig,
)

logger = logging.getLogger(__name__)


class FileSource(BaseDevice):
    """
    File-based playback device for replaying WAV files.
    
    This device allows users to select a WAV file and stream its contents
    as if it were live data, enabling replay and analysis of recorded data.
    
    Features:
    - Supports 8-bit, 16-bit, and 24-bit PCM WAV files
    - Real-time playback at the file's native sample rate
    - Pause/resume and seek functionality for playback control
    - Channels labeled "Channel 1", "Channel 2", etc. based on file content
    """

    @classmethod
    def device_class_name(cls) -> str:
        return "File Source"

    def __init__(self, queue_maxsize: int = 64) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        
        # File state
        self._file_path: Optional[Path] = None
        self._wav_file: Optional[wave.Wave_read] = None
        self._n_channels: int = 0
        self._sample_rate: int = 0
        self._sample_width: int = 0  # Bytes per sample
        self._n_frames: int = 0
        self._current_frame: int = 0
        
        # Playback control state
        self._paused: bool = False
        self._seek_requested: Optional[int] = None  # Frame to seek to
        self._lock = threading.Lock()
        
        # Worker thread
        self._worker: Optional[threading.Thread] = None

    # ---- Discovery APIs ----

    @classmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        """Return a single virtual device representing file playback."""
        return [
            DeviceInfo(
                id="file",
                name="File Playback",
                details={"type": "virtual", "description": "Play back recorded WAV files"},
            )
        ]

    def get_capabilities(self, device_id: str) -> Capabilities:
        """
        Return capabilities based on the currently loaded file.
        
        If no file is loaded yet, returns empty capabilities.
        """
        if self._sample_rate > 0:
            return Capabilities(
                max_channels_in=self._n_channels,
                sample_rates=[self._sample_rate],
                dtype="float32",
                notes="WAV file playback",
            )
        else:
            # No file loaded yet - return minimal capabilities
            return Capabilities(
                max_channels_in=0,
                sample_rates=None,  # Will be determined after file selection
                dtype="float32",
                notes="Select a WAV file to determine capabilities",
            )

    def list_available_channels(self, device_id: str) -> List[ChannelInfo]:
        """Return channel list based on the loaded WAV file."""
        if self._n_channels <= 0:
            return []
        return [
            ChannelInfo(id=i, name=f"Channel {i + 1}", units="V")
            for i in range(self._n_channels)
        ]

    # ---- Lifecycle ----

    def _open_impl(self, device_id: str) -> None:
        """
        Open a file dialog to select a WAV file, then parse its header.
        
        This method shows a file dialog to the user and parses the selected
        WAV file to determine its properties (channels, sample rate, etc.).
        
        Raises:
            RuntimeError: If no file is selected or the file cannot be parsed.
        """
        # Show file dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select WAV File",
            "",
            "WAV Files (*.wav);;All Files (*)",
        )
        
        if not file_path:
            raise RuntimeError("No file selected")
        
        path = Path(file_path)
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
        
        try:
            wav = wave.open(str(path), "rb")
        except Exception as exc:
            raise RuntimeError(f"Failed to open WAV file: {exc}") from exc
        
        # Parse WAV header
        self._file_path = path
        self._wav_file = wav
        self._n_channels = wav.getnchannels()
        self._sample_rate = wav.getframerate()
        self._sample_width = wav.getsampwidth()
        self._n_frames = wav.getnframes()
        self._current_frame = 0
        
        logger.info(
            "Opened WAV file: %s (%d channels, %d Hz, %d-bit, %d frames)",
            path.name,
            self._n_channels,
            self._sample_rate,
            self._sample_width * 8,
            self._n_frames,
        )

    def _close_impl(self) -> None:
        """Close the WAV file and reset state."""
        if self._wav_file is not None:
            try:
                self._wav_file.close()
            except Exception as e:
                logger.debug("Failed to close WAV file: %s", e)
            self._wav_file = None
        
        self._file_path = None
        self._n_channels = 0
        self._sample_rate = 0
        self._sample_width = 0
        self._n_frames = 0
        self._current_frame = 0
        self._paused = False
        self._seek_requested = None

    def _configure_impl(
        self,
        sample_rate: int,
        channels: Sequence[int],
        chunk_size: int,
        **options: Any,
    ) -> ActualConfig:
        """Configure the device for streaming."""
        if self._wav_file is None:
            raise RuntimeError("No file loaded; call open() first")
        
        # The sample rate must match the file's native rate
        if sample_rate != self._sample_rate:
            logger.warning(
                "Requested sample rate %d differs from file rate %d; using file rate",
                sample_rate,
                self._sample_rate,
            )
        
        # Build channel list
        selected = [ch for ch in self._available_channels if ch.id in channels]
        if not selected:
            selected = list(self._available_channels)
        
        return ActualConfig(
            sample_rate=self._sample_rate,
            channels=selected,
            chunk_size=chunk_size,
            dtype="float32",
        )

    def _start_impl(self) -> None:
        """Start the playback worker thread."""
        if self._worker is not None and self._worker.is_alive():
            return
        
        if self._wav_file is None:
            raise RuntimeError("No file loaded")
        
        # Reset position to beginning
        self._wav_file.rewind()
        self._current_frame = 0
        self._paused = False
        self._seek_requested = None
        
        self._worker = threading.Thread(
            target=self._run_loop,
            name="FileSource-Worker",
            daemon=True,
        )
        self._worker.start()

    def _stop_impl(self) -> None:
        """Stop the playback worker thread."""
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._worker = None

    # ---- Playback Control Interface ----

    @property
    def is_paused(self) -> bool:
        """Return True if playback is paused."""
        return self._paused

    @property
    def current_position_seconds(self) -> float:
        """Return the current playback position in seconds."""
        if self._sample_rate <= 0:
            return 0.0
        return self._current_frame / self._sample_rate

    @property
    def total_duration_seconds(self) -> float:
        """Return the total file duration in seconds."""
        if self._sample_rate <= 0:
            return 0.0
        return self._n_frames / self._sample_rate

    @property
    def current_frame(self) -> int:
        """Return the current frame position."""
        return self._current_frame

    @property
    def total_frames(self) -> int:
        """Return the total number of frames."""
        return self._n_frames

    def set_paused(self, paused: bool) -> None:
        """Set the paused state."""
        with self._lock:
            self._paused = paused

    def seek_to_position(self, seconds: float) -> None:
        """Seek to a position in the file (in seconds)."""
        if self._sample_rate <= 0:
            return
        frame = int(seconds * self._sample_rate)
        frame = max(0, min(frame, self._n_frames))
        with self._lock:
            self._seek_requested = frame

    def seek_to_frame(self, frame: int) -> None:
        """Seek to a specific frame in the file."""
        frame = max(0, min(frame, self._n_frames))
        with self._lock:
            self._seek_requested = frame

    # ---- Worker Thread ----

    def _run_loop(self) -> None:
        """Main playback loop running in a worker thread."""
        if self.config is None or self._wav_file is None:
            return
        
        chunk_size = self.config.chunk_size
        sample_rate = self._sample_rate
        chunk_duration = chunk_size / sample_rate
        
        next_deadline = time.perf_counter() + chunk_duration
        
        while not self.stop_event.is_set():
            # Check for pause and seek requests
            with self._lock:
                paused = self._paused
                seek_frame = self._seek_requested
                self._seek_requested = None
            
            # Handle seek request FIRST (even when paused)
            if seek_frame is not None:
                try:
                    self._wav_file.setpos(seek_frame)
                    self._current_frame = seek_frame
                except Exception as exc:
                    logger.warning("Seek failed: %s", exc)
            
            # If paused, just wait and loop
            if paused:
                time.sleep(0.01)
                next_deadline = time.perf_counter() + chunk_duration
                continue

            
            # Read chunk from file
            try:
                raw_bytes = self._wav_file.readframes(chunk_size)
            except Exception as exc:
                logger.error("Failed to read from WAV file: %s", exc)
                break
            
            if not raw_bytes:
                # End of file - pause and wait for seek or stop
                logger.info("End of file reached, paused at end")
                with self._lock:
                    self._paused = True
                # Stay in loop, waiting for seek or stop
                continue
            
            # Decode to float32
            data = self._decode_frames(raw_bytes)
            if data is None or data.size == 0:
                # Empty decode - pause and wait like EOF
                with self._lock:
                    self._paused = True
                continue
            
            frames_read = data.shape[0]
            self._current_frame += frames_read
            
            # Emit through base class
            self.emit_array(data, mono_time=time.monotonic())
            
            # Pace to real-time
            next_deadline += chunk_duration
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -0.5:
                # We're falling behind; reset deadline
                next_deadline = time.perf_counter() + chunk_duration
        
        # Streaming complete - transition back to 'open' state
        # This is handled by the base class when stop_event is set


    def _decode_frames(self, raw_bytes: bytes) -> Optional[np.ndarray]:
        """
        Convert raw WAV bytes to float32 (frames, channels) array.
        
        Handles 8-bit unsigned, 16-bit signed, and 24-bit signed PCM.
        """
        if not raw_bytes:
            return None
        
        try:
            if self._sample_width == 1:  # 8-bit unsigned
                data = np.frombuffer(raw_bytes, dtype=np.uint8)
                data = (data.astype(np.float32) - 128.0) / 128.0
            elif self._sample_width == 2:  # 16-bit signed
                data = np.frombuffer(raw_bytes, dtype=np.int16)
                data = data.astype(np.float32) / 32768.0
            elif self._sample_width == 3:  # 24-bit signed
                # Unpack 24-bit samples (3 bytes each, little-endian)
                n_samples = len(raw_bytes) // 3
                raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                # Create 32-bit integers by padding with sign extension
                padded = np.zeros(n_samples, dtype=np.int32)
                for i in range(n_samples):
                    b0 = raw_arr[i * 3]
                    b1 = raw_arr[i * 3 + 1]
                    b2 = raw_arr[i * 3 + 2]
                    # Little-endian: b0 is LSB, b2 is MSB
                    val = b0 | (b1 << 8) | (b2 << 16)
                    # Sign extend if negative
                    if val & 0x800000:
                        val |= 0xFF000000
                    padded[i] = np.int32(val)
                data = padded.astype(np.float32) / 8388608.0  # 2^23
            elif self._sample_width == 4:  # 32-bit signed
                data = np.frombuffer(raw_bytes, dtype=np.int32)
                data = data.astype(np.float32) / 2147483648.0  # 2^31
            else:
                logger.error("Unsupported sample width: %d", self._sample_width)
                return None
            
            # Reshape to (frames, channels)
            frames = len(data) // self._n_channels
            if frames == 0:
                return None
            return data.reshape((frames, self._n_channels))
        
        except Exception as exc:
            logger.error("Failed to decode WAV frames: %s", exc)
            return None
