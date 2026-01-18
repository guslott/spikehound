# daq/file_source.py
"""File-based playback device for replaying recorded WAV files as streaming data."""

from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path
from typing import List, Optional, Sequence, Any
import scipy.io.wavfile as wavfile

import numpy as np

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
        self._raw_data: Optional[np.ndarray] = None  # Mapped/Loaded raw audio data
        self._n_channels: int = 0
        self._sample_rate: int = 0
        self._dtype: np.dtype = np.float32
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
        Open a file dialog to select a WAV file, then read it via scipy.io.wavfile.
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
            # Use mmap=True to avoid loading large files entirely into RAM
            # valid only for real files, not file-like objects
            sr, data = wavfile.read(str(path), mmap=True)
        except Exception as exc:
            error_msg = str(exc)
            # Provide a more user-friendly error message for common corruption cases
            if "cannot access local variable" in error_msg or "'fs'" in error_msg:
                raise RuntimeError(
                    f"Failed to open WAV file: The file appears to be corrupted or incomplete. "
                    f"This can happen if recording was interrupted before the file was properly closed."
                ) from exc
            raise RuntimeError(f"Failed to open WAV file: {exc}") from exc
        
        # Determine shape
        if data.ndim == 1:
            n_frames = data.shape[0]
            n_channels = 1
            # Reshape to (frames, 1) for consistency
            # Note: cannot reshape mmap array easily if it's 1D? 
            # Actually we'll handle 1D slicing in loop
            self._is_mono = True
        else:
            n_frames, n_channels = data.shape
            self._is_mono = False

        self._file_path = path
        self._raw_data = data
        self._sample_rate = int(sr)
        self._n_channels = n_channels
        self._n_frames = n_frames
        self._current_frame = 0
        self._dtype = data.dtype
        
        logger.info(
            "Opened WAV file: %s (%d channels, %d Hz, %s, %d frames)",
            path.name,
            self._n_channels,
            self._sample_rate,
            self._dtype,
            self._n_frames,
        )

    def _close_impl(self) -> None:
        """Close the file (release mmap handle) and reset state."""
        # For mmap, just releasing the object is usually enough
        if self._raw_data is not None:
             # If it was a generic mmap, could confirm close?
             # numpy memmap has ._mmap.close() but plain ndarray doesn't
             self._raw_data = None
        
        self._file_path = None
        self._n_channels = 0
        self._sample_rate = 0
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
        if self._raw_data is None:
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
        
        if self._raw_data is None:
            raise RuntimeError("No file loaded")
        
        # Reset position to beginning
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
        if self.config is None or self._raw_data is None:
            return
        
        chunk_size = self.config.chunk_size
        sample_rate = self._sample_rate
        chunk_duration = chunk_size / sample_rate
        
        next_deadline = time.perf_counter() + chunk_duration
        
        total_frames = self._n_frames
        
        while not self.stop_event.is_set():
            # Check for pause and seek requests
            with self._lock:
                paused = self._paused
                seek_frame = self._seek_requested
                self._seek_requested = None
            
            # Handle seek request FIRST
            if seek_frame is not None:
                self._current_frame = max(0, min(seek_frame, total_frames))
            
            # If paused, just wait and loop
            if paused:
                time.sleep(0.01)
                next_deadline = time.perf_counter() + chunk_duration
                continue
            
            # Check if EOS
            if self._current_frame >= total_frames:
                # End of file - pause
                logger.info("End of file reached, paused at end")
                with self._lock:
                    self._paused = True
                continue
            
            # Read chunk from memory/mmap
            start = self._current_frame
            end = min(start + chunk_size, total_frames)
            
            # Slice the raw data
            # Handle 1D (mono) vs 2D (stereo/multi)
            if self._raw_data.ndim == 1:
                raw_chunk = self._raw_data[start:end]
                # Reshape to (frames, 1)
                raw_chunk = raw_chunk[:, np.newaxis]
            else:
                raw_chunk = self._raw_data[start:end, :]

            # Convert to float32 and normalize
            data = self._normalize_chunk(raw_chunk)

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

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Normalize audio chunk to float32 [-1, 1]."""
        # Note: input is (frames, channels)
        
        if chunk.dtype == np.float32 or chunk.dtype == np.float64:
            return chunk.astype(np.float32)
        elif chunk.dtype == np.int16:
            return chunk.astype(np.float32) / 32768.0
        elif chunk.dtype == np.int32:
            return chunk.astype(np.float32) / 2147483648.0
        elif chunk.dtype == np.uint8:
            return (chunk.astype(np.float32) - 128.0) / 128.0
        else:
            # Unknown type, just cast (or error?)
            # Scipy might return other types?
            return chunk.astype(np.float32)
