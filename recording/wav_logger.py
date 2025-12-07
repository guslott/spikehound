"""WAV file logger thread for recording streamed audio data.

Adapted from student implementation in feature/logging branch.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import wave
from typing import Optional, Union

import numpy as np

from shared.models import Chunk, EndOfStream

logger = logging.getLogger(__name__)


class WavLoggerThread:
    """
    Consumes Chunk objects from a queue and writes to a WAV file as 16-bit PCM.
    
    The thread handles:
    - Extracting samples from Chunk objects
    - Transposing (channels, samples) -> (samples, channels) for interleaved WAV
    - Converting float32 [-1, 1] to int16 PCM
    - Thread-safe start/stop lifecycle
    - Clean shutdown on EndOfStream sentinel
    """

    def __init__(
        self,
        data_queue: "queue.Queue[Union[Chunk, type[EndOfStream]]]",
        out_path: str,
        sample_rate: int,
        channels: int,
    ) -> None:
        """
        Initialize the WAV logger.
        
        Args:
            data_queue: Queue to consume Chunk objects from (typically logging_queue)
            out_path: Output file path for the WAV file
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
        """
        self._queue = data_queue
        self._out_path = out_path
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._wf: Optional[wave.Wave_write] = None
        self._frames_written: int = 0

    @property
    def frames_written(self) -> int:
        """Total number of audio frames written to the file."""
        return self._frames_written

    @property
    def duration_seconds(self) -> float:
        """Duration of recorded audio in seconds."""
        if self._sample_rate <= 0:
            return 0.0
        return self._frames_written / self._sample_rate

    def start(self) -> None:
        """Start the logger thread and open the WAV file for writing."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("WavLoggerThread already running")
            return

        # Ensure output directory exists
        out_dir = os.path.dirname(self._out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Open WAV file
        try:
            wf = wave.open(self._out_path, "wb")
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(self._sample_rate)
            self._wf = wf
        except Exception as exc:
            logger.error("Failed to open WAV file %s: %s", self._out_path, exc)
            raise

        self._stop_event.clear()
        self._frames_written = 0
        self._thread = threading.Thread(
            target=self._run,
            name="WavLoggerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "WavLoggerThread started: %s (sr=%d, ch=%d)",
            self._out_path,
            self._sample_rate,
            self._channels,
        )

    def stop(self, join_timeout: float = 2.0) -> None:
        """
        Stop the logger thread and finalize the WAV file.
        
        Args:
            join_timeout: Maximum time to wait for thread to finish
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logger.warning("WavLoggerThread did not stop within timeout")
            self._thread = None

        if self._wf is not None:
            try:
                self._wf.close()
            except Exception as exc:
                logger.warning("Error closing WAV file: %s", exc)
            self._wf = None

        logger.info(
            "WavLoggerThread stopped: %d frames (%.2f sec)",
            self._frames_written,
            self.duration_seconds,
        )

    def _run(self) -> None:
        """Main thread loop - consume chunks and write to WAV."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                # Handle end-of-stream sentinel
                if item is EndOfStream:
                    logger.debug("WavLoggerThread received EndOfStream")
                    break

                if not isinstance(item, Chunk):
                    logger.warning("WavLoggerThread received non-Chunk: %s", type(item))
                    continue

                self._write_chunk(item)
            finally:
                self._queue.task_done()

    def _write_chunk(self, chunk: Chunk) -> None:
        """
        Extract samples from a Chunk and write to WAV.
        
        Args:
            chunk: Chunk object with samples shaped (channels, samples)
        """
        if self._wf is None:
            return

        samples = chunk.samples  # shape: (channels, samples)
        if samples.size == 0:
            return

        # Transpose to (samples, channels) for interleaved WAV format
        # Then ensure contiguous memory layout
        interleaved = np.ascontiguousarray(samples.T, dtype=np.float32)

        # Handle channel count mismatch
        if interleaved.ndim == 1:
            interleaved = interleaved[:, np.newaxis]
        
        actual_channels = interleaved.shape[1]
        if actual_channels != self._channels:
            # Pad or truncate channels to match expected count
            if actual_channels < self._channels:
                padding = np.zeros(
                    (interleaved.shape[0], self._channels - actual_channels),
                    dtype=np.float32,
                )
                interleaved = np.hstack([interleaved, padding])
            else:
                interleaved = interleaved[:, :self._channels]

        # Convert float32 [-1, 1] to int16  
        # Clip to prevent overflow artifacts
        clipped = np.clip(interleaved, -1.0, 1.0)
        int16_data = (clipped * 32767.0).astype("<i2")  # little-endian int16

        # Write raw frames
        try:
            self._wf.writeframesraw(int16_data.tobytes())
            self._frames_written += int16_data.shape[0]
        except Exception as exc:
            logger.error("Error writing WAV frames: %s", exc)
