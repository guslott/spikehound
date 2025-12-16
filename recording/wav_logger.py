"""WAV file logger thread for recording streamed audio data.

Adapted from student implementation in feature/logging branch.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import struct
from typing import Optional, Union
import numpy as np

from shared.models import Chunk, EndOfStream

logger = logging.getLogger(__name__)

class WaveWriter32:
    """Writes 32-bit floating point WAV files (IEEE Float)."""
    
    def __init__(self, f, channels: int, sample_rate: int):
        self._f = f
        self._channels = channels
        self._sample_rate = sample_rate
        self._data_size = 0
        self._write_header()
        
    def _write_header(self):
        """Write placeholder WAV header."""
        # RIFF header
        self._f.write(b'RIFF')
        self._f.write(struct.pack('<I', 0))  # ChunkSize (placeholder)
        self._f.write(b'WAVE')
        
        # fmt chunk
        self._f.write(b'fmt ')
        self._f.write(struct.pack('<I', 16)) # Subchunk1Size (16 for PCM)
        self._f.write(struct.pack('<H', 3))  # AudioFormat (3 = IEEE Float)
        self._f.write(struct.pack('<H', self._channels))
        self._f.write(struct.pack('<I', self._sample_rate))
        bytes_per_sample = 4 # float32
        block_align = self._channels * bytes_per_sample
        byte_rate = self._sample_rate * block_align
        self._f.write(struct.pack('<I', byte_rate))
        self._f.write(struct.pack('<H', block_align))
        self._f.write(struct.pack('<H', bytes_per_sample * 8)) # BitsPerSample
        
        # data chunk
        self._f.write(b'data')
        self._f.write(struct.pack('<I', 0))  # Subchunk2Size (placeholder)
        
    def write_frames(self, data: np.ndarray):
        """Write raw float32 bytes."""
        byte_data = data.tobytes()
        self._f.write(byte_data)
        self._data_size += len(byte_data)
        
    def close(self):
        """Update header lengths and close."""
        if not self._f.closed:
            # Update sizes
            file_size = 4 + (8 + 16) + (8 + self._data_size) # RIFF + fmt + data
            self._f.seek(4)
            self._f.write(struct.pack('<I', file_size))
            
            # data chunk size
            self._f.seek(40)
            self._f.write(struct.pack('<I', self._data_size))
            self._f.close()

class WavLoggerThread:
    """
    Consumes Chunk objects from a queue and writes to a WAV file.
    Supports standard 16-bit PCM (default) or 32-bit Float (Pro).
    """

    def __init__(
        self,
        data_queue: "queue.Queue[Union[Chunk, type[EndOfStream]]]",
        out_path: str,
        sample_rate: int,
        channels: int,
        use_float32: bool = False,
    ) -> None:
        self._queue = data_queue
        self._out_path = out_path
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._use_float32 = use_float32

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # _writer is either WaveWriter32, or a wave.Wave_write object
        self._writer = None
        self._frames_written: int = 0

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def duration_seconds(self) -> float:
        if self._sample_rate <= 0:
            return 0.0
        return self._frames_written / self._sample_rate

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("WavLoggerThread already running")
            return

        out_dir = os.path.dirname(self._out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        try:
            if self._use_float32:
                f = open(self._out_path, "wb")
                self._writer = WaveWriter32(f, self._channels, self._sample_rate)
            else:
                import wave
                w = wave.open(self._out_path, "wb")
                w.setnchannels(self._channels)
                w.setsampwidth(2)  # 16-bit
                w.setframerate(self._sample_rate)
                self._writer = w
        except Exception as exc:
            logger.error("Failed to open WAV file %s: %s", self._out_path, exc)
            raise

        self._stop_event.clear()
        self._frames_written = 0
        self._thread = threading.Thread(
            target=self._run,
            name="WavLoggerThread",
            daemon=False,  # Non-daemon to ensure WAV header is finalized on exit
        )
        self._thread.start()
        
        fmt = "32-bit float" if self._use_float32 else "16-bit PCM"
        logger.info(
            "WavLoggerThread started: %s (sr=%d, ch=%d, %s)",
            self._out_path,
            self._sample_rate,
            self._channels,
            fmt,
        )

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logger.warning("WavLoggerThread did not stop within timeout")
            self._thread = None

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                logger.warning("Error closing WAV file: %s", exc)
            self._writer = None

        logger.info(
            "WavLoggerThread stopped: %d frames (%.2f sec)",
            self._frames_written,
            self.duration_seconds,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
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
        if self._writer is None:
            return

        samples = chunk.samples
        if samples.size == 0:
            return

        # Transpose to (samples, channels)
        interleaved = np.ascontiguousarray(samples.T, dtype=np.float32)

        if interleaved.ndim == 1:
            interleaved = interleaved[:, np.newaxis]
        
        actual_channels = interleaved.shape[1]
        if actual_channels != self._channels:
            if actual_channels < self._channels:
                padding = np.zeros(
                    (interleaved.shape[0], self._channels - actual_channels),
                    dtype=np.float32,
                )
                interleaved = np.hstack([interleaved, padding])
            else:
                interleaved = interleaved[:, :self._channels]

        try:
            if self._use_float32:
                # No clipping needed for float32
                # WaveWriter32.write_frames expects ndarray
                self._writer.write_frames(interleaved)
            else:
                # Convert to int16
                # (samples, channels) flattened to bytes
                # 32767 is max int16
                pcm = (interleaved * 32767).clip(-32768, 32767).astype(np.int16)
                # wave module expects bytes
                self._writer.writeframes(pcm.tobytes())
            
            self._frames_written += interleaved.shape[0]
            
        except Exception as exc:
            logger.error("Error writing WAV frames: %s", exc)
