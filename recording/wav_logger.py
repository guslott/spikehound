"""WAV file logger thread for recording streamed audio data.

Adapted from student implementation in feature/logging branch.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import struct
import time
from typing import Optional, Union
import numpy as np

from shared.models import Chunk, EndOfStream

logger = logging.getLogger(__name__)

class WaveWriter32:
    """Writes 32-bit IEEE-float WAV files (``WAVE_FORMAT_IEEE_FLOAT``).

    The header uses the non-PCM layout strict readers expect (finding 8.1): an
    18-byte ``fmt `` chunk (with ``cbSize``) plus a ``fact`` chunk carrying the
    per-channel sample-frame count. Every size field is patched from a byte
    offset captured at write time rather than a hardcoded ``seek`` — so the
    layout can change without silently corrupting the patch sites.

    :meth:`flush` patches those sizes incrementally and ``fsync``s, so a crash
    mid-recording still leaves a readable file instead of a 0-size placeholder
    (finding 8.2).
    """

    _BYTES_PER_SAMPLE = 4  # float32

    def __init__(self, f, channels: int, sample_rate: int):
        self._f = f
        self._channels = int(channels)
        self._sample_rate = int(sample_rate)
        self._data_size = 0
        # Byte offsets of the patchable size fields, captured in _write_header().
        self._riff_size_pos = 0
        self._fact_frames_pos = 0
        self._data_size_pos = 0
        self._write_header()

    def _write_header(self):
        """Write the header with placeholder sizes, recording their offsets."""
        f = self._f
        block_align = self._channels * self._BYTES_PER_SAMPLE
        byte_rate = self._sample_rate * block_align

        # RIFF header
        f.write(b'RIFF')
        self._riff_size_pos = f.tell()
        f.write(struct.pack('<I', 0))  # RIFF chunk size (patched on flush/close)
        f.write(b'WAVE')

        # fmt chunk — 18 bytes incl. cbSize, as required for non-PCM formats
        f.write(b'fmt ')
        f.write(struct.pack('<I', 18))
        f.write(struct.pack('<H', 3))  # AudioFormat = 3 (IEEE Float)
        f.write(struct.pack('<H', self._channels))
        f.write(struct.pack('<I', self._sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', self._BYTES_PER_SAMPLE * 8))  # BitsPerSample
        f.write(struct.pack('<H', 0))  # cbSize (no format extension)

        # fact chunk — sample-frame count per channel (recommended for non-PCM)
        f.write(b'fact')
        f.write(struct.pack('<I', 4))
        self._fact_frames_pos = f.tell()
        f.write(struct.pack('<I', 0))  # frame count (patched on flush/close)

        # data chunk
        f.write(b'data')
        self._data_size_pos = f.tell()
        f.write(struct.pack('<I', 0))  # data size (patched on flush/close)

    def write_frames(self, data: np.ndarray):
        """Append interleaved float32 frames."""
        byte_data = np.ascontiguousarray(data, dtype=np.float32).tobytes()
        self._f.write(byte_data)
        self._data_size += len(byte_data)

    def _patch_sizes(self):
        """Patch the RIFF/fact/data size fields from the current end position."""
        f = self._f
        end = f.tell()
        block_align = (self._channels * self._BYTES_PER_SAMPLE) or 1
        frame_count = self._data_size // block_align

        f.seek(self._riff_size_pos)
        f.write(struct.pack('<I', max(0, end - 8)))  # all bytes after 'RIFF'+size
        f.seek(self._fact_frames_pos)
        f.write(struct.pack('<I', frame_count))
        f.seek(self._data_size_pos)
        f.write(struct.pack('<I', self._data_size))
        f.seek(end)  # restore the append position for continued writing

    def flush(self):
        """Patch header sizes and fsync so a partial file stays readable (8.2)."""
        if self._f.closed:
            return
        self._patch_sizes()
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except (OSError, ValueError):
            # Best-effort: some file-likes / platforms don't support fsync.
            pass

    def close(self):
        """Finalize header sizes, fsync, and close."""
        if not self._f.closed:
            self._patch_sizes()
            self._f.flush()
            try:
                os.fsync(self._f.fileno())
            except (OSError, ValueError):
                pass
            self._f.close()

class WavLoggerThread:
    """
    Consumes Chunk objects from a queue and writes to a WAV file.
    Supports standard 16-bit PCM (default) or 32-bit Float (Pro).
    """

    # How often the float32 writer patches header sizes + fsyncs so a crash
    # mid-recording leaves a readable file (finding 8.2). The 16-bit stdlib
    # `wave` writer only finalizes on close(), so this applies to float32 only.
    _FLUSH_INTERVAL_SEC = 1.0

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

    def _maybe_flush(self, last_flush: float) -> float:
        """Periodically patch header sizes + fsync the float32 writer (8.2)."""
        if not self._use_float32 or self._writer is None:
            return last_flush
        now = time.monotonic()
        if now - last_flush < self._FLUSH_INTERVAL_SEC:
            return last_flush
        try:
            self._writer.flush()
        except Exception:
            logger.debug("Periodic WAV flush failed", exc_info=True)
        return now

    def _run(self) -> None:
        last_flush = time.monotonic()
        while True:
            try:
                # Small timeout to allow periodic check of stop_event
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                last_flush = self._maybe_flush(last_flush)
                if self._stop_event.is_set():
                    break
                continue

            try:
                if item is EndOfStream:
                    logger.debug("WavLoggerThread received EndOfStream")
                    break

                if not isinstance(item, Chunk):
                    logger.warning("WavLoggerThread received non-Chunk: %s", type(item))
                    continue

                self._write_chunk(item)
            except Exception:
                logger.error("Error in WavLoggerThread loop", exc_info=True)
            finally:
                self._queue.task_done()

            last_flush = self._maybe_flush(last_flush)

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
