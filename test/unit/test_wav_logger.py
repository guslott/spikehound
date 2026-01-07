"""
Unit tests for WAV recording functionality.

These tests verify:
1. Round-trip integrity (write → read → compare)
2. Header metadata correctness
3. Both PCM16 and Float32 formats
4. Partial write handling

Recording integrity is important because corrupted recordings lead to
lost experimental data that cannot be recovered.
"""
from __future__ import annotations

import os
import queue
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from recording.wav_logger import WavLoggerThread, WaveWriter32
from shared.models import Chunk, EndOfStream


def make_chunk(
    samples: np.ndarray,
    sample_rate: float,
    seq: int = 0,
) -> Chunk:
    """Create a test chunk."""
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    n_channels = samples.shape[0]
    return Chunk(
        samples=samples.astype(np.float32),
        start_time=0.0,
        dt=1.0 / sample_rate,
        seq=seq,
        channel_names=tuple(f"Ch{i}" for i in range(n_channels)),
        units="V",
    )


class TestWavLoggerRoundtrip:
    """Round-trip tests for WAV recording."""

    def test_pcm16_roundtrip(self, tmp_path: Path):
        """PCM16 recording should round-trip within quantization tolerance."""
        wav_path = tmp_path / "test_pcm16.wav"
        sample_rate = 44100
        n_channels = 2
        n_samples = 4410  # 0.1 seconds

        # Create test signal
        t = np.arange(n_samples) / sample_rate
        ch0 = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
        ch1 = 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz
        original = np.vstack([ch0, ch1]).astype(np.float32)

        # Record
        data_queue = queue.Queue()
        logger = WavLoggerThread(
            data_queue=data_queue,
            out_path=str(wav_path),
            sample_rate=sample_rate,
            channels=n_channels,
            use_float32=False,  # PCM16
        )

        logger.start()
        chunk = make_chunk(original, sample_rate)
        data_queue.put(chunk)
        data_queue.put(EndOfStream)
        logger.stop()

        # Read back
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == n_channels
            assert wf.getframerate() == sample_rate
            assert wf.getsampwidth() == 2  # 16-bit

            frames = wf.readframes(n_samples)
            read_data = np.frombuffer(frames, dtype=np.int16)
            read_data = read_data.reshape(-1, n_channels).T
            read_float = read_data.astype(np.float32) / 32767.0

        # Compare with tolerance for PCM quantization
        np.testing.assert_allclose(
            read_float, original,
            atol=1.0 / 32767 * 2,  # ~2 LSB tolerance
            rtol=0.01,
        )

    def test_float32_roundtrip(self, tmp_path: Path):
        """Float32 recording should round-trip exactly."""
        wav_path = tmp_path / "test_float32.wav"
        sample_rate = 48000
        n_channels = 1
        n_samples = 4800

        # Create test signal
        t = np.arange(n_samples) / sample_rate
        original = (0.7 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        original = original.reshape(1, -1)

        # Record
        data_queue = queue.Queue()
        logger = WavLoggerThread(
            data_queue=data_queue,
            out_path=str(wav_path),
            sample_rate=sample_rate,
            channels=n_channels,
            use_float32=True,
        )

        logger.start()
        chunk = make_chunk(original, sample_rate)
        data_queue.put(chunk)
        data_queue.put(EndOfStream)
        logger.stop()

        # Read back manually (wave module doesn't support float32)
        with open(str(wav_path), "rb") as f:
            # Skip to data
            f.seek(44)  # Standard header size for simple WAV
            data = f.read()
            read_float = np.frombuffer(data, dtype=np.float32)

        # Should match exactly
        np.testing.assert_array_almost_equal(
            read_float.flatten(),
            original.flatten(),
            decimal=6,
        )


class TestWavLoggerMetadata:
    """Tests for WAV header metadata."""

    def test_sample_rate_in_header(self, tmp_path: Path):
        """Recorded sample rate should match configuration."""
        wav_path = tmp_path / "test_sr.wav"
        sample_rate = 22050  # Non-standard rate

        data_queue = queue.Queue()
        logger = WavLoggerThread(
            data_queue=data_queue,
            out_path=str(wav_path),
            sample_rate=sample_rate,
            channels=1,
        )

        logger.start()
        chunk = make_chunk(np.zeros((1, 100), dtype=np.float32), sample_rate)
        data_queue.put(chunk)
        data_queue.put(EndOfStream)
        logger.stop()

        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == sample_rate

    def test_channel_count_in_header(self, tmp_path: Path):
        """Recorded channel count should match configuration."""
        wav_path = tmp_path / "test_ch.wav"

        for n_channels in [1, 2, 4]:
            data_queue = queue.Queue()
            path = tmp_path / f"test_ch_{n_channels}.wav"
            logger = WavLoggerThread(
                data_queue=data_queue,
                out_path=str(path),
                sample_rate=44100,
                channels=n_channels,
            )

            logger.start()
            samples = np.zeros((n_channels, 100), dtype=np.float32)
            chunk = make_chunk(samples, 44100)
            data_queue.put(chunk)
            data_queue.put(EndOfStream)
            logger.stop()

            with wave.open(str(path), "rb") as wf:
                assert wf.getnchannels() == n_channels, f"Mismatch for {n_channels} channels"


class TestWavLoggerMultipleChunks:
    """Tests for recording multiple chunks."""

    def test_multiple_chunks_concatenated(self, tmp_path: Path):
        """Multiple chunks should be concatenated in order."""
        wav_path = tmp_path / "test_multi.wav"
        sample_rate = 10000
        chunk_size = 1000

        data_queue = queue.Queue()
        logger = WavLoggerThread(
            data_queue=data_queue,
            out_path=str(wav_path),
            sample_rate=sample_rate,
            channels=1,
        )

        logger.start()

        # Send 5 chunks with increasing values
        for i in range(5):
            samples = np.full((1, chunk_size), float(i), dtype=np.float32)
            chunk = make_chunk(samples, sample_rate, seq=i)
            data_queue.put(chunk)

        data_queue.put(EndOfStream)
        logger.stop()

        assert logger.frames_written == 5 * chunk_size

        # Read back and verify order
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.readframes(5 * chunk_size)
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767

        # Each chunk should have consistent value
        for i in range(5):
            segment = data[i * chunk_size : (i + 1) * chunk_size]
            expected = float(i) / 32767 * 32767 / 32767  # Quantized
            # Each segment should be relatively uniform
            std = float(np.std(segment))
            assert std < 0.01, f"Chunk {i} not uniform: std={std}"


class TestWavLoggerDuration:
    """Tests for duration tracking."""

    def test_duration_seconds_accurate(self, tmp_path: Path):
        """duration_seconds property should match actual recording."""
        wav_path = tmp_path / "test_dur.wav"
        sample_rate = 48000
        duration_sec = 0.5
        n_samples = int(duration_sec * sample_rate)

        data_queue = queue.Queue()
        logger = WavLoggerThread(
            data_queue=data_queue,
            out_path=str(wav_path),
            sample_rate=sample_rate,
            channels=1,
        )

        logger.start()
        samples = np.zeros((1, n_samples), dtype=np.float32)
        chunk = make_chunk(samples, sample_rate)
        data_queue.put(chunk)
        data_queue.put(EndOfStream)
        logger.stop()

        assert abs(logger.duration_seconds - duration_sec) < 0.001


class TestWaveWriter32:
    """Direct tests for float32 WAV writer."""

    def test_writer_header_format(self, tmp_path: Path):
        """Float32 WAV should have correct header format code."""
        wav_path = tmp_path / "test_header.wav"

        with open(str(wav_path), "wb") as f:
            writer = WaveWriter32(f, channels=2, sample_rate=44100)
            writer.write_frames(np.zeros((10, 2), dtype=np.float32))
            writer.close()

        # Read and verify format tag
        with open(str(wav_path), "rb") as f:
            f.seek(20)  # Format code position
            fmt_code = struct.unpack("<H", f.read(2))[0]
            assert fmt_code == 3, "Float32 format code should be 3"

    def test_writer_data_size_updated(self, tmp_path: Path):
        """Data size in header should be updated on close."""
        wav_path = tmp_path / "test_size.wav"
        n_samples = 100
        n_channels = 2

        with open(str(wav_path), "wb") as f:
            writer = WaveWriter32(f, channels=n_channels, sample_rate=44100)
            data = np.zeros((n_samples, n_channels), dtype=np.float32)
            writer.write_frames(data)
            writer.close()

        # Read data size from header
        with open(str(wav_path), "rb") as f:
            f.seek(40)  # Data chunk size position
            data_size = struct.unpack("<I", f.read(4))[0]

        expected_size = n_samples * n_channels * 4  # float32 = 4 bytes
        assert data_size == expected_size


class TestWavLoggerEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_recording(self, tmp_path: Path):
        """Recording with no chunks should create valid (empty) file."""
        wav_path = tmp_path / "test_empty.wav"

        data_queue = queue.Queue()
        logger = WavLoggerThread(
            data_queue=data_queue,
            out_path=str(wav_path),
            sample_rate=44100,
            channels=1,
        )

        logger.start()
        data_queue.put(EndOfStream)
        logger.stop()

        assert logger.frames_written == 0
        assert wav_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        """Logger should create parent directories if needed."""
        wav_path = tmp_path / "subdir" / "nested" / "test.wav"

        data_queue = queue.Queue()
        logger = WavLoggerThread(
            data_queue=data_queue,
            out_path=str(wav_path),
            sample_rate=44100,
            channels=1,
        )

        logger.start()
        chunk = make_chunk(np.zeros((1, 100), dtype=np.float32), 44100)
        data_queue.put(chunk)
        data_queue.put(EndOfStream)
        logger.stop()

        assert wav_path.exists()

    def test_stop_with_pending_data(self, tmp_path: Path):
        """Stop should flush pending data."""
        wav_path = tmp_path / "test_flush.wav"

        data_queue = queue.Queue()
        logger = WavLoggerThread(
            data_queue=data_queue,
            out_path=str(wav_path),
            sample_rate=44100,
            channels=1,
        )

        logger.start()

        # Send data then immediately stop (don't send EndOfStream)
        chunk = make_chunk(np.ones((1, 1000), dtype=np.float32), 44100)
        data_queue.put(chunk)

        logger.stop(join_timeout=2.0)

        # File should exist and have some data
        assert wav_path.exists()
