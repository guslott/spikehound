from __future__ import annotations

import wave as wave_mod

import numpy as np
import scipy.io.wavfile as wavfile

from daq.base_device import ActualConfig, ChannelInfo
from daq.file_source import FileSource


def test_24bit_wav_reads_and_normalizes_to_unit_range(tmp_path):
    """Finding 6.6b: the docstring promises 24-bit PCM support.

    scipy.io.wavfile reads 24-bit samples *left-justified* into int32 (value << 8),
    so the int32 normalization branch maps them to [-1, 1] without a dedicated
    24-bit branch. This pins that 24-bit playback actually works (and would catch a
    scipy behavior change that silently mis-scaled it).
    """
    sr = 16_000
    full = 1 << 23  # 24-bit full scale
    vals = [0, full // 2, -(full // 2), full - 1, -full]  # includes the rails
    frames = b"".join(int(v).to_bytes(3, "little", signed=True) for v in vals)

    path = tmp_path / "tone24.wav"
    w = wave_mod.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(3)  # 24-bit
    w.setframerate(sr)
    w.writeframes(frames)
    w.close()

    read_sr, data = wavfile.read(str(path))
    assert read_sr == sr
    assert data.dtype == np.int32, "scipy is expected to left-justify 24-bit into int32"

    norm = FileSource()._normalize_chunk(data)
    assert norm.dtype == np.float32

    expected = np.array([v / full for v in vals], dtype=np.float32)
    np.testing.assert_allclose(norm, expected, atol=1e-6)
    assert norm[-1] == -1.0  # negative full scale lands exactly on the rail


def test_run_loop_emits_only_active_channels(monkeypatch):
    source = FileSource()
    source._raw_data = np.arange(128, dtype=np.int16).reshape(64, 2)
    source._sample_rate = 20_000
    source._n_channels = 2
    source._n_frames = 64
    source._available_channels = [
        ChannelInfo(id=0, name="Channel 1", units="V"),
        ChannelInfo(id=1, name="Channel 2", units="V"),
    ]
    source._active_channel_ids = [0]
    source.config = ActualConfig(
        sample_rate=20_000,
        channels=[source._available_channels[0]],
        chunk_size=32,
        dtype="float32",
    )

    emitted_shapes: list[tuple[int, int]] = []

    def _capture_emit(data, *, mono_time=None, device_time=None):
        emitted_shapes.append(data.shape)
        source.stop_event.set()

    monkeypatch.setattr(source, "emit_array", _capture_emit)

    source._run_loop()

    assert emitted_shapes == [(32, 1)]


def test_run_loop_lets_base_device_compute_chunk_timestamps(monkeypatch):
    source = FileSource()
    source._raw_data = np.arange(128, dtype=np.int16).reshape(64, 2)
    source._sample_rate = 20_000
    source._n_channels = 2
    source._n_frames = 64
    source._available_channels = [
        ChannelInfo(id=0, name="Channel 1", units="V"),
        ChannelInfo(id=1, name="Channel 2", units="V"),
    ]
    source._active_channel_ids = [0]
    source.config = ActualConfig(
        sample_rate=20_000,
        channels=[source._available_channels[0]],
        chunk_size=32,
        dtype="float32",
    )

    seen_mono_times: list[float | None] = []

    def _capture_emit(data, *, mono_time=None, device_time=None):
        seen_mono_times.append(mono_time)
        source.stop_event.set()

    monkeypatch.setattr(source, "emit_array", _capture_emit)

    source._run_loop()

    assert seen_mono_times == [None]


def test_run_loop_breaks_when_raw_data_detached_midloop(monkeypatch):
    """Finding 6.3: if close() detaches _raw_data while the worker runs, the
    loop must exit cleanly instead of slicing None (use-after-close)."""
    source = FileSource()
    source._raw_data = np.arange(256, dtype=np.int16).reshape(128, 2)
    source._sample_rate = 20_000
    source._n_channels = 2
    source._n_frames = 128
    source._available_channels = [
        ChannelInfo(id=0, name="Channel 1", units="V"),
        ChannelInfo(id=1, name="Channel 2", units="V"),
    ]
    source._active_channel_ids = [0]
    source.config = ActualConfig(
        sample_rate=20_000,
        channels=[source._available_channels[0]],
        chunk_size=32,
        dtype="float32",
    )

    emit_count = 0

    def _detaching_emit(data, *, mono_time=None, device_time=None):
        nonlocal emit_count
        emit_count += 1
        # Simulate a concurrent close() detaching the data after the first emit.
        source._raw_data = None

    monkeypatch.setattr(source, "emit_array", _detaching_emit)

    # stop_event is never set; the loop must still exit via the None snapshot
    # guard rather than crashing on the next iteration's slice of None.
    source._run_loop()

    assert emit_count == 1
