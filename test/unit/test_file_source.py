from __future__ import annotations

import numpy as np

from daq.base_device import ActualConfig, ChannelInfo
from daq.file_source import FileSource


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
