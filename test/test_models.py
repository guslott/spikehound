import pickle

import numpy as np
import pytest

from core import Chunk, Event
from daq.simulated_source import SimulatedPhysiologySource


def test_chunk_samples_are_channel_major_and_readonly():
    rng = np.random.default_rng(123)
    samples = rng.standard_normal((3, 64)).astype(np.float32)
    chunk = Chunk(
        samples=samples,
        start_time=1.5,
        dt=0.001,
        seq=42,
        channel_names=("ch0", "ch1", "ch2"),
        units="V",
        meta={"start_sample": 100},
    )

    assert chunk.samples.shape == (3, 64)
    assert chunk.n_channels == 3
    assert chunk.n_samples == 64
    assert not chunk.samples.flags.writeable

    original_value = chunk.samples[0, 0]
    samples[0, 0] = 999.0  # mutate the source array; chunk should remain unchanged
    assert chunk.samples[0, 0] == original_value

    with pytest.raises(ValueError):
        chunk.samples[0, 0] = 0.0


def test_emit_array_produces_monotonic_sequence_and_metadata():
    source = SimulatedPhysiologySource()
    device = source.list_available_devices()[0]
    source.open(device.id)

    available_channels = source.list_available_channels(device.id)
    channel_ids = [ch.id for ch in available_channels]
    cfg = source.configure(
        sample_rate=1000,
        channels=channel_ids,
        chunk_size=8,
        num_units=1,
    )

    frames = cfg.chunk_size
    chans = len(cfg.channels)
    data = np.ones((frames, chans), dtype=np.float32)
    dt = 1.0 / cfg.sample_rate

    chunk_a = source.emit_array(data, mono_time=1.0)
    chunk_b = source.emit_array(data, mono_time=1.0 + frames * dt)

    assert chunk_a.seq == 0
    assert chunk_b.seq == 1
    assert chunk_a.meta is not None and chunk_a.meta["start_sample"] == 0
    assert chunk_b.meta is not None and chunk_b.meta["start_sample"] == frames
    assert np.array_equal(chunk_a.samples, data.T)
    assert chunk_b.start_time == pytest.approx(chunk_a.start_time + chunk_a.n_samples * chunk_a.dt)

    source.close()


def test_chunk_and_event_pickle_roundtrip():
    rng = np.random.default_rng(321)
    samples = rng.standard_normal((2, 32)).astype(np.float32)
    chunk = Chunk(
        samples=samples,
        start_time=2.0,
        dt=0.0005,
        seq=7,
        channel_names=("left", "right"),
        units="V",
        meta={"start_sample": 224, "device_time": 1.25},
    )

    restored_chunk = pickle.loads(pickle.dumps(chunk))
    assert restored_chunk.seq == chunk.seq
    assert restored_chunk.start_time == chunk.start_time
    assert restored_chunk.dt == chunk.dt
    assert restored_chunk.channel_names == chunk.channel_names
    assert restored_chunk.units == chunk.units
    assert restored_chunk.meta == chunk.meta
    assert np.array_equal(restored_chunk.samples, chunk.samples)
    assert not restored_chunk.samples.flags.writeable

    window = rng.standard_normal(64).astype(np.float32)
    event = Event(
        t=3.5,
        chan=1,
        window=window,
        properties={"peak_amp": 0.87, "energy": float(np.dot(window, window))},
        params={"threshold": 0.5, "window_size": 64},
    )

    restored_event = pickle.loads(pickle.dumps(event))
    assert restored_event.t == event.t
    assert restored_event.chan == event.chan
    assert restored_event.properties == event.properties
    assert restored_event.params == event.params
    assert np.array_equal(restored_event.window, event.window)
    assert not restored_event.window.flags.writeable

    with pytest.raises(ValueError):
        restored_event.window[0] = 0.0
