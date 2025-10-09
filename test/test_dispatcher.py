import math
import queue

import numpy as np
from core import (
    ChannelFilterSettings,
    Chunk,
    Dispatcher,
    EndOfStream,
    FilterSettings,
    SignalConditioner,
)


def _drain_chunks(target_queue: "queue.Queue"):
    items = []
    while True:
        try:
            item = target_queue.get_nowait()
        except queue.Empty:
            break
        if item is EndOfStream:
            continue
        items.append(item)
    return items


def _make_chunk(samples: np.ndarray, *, start_time: float, dt: float, seq: int) -> Chunk:
    return Chunk(
        samples=samples,
        start_time=start_time,
        dt=dt,
        seq=seq,
        channel_names=tuple(f"ch{i}" for i in range(samples.shape[0])),
        units="V",
        meta={"start_sample": seq * samples.shape[1]},
    )


def test_dispatcher_filters_dc_and_notch_at_target_frequency():
    fs = 1_000.0
    dt = 1.0 / fs
    duration = 1.0
    frames = int(duration * fs)
    t = np.arange(frames, dtype=np.float32) * dt
    dc_offset = 0.75
    notch_freq = 60.0
    sine = np.sin(2.0 * math.pi * notch_freq * t)
    raw_signal = (dc_offset + sine).astype(np.float32)
    samples = raw_signal.reshape(1, -1)
    chunk = _make_chunk(samples, start_time=0.0, dt=dt, seq=0)

    raw_queue: "queue.Queue" = queue.Queue()
    visualization_queue: "queue.Queue" = queue.Queue()
    analysis_queue: "queue.Queue" = queue.Queue()
    audio_queue: "queue.Queue" = queue.Queue()
    logging_queue: "queue.Queue" = queue.Queue()

    settings = FilterSettings(
        default=ChannelFilterSettings(
            ac_couple=True,
            ac_cutoff_hz=1.0,
            notch_enabled=True,
            notch_freq_hz=notch_freq,
            notch_q=35.0,
        )
    )
    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        analysis_queue,
        audio_queue,
        logging_queue,
        filter_settings=settings,
    )
    dispatcher.start()
    raw_queue.put(chunk)
    raw_queue.put(EndOfStream)
    dispatcher.join(timeout=2.0)

    filtered_chunks = _drain_chunks(visualization_queue)
    assert len(filtered_chunks) == 1
    filtered = np.asarray(filtered_chunks[0].samples)[0]

    # AC coupling should remove DC
    assert abs(float(np.mean(filtered))) < 5e-3

    conditioner_no_notch = SignalConditioner(
        FilterSettings(default=ChannelFilterSettings(ac_couple=True, ac_cutoff_hz=1.0))
    )
    baseline = conditioner_no_notch.process(chunk)[0]
    baseline -= np.mean(baseline)
    baseline_rms = float(np.sqrt(np.mean(baseline**2)))

    filtered_rms = float(np.sqrt(np.mean(filtered**2)))
    assert filtered_rms < baseline_rms * 0.35

    # Ensure raw chunks routed untouched
    logged_chunks = _drain_chunks(logging_queue)
    assert len(logged_chunks) == 1
    assert logged_chunks[0] is chunk


def test_dispatcher_preserves_filter_state_across_chunks():
    fs = 2_000.0
    dt = 1.0 / fs
    total_frames = 200
    impulse = np.zeros(total_frames, dtype=np.float32)
    impulse[10] = 1.0

    first_len = 120
    second_len = total_frames - first_len

    chunk1 = _make_chunk(impulse[:first_len].reshape(1, -1), start_time=0.0, dt=dt, seq=0)
    chunk2 = _make_chunk(
        impulse[first_len:].reshape(1, -1),
        start_time=chunk1.start_time + chunk1.n_samples * chunk1.dt,
        dt=dt,
        seq=1,
    )

    raw_queue: "queue.Queue" = queue.Queue()
    visualization_queue: "queue.Queue" = queue.Queue()
    analysis_queue: "queue.Queue" = queue.Queue()
    audio_queue: "queue.Queue" = queue.Queue()
    logging_queue: "queue.Queue" = queue.Queue()

    settings = FilterSettings(
        default=ChannelFilterSettings(lowpass_hz=200.0, lowpass_order=4)
    )
    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        analysis_queue,
        audio_queue,
        logging_queue,
        filter_settings=settings,
    )
    dispatcher.start()
    raw_queue.put(chunk1)
    raw_queue.put(chunk2)
    raw_queue.put(EndOfStream)
    dispatcher.join(timeout=2.0)

    filtered_chunks = _drain_chunks(visualization_queue)
    assert len(filtered_chunks) == 2
    combined = np.concatenate([np.asarray(c.samples)[0] for c in filtered_chunks])

    conditioner = SignalConditioner(settings)
    full_chunk = _make_chunk(impulse.reshape(1, -1), start_time=0.0, dt=dt, seq=0)
    expected = conditioner.process(full_chunk)[0]

    assert np.allclose(combined, expected, atol=1e-4)


def test_dispatcher_fan_out_and_backpressure_tracking():
    raw_queue: "queue.Queue" = queue.Queue()
    visualization_queue: "queue.Queue" = queue.Queue(maxsize=1)
    analysis_queue: "queue.Queue" = queue.Queue(maxsize=1)
    audio_queue: "queue.Queue" = queue.Queue(maxsize=1)
    logging_queue: "queue.Queue" = queue.Queue()

    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        analysis_queue,
        audio_queue,
        logging_queue,
        filter_settings=FilterSettings(),
    )
    dispatcher.start()

    fs = 1_000.0
    dt = 1.0 / fs
    frames = 16

    for seq in range(3):
        value = float(seq + 1)
        samples = (np.ones((1, frames), dtype=np.float32) * value)
        chunk = _make_chunk(samples, start_time=seq * frames * dt, dt=dt, seq=seq)
        raw_queue.put(chunk)

    raw_queue.put(EndOfStream)
    dispatcher.join(timeout=2.0)

    stats = dispatcher.snapshot()
    assert stats["received"] == 3
    assert stats["processed"] == 3
    forwarded = stats["forwarded"]
    assert forwarded.get("visualization") == 3
    assert forwarded.get("analysis") == 3
    assert forwarded.get("audio") == 3
    assert forwarded.get("logging") == 3

    evicted = stats["evicted"]
    assert evicted.get("visualization", 0) >= 1
    assert evicted.get("analysis", 0) >= 1
    assert evicted.get("audio", 0) >= 1
    assert evicted.get("logging", 0) == 0

    # Logging queue should hold all raw chunks plus sentinel
    logged_chunks = _drain_chunks(logging_queue)
    assert len(logged_chunks) == 3
    assert logged_chunks[0].seq == 0
    assert logged_chunks[-1].seq == 2
