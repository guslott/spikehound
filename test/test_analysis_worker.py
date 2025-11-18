import numpy as np
import pytest

from analysis.analysis_worker import AnalysisWorker, _peak_frequency_sinc
from analysis.settings import AnalysisSettingsStore
from shared.models import Chunk
from shared.event_buffer import AnalysisEvents, EventRingBuffer
from shared.types import Event


class _DummyController:
    def __init__(self, *, capacity: int = 8) -> None:
        self.event_buffer = EventRingBuffer(capacity=capacity)
        self.analysis_settings_store = AnalysisSettingsStore()

    def register_analysis_queue(self, _queue):
        return 1

    def unregister_analysis_queue(self, _token):
        return None


def _stub_event(event_id: int) -> Event:
    return Event(
        id=event_id,
        channelId=0,
        thresholdValue=0.5,
        crossingIndex=event_id,
        crossingTimeSec=float(event_id),
        firstSampleTimeSec=float(event_id) - 0.005,
        sampleRateHz=1000.0,
        windowMs=10.0,
        preMs=5.0,
        postMs=5.0,
        samples=np.zeros(1, dtype=np.float32),
    )


def test_event_ring_buffer_drops_oldest() -> None:
    buf = EventRingBuffer(capacity=2)
    buf.push(_stub_event(1))
    buf.push(_stub_event(2))
    buf.push(_stub_event(3))
    ids = [ev.id for ev in buf.peek_all()]
    assert ids == [2, 3], "oldest entry should be evicted once capacity is exceeded"

    buf.push(_stub_event(4))
    ids = [ev.id for ev in buf.drain()]
    assert ids == [3, 4]
    assert buf.drain() == []


def test_worker_window_copy_and_timing() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker._channel_index = 0  # direct access for controlled test
    worker.configure_threshold(True, 0.25)
    with worker._state_lock:
        worker._event_window_ms = 10.0

    sr = 20_000
    dt = 1.0 / sr
    crossing_idx = 200
    data = np.zeros((1, 400), dtype=np.float32)
    data[0, crossing_idx] = 1.0
    chunk = Chunk(
        samples=data,
        start_time=1.0,
        dt=dt,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    worker._detect_events(chunk)
    events = controller.event_buffer.drain()
    assert len(events) == 1
    ev = events[0]

    half_samples = int(round((10.0 / 2.0) * sr / 1000.0))
    assert ev.samples.size == 2 * half_samples + 1

    pre_samples = int(round((ev.crossingTimeSec - ev.firstSampleTimeSec) * sr))
    post_samples = ev.samples.size - pre_samples - 1
    assert pre_samples == half_samples
    assert post_samples == half_samples

    expected_first = ev.crossingTimeSec - (ev.windowMs / 2000.0)
    assert ev.firstSampleTimeSec == pytest.approx(expected_first, rel=1e-7)


def test_worker_rejects_event_crossing_secondary_threshold() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker._channel_index = 0
    worker.configure_threshold(True, 0.25, secondary_enabled=True, secondary_value=0.5)

    sr = 20_000
    dt = 1.0 / sr
    data = np.zeros((1, 400), dtype=np.float32)
    crossing_idx = 200
    data[0, crossing_idx] = 0.3  # crosses primary threshold
    data[0, crossing_idx + 10] = 0.6  # exceeds secondary threshold inside the window
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    worker._detect_events(chunk)
    assert controller.event_buffer.drain() == []


def test_worker_accepts_event_when_secondary_not_crossed() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker._channel_index = 0
    worker.configure_threshold(True, 0.25, secondary_enabled=True, secondary_value=0.5)

    sr = 20_000
    dt = 1.0 / sr
    data = np.zeros((1, 400), dtype=np.float32)
    crossing_idx = 200
    data[0, crossing_idx] = 0.4  # crosses primary threshold
    data[0, crossing_idx + 10] = 0.45  # stays below secondary threshold
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    worker._detect_events(chunk)
    events = controller.event_buffer.drain()
    assert len(events) == 1
    assert events[0].crossingIndex == crossing_idx


def test_worker_tracks_interval_since_last_event() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker._channel_index = 0
    worker.configure_threshold(True, 0.25)

    sr = 20_000
    dt = 1.0 / sr
    data = np.zeros((1, 800), dtype=np.float32)
    first_idx = 200
    second_idx = 600
    data[0, first_idx] = 0.4
    data[0, second_idx] = 0.45
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    worker._detect_events(chunk)
    events = controller.event_buffer.drain()
    assert len(events) == 2
    first_event, second_event = events
    assert np.isnan(first_event.intervalSinceLastSec)
    expected_interval = (second_idx - first_idx) * dt
    assert second_event.intervalSinceLastSec == pytest.approx(expected_interval, rel=1e-6)
    assert second_event.properties.get("interval_sec") == pytest.approx(expected_interval, rel=1e-6)


def test_peak_frequency_sinc_detects_clean_tone() -> None:
    sr = 20_000
    duration = 0.01
    freq = 250.0
    t = np.arange(int(sr * duration)) / sr
    wave = 0.6 * np.sin(2 * np.pi * freq * t)
    center = len(wave) // 2
    assert _peak_frequency_sinc(wave, sr, center_index=center) == pytest.approx(freq, rel=0.05)


def test_peak_frequency_sinc_ignores_dc_and_slope() -> None:
    sr = 20_000
    duration = 0.012
    freq = 180.0
    t = np.arange(int(sr * duration)) / sr
    wave = 0.4 * np.sin(2 * np.pi * freq * t)
    wave += 0.3  # DC offset
    wave += 0.05 * (t - t.mean())  # linear drift
    center = len(wave) // 2
    assert _peak_frequency_sinc(wave, sr, center_index=center) == pytest.approx(freq, rel=0.08)


def test_peak_frequency_sinc_focuses_on_localized_event() -> None:
    sr = 20_000
    samples = int(sr * 0.02)
    wave = np.zeros(samples, dtype=np.float64)
    burst_len = int(sr * 0.004)
    start = samples // 2 - burst_len // 2
    t = np.arange(burst_len) / sr
    freq = 220.0
    burst = np.sin(2 * np.pi * freq * t) * np.hanning(burst_len)
    wave[start : start + burst_len] = burst
    center = start + burst_len // 2
    assert _peak_frequency_sinc(wave, sr, center_index=center) == pytest.approx(freq, rel=0.1)

def test_analysis_events_pull_since() -> None:
    buf = EventRingBuffer(capacity=2)
    bus = AnalysisEvents(buf)
    buf.push(_stub_event(1))
    events, last_id = bus.pull_events()
    assert [ev.id for ev in events] == [1]
    assert last_id == 1

    buf.push(_stub_event(2))
    buf.push(_stub_event(3))  # evict id 1
    events, last_id = bus.pull_events(last_id)
    assert [ev.id for ev in events] == [2, 3]
    assert last_id == 3

    events, last_id = bus.pull_events(last_id)
    assert events == []
