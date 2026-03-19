import numpy as np
import pytest

from analysis import AnalysisBatch
from analysis.analysis_worker import AnalysisWorker
from analysis.metrics import peak_frequency_sinc as _peak_frequency_sinc
from analysis.settings import AnalysisSettingsStore
from shared.models import ChannelInfo, Chunk


class _DummyController:
    def __init__(self, *, active_channels: list[ChannelInfo] | None = None) -> None:
        self.analysis_settings_store = AnalysisSettingsStore()
        self._active_channels = list(active_channels or [])

    def register_analysis_queue(self, _queue):
        return 1

    def unregister_analysis_queue(self, _token):
        return None

    def active_channels(self):
        return list(self._active_channels)


def _next_batch(worker: AnalysisWorker) -> AnalysisBatch:
    batch = worker.output_queue.get_nowait()
    assert isinstance(batch, AnalysisBatch)
    return batch


def test_worker_emits_analysis_batch_without_legacy_event_meta() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker._channel_index = 0
    worker.configure_threshold(True, 0.25)

    sr = 20_000
    dt = 1.0 / sr
    crossing_idx = 200
    data = np.zeros((1, 400), dtype=np.float32)
    data[0, crossing_idx] = 1.0
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    worker._forward_chunk(chunk)
    batch = _next_batch(worker)

    assert len(batch.events) == 1
    assert batch.events[0].crossingIndex == crossing_idx
    assert "analysis_events" not in (batch.chunk.meta or {})


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

    events = worker._detect_events(chunk)
    assert len(events) == 1
    ev = events[0]

    window_samples = int(round(10.0 * sr / 1000.0))
    assert ev.samples.size == window_samples

    pre_samples = int(round((ev.crossingTimeSec - ev.firstSampleTimeSec) * sr))
    post_samples = ev.samples.size - pre_samples

    expected_pre = window_samples // 3
    expected_post = window_samples - expected_pre

    assert pre_samples == expected_pre
    assert post_samples == expected_post

    expected_first = ev.crossingTimeSec - (expected_pre / sr)
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
    data[0, crossing_idx] = 0.3
    data[0, crossing_idx + 10] = 0.6
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    events = worker._detect_events(chunk)
    assert events == []


def test_worker_accepts_event_when_secondary_not_crossed() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker._channel_index = 0
    worker.configure_threshold(True, 0.25, secondary_enabled=True, secondary_value=0.5)

    sr = 20_000
    dt = 1.0 / sr
    data = np.zeros((1, 400), dtype=np.float32)
    crossing_idx = 200
    data[0, crossing_idx] = 0.4
    data[0, crossing_idx + 10] = 0.45
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    events = worker._detect_events(chunk)
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

    events = worker._detect_events(chunk)
    assert len(events) == 2
    first_event, second_event = events
    assert np.isnan(first_event.intervalSinceLastSec)
    expected_interval = (second_idx - first_idx) * dt
    assert second_event.intervalSinceLastSec == pytest.approx(expected_interval, rel=1e-6)
    assert second_event.properties.get("interval_sec") == pytest.approx(expected_interval, rel=1e-6)


def test_event_ids_remain_monotonic_across_workers(monkeypatch) -> None:
    monkeypatch.setattr(AnalysisWorker, "_global_event_id", 0)

    controller = _DummyController()
    worker_a = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker_b = AnalysisWorker(controller, "ch1", sample_rate=20_000)
    worker_a._channel_index = 0
    worker_b._channel_index = 0
    worker_a.configure_threshold(True, 0.25)
    worker_b.configure_threshold(True, 0.25)

    sr = 20_000
    dt = 1.0 / sr

    first = np.zeros((1, 400), dtype=np.float32)
    first[0, 200] = 0.5
    events_a = worker_a._detect_events(
        Chunk(
            samples=first,
            start_time=0.0,
            dt=dt,
            seq=0,
            channel_names=("ch0",),
            units="V",
            meta={"start_sample": 0},
        )
    )
    assert [event.id for event in events_a] == [1]

    second = np.zeros((1, 400), dtype=np.float32)
    second[0, 220] = 0.6
    events_b = worker_b._detect_events(
        Chunk(
            samples=second,
            start_time=0.1,
            dt=dt,
            seq=1,
            channel_names=("ch1",),
            units="V",
            meta={"start_sample": 400},
        )
    )
    assert [event.id for event in events_b] == [2]


def test_worker_manual_detection_uses_actual_channel_id() -> None:
    controller = _DummyController(
        active_channels=[
            ChannelInfo(id=3, name="ch0", units="V"),
            ChannelInfo(id=11, name="ch11", units="V"),
        ]
    )
    worker = AnalysisWorker(controller, "ch11", sample_rate=20_000)
    worker.configure_threshold(True, 0.25)

    sr = 20_000
    dt = 1.0 / sr
    data = np.zeros((2, 400), dtype=np.float32)
    data[1, 200] = 0.8
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0", "ch11"),
        units="V",
        meta={"start_sample": 0},
    )

    worker._forward_chunk(chunk)
    batch = _next_batch(worker)
    assert len(batch.events) == 1
    assert batch.events[0].channelId == 11


def test_worker_auto_detection_uses_actual_channel_id() -> None:
    controller = _DummyController(
        active_channels=[
            ChannelInfo(id=3, name="ch0", units="V"),
            ChannelInfo(id=11, name="ch11", units="V"),
        ]
    )
    worker = AnalysisWorker(controller, "ch11", sample_rate=20_000)
    worker.configure_threshold(enabled=True, value=0.5, auto_detect=True)

    sr = 20_000
    dt = 1.0 / sr
    data = np.zeros((2, 500), dtype=np.float32)
    data[1, 250] = -10.0
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0", "ch11"),
        units="V",
        meta={"start_sample": 0},
    )

    worker._forward_chunk(chunk)
    batch = _next_batch(worker)
    assert len(batch.events) == 1
    assert batch.events[0].channelId == 11


def test_worker_auto_detection_preserves_threshold_polarity() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker._channel_index = 0
    worker.configure_threshold(enabled=True, value=0.5, auto_detect=True)

    sr = 20_000
    dt = 1.0 / sr
    data = np.zeros((1, 600), dtype=np.float32)
    data[0, 150] = -10.0
    data[0, 350] = 10.0
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    events = worker._detect_events(chunk)

    assert len(events) == 2
    by_index = {event.crossingIndex: event for event in events}
    assert by_index[150].thresholdValue < 0.0
    assert by_index[350].thresholdValue > 0.0


def test_worker_allows_second_event_after_previous_window_end() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker._channel_index = 0
    worker.configure_threshold(True, 0.25)
    with worker._state_lock:
        worker._event_window_ms = 10.0

    sr = 20_000
    dt = 1.0 / sr
    data = np.zeros((1, 600), dtype=np.float32)
    data[0, 200] = 0.5
    data[0, 350] = 0.55
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=dt,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    events = worker._detect_events(chunk)
    assert [event.crossingIndex for event in events] == [200, 350]


def test_worker_manual_plateau_emits_single_event() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=1_000)
    worker._channel_index = 0
    worker.configure_threshold(True, 0.5)
    with worker._state_lock:
        worker._event_window_ms = 10.0

    data = np.zeros((1, 300), dtype=np.float32)
    data[0, 100:150] = 1.0
    chunk = Chunk(
        samples=data,
        start_time=0.0,
        dt=0.001,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={"start_sample": 0},
    )

    events = worker._detect_events(chunk)

    assert len(events) == 1
    assert events[0].crossingIndex == 100


def test_worker_manual_plateau_across_chunks_does_not_duplicate() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=1_000)
    worker._channel_index = 0
    worker.configure_threshold(True, 0.5)
    with worker._state_lock:
        worker._event_window_ms = 10.0

    first = np.zeros((1, 120), dtype=np.float32)
    first[0, 95:] = 1.0
    second = np.zeros((1, 120), dtype=np.float32)
    second[0, :25] = 1.0

    worker._detect_events(
        Chunk(
            samples=first,
            start_time=0.0,
            dt=0.001,
            seq=0,
            channel_names=("ch0",),
            units="V",
            meta={"start_sample": 0},
        )
    )
    events = worker._detect_events(
        Chunk(
            samples=second,
            start_time=0.120,
            dt=0.001,
            seq=1,
            channel_names=("ch0",),
            units="V",
            meta={"start_sample": 120},
        )
    )

    assert events == []


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
    wave += 0.3
    wave += 0.05 * (t - t.mean())
    center = len(wave) // 2
    assert _peak_frequency_sinc(wave, sr, center_index=center) == pytest.approx(freq, rel=0.08)


def test_peak_frequency_sinc_focuses_on_localized_event() -> None:
    sr = 20_000
    samples = int(sr * 0.02)
    wave = np.zeros(samples, dtype=np.float64)
    burst_len = int(sr * 0.010)
    start = samples // 2 - burst_len // 2
    t = np.arange(burst_len) / sr
    freq = 220.0
    burst = np.sin(2 * np.pi * freq * t) * np.hanning(burst_len)
    wave[start : start + burst_len] = burst
    center = start + burst_len // 2
    assert _peak_frequency_sinc(wave, sr, center_index=center) == pytest.approx(freq, rel=0.1)


def test_worker_updates_auto_detector_window_on_settings_change() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)

    worker.configure_threshold(enabled=True, value=0.5, auto_detect=True)

    assert worker._auto_detector is not None
    assert worker._auto_detector._window_ms == 10.0

    controller.analysis_settings_store.update(event_window_ms=20.0)

    assert worker._event_window_ms == 20.0
    assert worker._auto_detector._window_ms == 20.0


def test_worker_update_sample_rate_resets_auto_detector_state() -> None:
    controller = _DummyController()
    worker = AnalysisWorker(controller, "ch0", sample_rate=20_000)
    worker.configure_threshold(enabled=True, value=0.5, auto_detect=True)

    assert worker._auto_detector is not None
    worker._last_crossing_time_sec = 1.23
    worker._last_window_end_sample = 456
    worker._auto_detector._residue = np.ones((1, 4), dtype=np.float32)

    worker.update_sample_rate(10_000.0)

    assert worker.sample_rate == pytest.approx(10_000.0)
    assert worker._auto_detector._sample_rate == pytest.approx(10_000.0)
    assert worker._auto_detector._residue is None
    assert worker._last_crossing_time_sec is None
    assert worker._last_window_end_sample == -10**12
