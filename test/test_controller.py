import queue
import time
from types import SimpleNamespace

import pytest
from core import FilterSettings, PipelineController
from daq.simulated_source import SimulatedPhysiologySource
from shared.models import TriggerConfig


def _drain(queue_obj):
    while True:
        try:
            queue_obj.get_nowait()
        except queue.Empty:
            break


def test_pipeline_controller_start_stop_cycles():
    controller = PipelineController(
        filter_settings=FilterSettings(),
        visualization_queue_size=64,
        audio_queue_size=16,
        logging_queue_size=64,
        dispatcher_poll_timeout=0.01,
    )

    # SimulatedPhysiologySource supports both 10kHz and 20kHz.
    for sample_rate in (10000, 20000):
        actual = controller.switch_source(
            SimulatedPhysiologySource,
            configure_kwargs={
                "sample_rate": sample_rate,
                "chunk_size": 64,
                "num_units": 2,
            },
        )
        assert actual.sample_rate == sample_rate

        for _ in range(50):
            controller.start()
            time.sleep(0.01)
            controller.stop()
            _drain(controller.visualization_queue)

        assert not controller.running

    controller.shutdown()


def test_pipeline_controller_backpressure_tracks_evictions():
    controller = PipelineController(
        filter_settings=FilterSettings(),
        visualization_queue_size=1,
        audio_queue_size=1,
        logging_queue_size=8,
        dispatcher_poll_timeout=0.01,
    )

    controller.switch_source(
        SimulatedPhysiologySource,
        configure_kwargs={
            "sample_rate": 20000,
            "chunk_size": 32,
            "num_units": 3,
        },
    )
    controller.update_trigger_config(
        TriggerConfig(
            channel_index=0,
            threshold=0.0,
            hysteresis=0.0,
            pretrigger_frac=0.0,
            window_sec=1.0,
            mode="repeated",
        )
    )

    controller.start()
    time.sleep(0.2)
    controller.stop()

    stats = controller.dispatcher_stats()
    forwarded = stats.get("forwarded", {})
    evicted = stats.get("evicted", {})
    assert forwarded.get("visualization", 0) > 0
    assert evicted.get("visualization", 0) >= 1

    controller.shutdown()


class _FailingSource:
    def __init__(self, *, stop_raises: bool = False, close_raises: bool = False) -> None:
        self.running = True
        self.stop_raises = stop_raises
        self.close_raises = close_raises
        self.data_queue = queue.Queue()
        self.stop_calls = 0
        self.close_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False
        if self.stop_raises:
            raise RuntimeError("source stop failed")

    def close(self) -> None:
        self.close_calls += 1
        if self.close_raises:
            raise RuntimeError("source close failed")


class _TrackingDispatcher:
    def __init__(self, *, stop_raises: bool = False) -> None:
        self.stop_raises = stop_raises
        self.calls: list[str] = []

    def clear_active_channels(self) -> None:
        self.calls.append("clear_active_channels")

    def reset_buffers(self) -> None:
        self.calls.append("reset_buffers")

    def emit_empty_tick(self) -> None:
        self.calls.append("emit_empty_tick")

    def stop(self) -> None:
        self.calls.append("stop")
        if self.stop_raises:
            raise RuntimeError("dispatcher stop failed")


def _raiser(message: str):
    def _raise() -> None:
        raise RuntimeError(message)

    return _raise


def test_stop_propagates_shutdown_failure_after_finishing_cleanup():
    controller = PipelineController(filter_settings=FilterSettings())
    source = _FailingSource(stop_raises=True)
    dispatcher = _TrackingDispatcher()

    controller._source = source
    controller._dispatcher = dispatcher
    controller._streaming = True
    controller._running = True

    with pytest.raises(RuntimeError, match="Streaming shutdown failed"):
        controller.stop(join=True)

    assert controller.running is False
    assert controller._streaming is False
    assert dispatcher.calls == [
        "clear_active_channels",
        "reset_buffers",
        "emit_empty_tick",
        "stop",
    ]


def test_shutdown_aggregates_teardown_failures_and_clears_state():
    controller = PipelineController(filter_settings=FilterSettings())
    source = _FailingSource(stop_raises=True, close_raises=True)
    dispatcher = _TrackingDispatcher(stop_raises=True)
    wav_logger = SimpleNamespace(stop=_raiser("wav stop failed"))
    audio_manager = SimpleNamespace(stop=_raiser("audio stop failed"))

    controller._source = source
    controller._dispatcher = dispatcher
    controller._wav_logger = wav_logger
    controller._audio_manager = audio_manager
    controller._streaming = True
    controller._running = True
    controller._actual_config = SimpleNamespace(sample_rate=20_000)
    controller._device_id = "device-1"
    controller._configure_kwargs = {"sample_rate": 20_000}

    with pytest.raises(RuntimeError, match="Pipeline shutdown failed"):
        controller.shutdown()

    assert controller._wav_logger is None
    assert controller._source is None
    assert controller._dispatcher is None
    assert controller._actual_config is None
    assert controller._device_id is None
    assert controller._configure_kwargs == {}
    assert controller.running is False
