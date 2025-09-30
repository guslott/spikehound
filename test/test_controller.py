import queue
import time

from core import FilterSettings, PipelineController
from daq.simulated_source import SimulatedPhysiologySource


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
        analysis_queue_size=16,
        audio_queue_size=16,
        logging_queue_size=64,
        dispatcher_poll_timeout=0.01,
    )

    for sample_rate in (1000, 4000):
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
        analysis_queue_size=1,
        audio_queue_size=1,
        logging_queue_size=8,
        dispatcher_poll_timeout=0.01,
    )

    controller.switch_source(
        SimulatedPhysiologySource,
        configure_kwargs={
            "sample_rate": 2000,
            "chunk_size": 32,
            "num_units": 3,
        },
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
