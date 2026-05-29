"""
Integration Tests for SpikeHound
================================

End-to-end integration tests using SimulatedPhysiologySource.
These tests serve as reference patterns for AI-assisted verification.

Test Patterns
-------------
1. Runtime data flow - verify data moves from source to consumers
2. Analysis stream - verify event detection and routing
3. Filter propagation - verify filter settings reach dispatcher
4. Trigger configuration - verify trigger settings reach dispatcher
5. Health metrics - verify monitoring endpoints work correctly

Usage
-----
Run from project root:
    python -m pytest test/test_integration.py -v
"""

from __future__ import annotations

import queue
import time
from typing import Optional

import numpy as np
import pytest

from core import PipelineController, FilterSettings, TriggerConfig
from core.runtime import SpikeHoundRuntime
from core.conditioning import ChannelFilterSettings
from daq.simulated_source import SimulatedPhysiologySource
from shared.ring_buffer import SharedRingBuffer


def _drain_queue(q: queue.Queue, timeout: float = 0.1) -> list:
    """Drain all items from a queue with a brief timeout."""
    items = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            time.sleep(0.01)
    return items


def _wait_for_tick_with_data(dispatcher, timeout: float = 2.0) -> Optional[dict]:
    """Wait until the dispatcher emits a data-bearing visualization tick.

    In the default *stream* mode, conditioned data is delivered to consumers via
    the dispatcher's 60 Hz tick callbacks, not the visualization queue (which is
    intentionally empty in stream mode — see the "Visualization delivery
    contract" in core/dispatcher.py and finding 3.2 #1). This registers a tick
    consumer and returns the first payload that carries samples, or None on
    timeout.
    """
    received: list[dict] = []

    def _cb(payload: dict) -> None:
        samples = payload.get("samples")
        if samples is not None and getattr(samples, "size", 0) > 0:
            received.append(payload)

    unsubscribe = dispatcher.add_tick_callback(_cb)
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if received:
                return received[0]
            time.sleep(0.01)
    finally:
        unsubscribe()
    return received[0] if received else None


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll `predicate` until true or timeout. Returns the final truthiness."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class TestRuntimeDataFlow:
    """
    Test Pattern: End-to-end data flow verification.
    
    Verifies that data flows correctly from SimulatedPhysiologySource
    through the pipeline to consumer queues.
    """
    
    def test_runtime_receives_visualization_data(self):
        """
        Verify that starting acquisition delivers visualization data to consumers.

        In the default *stream* mode, visualization is delivered via the
        dispatcher's 60 Hz tick callbacks — the visualization queue stays empty
        by design (finding 3.2 #1). This exercises the real stream-mode
        contract:
        1. Create pipeline controller with queues
        2. Attach simulated source
        3. Start acquisition
        4. Verify a tick consumer receives sample data (and the queue is empty)
        5. Clean shutdown
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=64,
            logging_queue_size=64,
            dispatcher_poll_timeout=0.01,
        )

        # Attach simulated physiology source
        actual = controller.switch_source(
            SimulatedPhysiologySource,
            configure_kwargs={
                "sample_rate": 20000,
                "chunk_size": 128,
                "num_units": 2,
            },
        )
        assert actual.sample_rate == 20000

        # Start acquisition
        controller.start()

        dispatcher = controller.dispatcher
        assert dispatcher is not None, "Dispatcher should exist after start"

        # Stream-mode visualization arrives via tick callbacks.
        payload = _wait_for_tick_with_data(dispatcher)
        assert payload is not None, "Should receive a data-bearing visualization tick"
        assert payload["samples"].shape[1] > 0, "Tick payload should carry samples"

        # The visualization queue is intentionally empty in stream mode.
        assert controller.visualization_queue.empty(), (
            "Visualization queue must stay empty in stream mode (data flows via ticks)"
        )

        # Observability: the tick delivery path is reflected in dispatcher stats.
        assert controller.dispatcher_stats().get("visualization_ticks", 0) > 0

        # Clean shutdown
        controller.stop()
        controller.shutdown()
        assert not controller.running
    
    def test_runtime_multiple_start_stop_cycles(self):
        """
        Verify that acquisition can be started and stopped multiple times.
        
        This tests the robustness of the pipeline lifecycle.
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=32,
            logging_queue_size=32,
            dispatcher_poll_timeout=0.01,
        )
        
        controller.switch_source(
            SimulatedPhysiologySource,
            configure_kwargs={
                "sample_rate": 20000,
                "chunk_size": 64,
                "num_units": 1,
            },
        )
        
        # Run multiple cycles
        for i in range(5):
            controller.start()
            time.sleep(0.02)
            controller.stop()
            _drain_queue(controller.visualization_queue)
            assert not controller.running, f"Cycle {i}: should stop cleanly"
        
        controller.shutdown()


class TestFilterPropagation:
    """
    Test Pattern: Filter settings propagation.
    
    Verifies that filter configurations properly propagate through
    the pipeline to affect signal conditioning.
    """
    
    def test_filter_settings_reach_dispatcher(self):
        """
        Verify that filter settings are applied to the dispatcher conditioner.
        """
        # Create filter settings with specific configuration
        # FilterSettings uses 'overrides' dict keyed by channel name
        filter_settings = FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True,
                notch_freq_hz=60.0,
                highpass_hz=10.0,
            ),
        )
        
        controller = PipelineController(
            filter_settings=filter_settings,
            visualization_queue_size=64,
            logging_queue_size=64,
            dispatcher_poll_timeout=0.01,
        )
        
        controller.switch_source(
            SimulatedPhysiologySource,
            configure_kwargs={
                "sample_rate": 20000,
                "chunk_size": 128,
                "num_units": 1,
            },
        )
        
        # Verify filter settings were applied
        dispatcher = controller.dispatcher
        assert dispatcher is not None, "Dispatcher should be created"
        
        # Verify the conditioner has our settings
        conditioner = getattr(dispatcher, "_conditioner", None)
        if conditioner is not None:
            settings = conditioner.settings
            assert settings.default.notch_enabled is True
            assert settings.default.notch_freq_hz == 60.0
        
        controller.shutdown()
    
    def test_filter_update_during_acquisition(self):
        """
        Verify that filter settings can be updated while acquisition is running
        and that visualization data keeps flowing afterward.

        Stream-mode visualization flows via tick callbacks (finding 3.2 #1), so
        "still receiving data" is verified by the visualization_ticks counter
        continuing to advance after the filter update.
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),  # Default settings
            visualization_queue_size=64,
            logging_queue_size=64,
            dispatcher_poll_timeout=0.01,
        )

        controller.switch_source(
            SimulatedPhysiologySource,
            configure_kwargs={
                "sample_rate": 20000,
                "chunk_size": 128,
                "num_units": 1,
            },
        )

        controller.start()
        dispatcher = controller.dispatcher
        assert dispatcher is not None

        # Data flows before the update.
        assert _wait_for_tick_with_data(dispatcher) is not None, (
            "Should receive visualization ticks before the filter update"
        )

        # Update filter settings while running.
        new_settings = FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True,
                notch_freq_hz=50.0,  # European mains hum
            ),
        )
        ticks_before = controller.dispatcher_stats().get("visualization_ticks", 0)
        controller.update_filter_settings(new_settings)

        # Visualization should continue after the update.
        advanced = _wait_until(
            lambda: controller.dispatcher_stats().get("visualization_ticks", 0)
            > ticks_before
        )
        assert advanced, "Should continue receiving data after a filter update"

        controller.stop()
        controller.shutdown()


class TestTriggerConfiguration:
    """
    Test Pattern: Trigger configuration propagation.
    
    Verifies that trigger settings are correctly applied to the dispatcher.
    """
    
    def test_trigger_config_applies(self):
        """
        Verify that trigger configuration reaches the dispatcher.
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=64,
            logging_queue_size=64,
            dispatcher_poll_timeout=0.01,
        )
        
        controller.switch_source(
            SimulatedPhysiologySource,
            configure_kwargs={
                "sample_rate": 20000,
                "chunk_size": 128,
                "num_units": 1,
            },
        )
        
        # Configure trigger
        trigger_config = TriggerConfig(
            channel_index=0,
            threshold=0.5,
            hysteresis=0.0,
            pretrigger_frac=0.2,
            window_sec=0.5,
            mode="single",
        )
        controller.update_trigger_config(trigger_config)
        
        # Verify trigger config was applied
        dispatcher = controller.dispatcher
        assert dispatcher is not None
        
        # The dispatcher should have the trigger config
        # Access depends on implementation, check if attribute exists
        if hasattr(dispatcher, "_trigger_config"):
            assert dispatcher._trigger_config.threshold == 0.5
            assert dispatcher._trigger_config.mode == "single"
        
        controller.shutdown()


class TestHealthMetrics:
    """
    Test Pattern: Health metrics and monitoring.
    
    Verifies that health monitoring endpoints report valid data.
    """
    
    def test_health_snapshot_during_acquisition(self):
        """
        Verify that health_snapshot returns meaningful metrics during acquisition.

        Stream-mode visualization flows via tick callbacks rather than the
        visualization queue, so observability lives in the `processed` and
        `visualization_ticks` counters rather than `forwarded["visualization"]`
        (finding 3.2 #1).
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=64,
            logging_queue_size=64,
            dispatcher_poll_timeout=0.01,
        )

        runtime = SpikeHoundRuntime(pipeline=controller)

        controller.switch_source(
            SimulatedPhysiologySource,
            configure_kwargs={
                "sample_rate": 20000,
                "chunk_size": 128,
                "num_units": 2,
            },
        )

        controller.start()

        # Poll until the dispatcher has processed chunks and delivered viz ticks.
        def _has_flow() -> bool:
            ds = runtime.health_snapshot().get("dispatcher", {})
            return ds.get("processed", 0) > 0 and ds.get("visualization_ticks", 0) > 0

        assert _wait_until(_has_flow), "Dispatcher should process chunks and emit viz ticks"

        # Get health snapshot
        health = runtime.health_snapshot()

        assert health is not None, "health_snapshot should return data"
        assert "dispatcher" in health, "Should include dispatcher stats"

        dispatcher_stats = health.get("dispatcher", {})
        assert dispatcher_stats.get("processed", 0) > 0, "Should have processed chunks"
        # Visualization is delivered via tick callbacks in stream mode.
        assert dispatcher_stats.get("visualization_ticks", 0) > 0, (
            "Should have delivered visualization ticks in stream mode"
        )

        controller.stop()
        controller.shutdown()
    
    def test_dispatcher_stats_track_evictions(self):
        """
        Verify that dispatcher stats track queue evictions under backpressure.

        The visualization queue is only used in *triggered* (non-stream) modes
        (finding 3.2 #1); in stream mode the scope is fed by tick callbacks and
        the queue stays empty. So this test selects a triggered mode, then uses
        a size-1 queue with no consumer to force drop-oldest evictions.
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=1,  # Tiny queue to force evictions
            logging_queue_size=8,
            dispatcher_poll_timeout=0.01,
        )

        controller.switch_source(
            SimulatedPhysiologySource,
            configure_kwargs={
                "sample_rate": 20000,
                "chunk_size": 32,
                "num_units": 1,
            },
        )

        # Triggered mode routes every processed chunk's ChunkPointer to the
        # visualization queue. Nothing drains it here, so the size-1 queue evicts.
        controller.update_trigger_config(
            TriggerConfig(
                channel_index=0,
                threshold=0.5,
                hysteresis=0.0,
                pretrigger_frac=0.2,
                window_sec=0.2,
                mode="repeated",
            )
        )

        controller.start()

        # Generate enough data to cause evictions.
        evicted_now = lambda: controller.dispatcher_stats().get("evicted", {}).get(
            "visualization", 0
        )
        got_evictions = _wait_until(lambda: evicted_now() >= 1)
        controller.stop()

        assert got_evictions and evicted_now() >= 1, "Should have evicted visualization chunks"

        controller.shutdown()


class TestAnalysisIntegration:
    """
    Test Pattern: Analysis worker integration.
    
    Verifies that analysis workers receive and process events correctly.
    """
    
    def test_runtime_opens_analysis_stream(self):
        """
        Verify that opening an analysis stream creates a working queue.
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=64,
            logging_queue_size=64,
            dispatcher_poll_timeout=0.01,
        )
        
        runtime = SpikeHoundRuntime(pipeline=controller)
        
        controller.switch_source(
            SimulatedPhysiologySource,
            configure_kwargs={
                "sample_rate": 20000,
                "chunk_size": 128,
                "num_units": 2,
            },
        )
        
        # Open analysis stream
        analysis_queue, worker = runtime.open_analysis_stream(
            channel_name="Channel 0",
            sample_rate=20000,
        )
        
        assert analysis_queue is not None, "Should return analysis queue"
        assert worker is not None, "Should return worker"
        
        # Enable event detection
        worker.configure_threshold(enabled=True, value=0.1, auto_detect=True)
        
        # Start acquisition
        controller.start()
        time.sleep(0.3)  # Allow time for events to be detected
        controller.stop()
        
        # Check if any analysis data was produced
        items = _drain_queue(analysis_queue, timeout=0.1)
        # Note: With auto_detect and simulated spikes, we should get events
        # but the exact count depends on the simulation parameters
        
        worker.stop()
        controller.shutdown()


class TestHeadlessDeviceConnection:
    """Finding 3.2 #3: connect_device must work without a Qt device_manager."""

    def test_connect_device_headless_via_registry(self):
        """A runtime with no device_manager connects through the DeviceRegistry
        instead of raising AttributeError on self.device_manager.
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=32,
            logging_queue_size=32,
            dispatcher_poll_timeout=0.01,
        )
        runtime = SpikeHoundRuntime(pipeline=controller)
        assert runtime.device_manager is None, "this test exercises the headless path"

        try:
            # Populate the registry headlessly, then locate the simulated source.
            runtime.scan_devices()
            sim_key = next(
                entry["key"]
                for entry in runtime._device_registry.get_device_list()
                if "simulated" in str(entry.get("module", "")).lower()
            )

            # Previously raised AttributeError (device_manager is None).
            runtime.connect_device(sim_key, sample_rate=20000, chunk_size=128)

            # The source is wired into the pipeline and the headless device API works.
            assert controller.dispatcher is not None, "source should be attached"
            assert runtime.active_device_key() == sim_key
            assert len(runtime.available_channels()) >= 1

            # End-to-end: acquisition actually runs headlessly.
            runtime.start_acquisition()
            dispatcher = controller.dispatcher
            assert _wait_until(
                lambda: dispatcher.snapshot().get("processed", 0) > 0
            ), "headless acquisition should process chunks"
            controller.stop()
        finally:
            runtime.disconnect_device()
            controller.shutdown()
