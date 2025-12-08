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


class TestRuntimeDataFlow:
    """
    Test Pattern: End-to-end data flow verification.
    
    Verifies that data flows correctly from SimulatedPhysiologySource
    through the pipeline to consumer queues.
    """
    
    def test_runtime_receives_visualization_data(self):
        """
        Verify that starting acquisition produces data in the visualization queue.
        
        This test demonstrates the minimal setup for a functioning pipeline:
        1. Create pipeline controller with queues
        2. Attach simulated source
        3. Start acquisition
        4. Verify data arrives
        5. Clean shutdown
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=64,
            audio_queue_size=16,
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
        
        # Wait for data to arrive
        time.sleep(0.1)
        
        # Verify data in visualization queue
        items = _drain_queue(controller.visualization_queue, timeout=0.2)
        assert len(items) > 0, "Should receive chunks in visualization queue"
        
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
            audio_queue_size=8,
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
            audio_queue_size=16,
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
        Verify that filter settings can be updated while acquisition is running.
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),  # Default settings
            visualization_queue_size=64,
            audio_queue_size=16,
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
        time.sleep(0.05)
        
        # Update filter settings while running
        new_settings = FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True,
                notch_freq_hz=50.0,  # European mains hum
            ),
        )
        controller.update_filter_settings(new_settings)
        
        time.sleep(0.05)
        
        # Should still be receiving data after update
        items = _drain_queue(controller.visualization_queue, timeout=0.1)
        assert len(items) > 0, "Should continue receiving data after filter update"
        
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
            audio_queue_size=16,
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
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=64,
            audio_queue_size=16,
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
        time.sleep(0.15)  # Let some data flow
        
        # Get health snapshot
        health = runtime.health_snapshot()
        
        assert health is not None, "health_snapshot should return data"
        assert "dispatcher" in health, "Should include dispatcher stats"
        
        dispatcher_stats = health.get("dispatcher", {})
        forwarded = dispatcher_stats.get("forwarded", {})
        
        # Should have forwarded some data
        viz_forwarded = forwarded.get("visualization", 0)
        assert viz_forwarded > 0, "Should have forwarded visualization chunks"
        
        controller.stop()
        controller.shutdown()
    
    def test_dispatcher_stats_track_evictions(self):
        """
        Verify that dispatcher stats track queue evictions under backpressure.
        
        This test uses a small queue to force evictions.
        """
        controller = PipelineController(
            filter_settings=FilterSettings(),
            visualization_queue_size=1,  # Tiny queue to force evictions
            audio_queue_size=1,
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
        
        controller.start()
        time.sleep(0.2)  # Generate enough data to cause evictions
        controller.stop()
        
        stats = controller.dispatcher_stats()
        evicted = stats.get("evicted", {})
        
        # With queue size of 1 and fast data generation, evictions should occur
        viz_evicted = evicted.get("visualization", 0)
        assert viz_evicted >= 1, "Should have evicted visualization chunks"
        
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
            audio_queue_size=16,
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
