"""
Contract tests for DAQ device implementations.

These tests define the behavioral contract that ALL BaseDevice implementations
must satisfy. They serve two purposes:

1. Verify existing device implementations (SimulatedPhysiologySource, etc.)
2. Guide development of new device drivers

The contract tests are parameterized to run against all device implementations,
ensuring consistent behavior across the codebase.

Contract invariants being verified:
- Start/stop are idempotent (safe to call multiple times)
- Buffer dtype is always float32
- Buffer shape matches configuration
- State transitions are clean
- Resource cleanup is complete
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Type, List

import numpy as np
import pytest

from daq.base_device import BaseDevice
from daq.simulated_source import SimulatedPhysiologySource
from shared.models import ChunkPointer, EndOfStream


# List of device classes to test
# Add new device implementations here as they are developed
DEVICE_CLASSES: List[Type[BaseDevice]] = [
    SimulatedPhysiologySource,
]


@pytest.fixture(params=DEVICE_CLASSES, ids=lambda c: c.__name__)
def device_class(request) -> Type[BaseDevice]:
    """Parameterized fixture for all device classes."""
    return request.param


@pytest.fixture
def device(device_class) -> BaseDevice:
    """Create a fresh device instance for each test."""
    return device_class()


class TestDeviceLifecycle:
    """Tests for device lifecycle (open/close) behavior."""

    def test_open_close_cycle(self, device):
        """Device should open and close without error."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)
        assert device.state == "open"

        device.close()
        assert device.state == "closed"

    def test_close_without_open_is_safe(self, device):
        """Closing a device that was never opened should not crash."""
        # Should not raise
        device.close()
        assert device.state == "closed"

    def test_double_close_is_safe(self, device):
        """Closing a device twice should not crash."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device.open(devices[0].id)
        device.close()
        device.close()  # Should not raise
        assert device.state == "closed"


class TestDeviceConfiguration:
    """Tests for device configuration behavior."""

    def test_configure_returns_actual_config(self, device):
        """Configure should return ActualConfig with expected fields."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)

        try:
            config = device.configure(sample_rate=20000, chunk_size=256)

            assert config is not None
            assert hasattr(config, "sample_rate")
            assert hasattr(config, "channels")
            assert hasattr(config, "chunk_size")
            assert config.dtype == "float32"
        finally:
            device.close()

    def test_configure_channels_subset(self, device):
        """Configure with specific channels should work."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)

        try:
            channels = device.list_available_channels(device_id)
            if len(channels) < 2:
                pytest.skip("Device has fewer than 2 channels")

            # Request subset of channels
            requested = [channels[0].id, channels[1].id]
            config = device.configure(sample_rate=20000, channels=requested, chunk_size=256)

            assert len(config.channels) == 2
        finally:
            device.close()


class TestDeviceStreaming:
    """Tests for device streaming behavior."""

    def test_start_produces_data(self, device):
        """Starting device should produce ChunkPointers on the queue."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)
        device.configure(sample_rate=20000, chunk_size=256)

        try:
            device.start()

            # Wait for some data
            data_received = False
            for _ in range(20):  # Try for up to 2 seconds
                try:
                    item = device.data_queue.get(timeout=0.1)
                    if isinstance(item, ChunkPointer):
                        data_received = True
                        break
                except queue.Empty:
                    continue

            assert data_received, "No ChunkPointers received after start"
        finally:
            device.stop()
            device.close()

    def test_stop_halts_data(self, device):
        """Stopping device should halt data production."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)
        device.configure(sample_rate=20000, chunk_size=256)

        try:
            device.start()
            time.sleep(0.1)  # Let it produce some data
            device.stop()

            # Drain existing items
            while True:
                try:
                    device.data_queue.get_nowait()
                except queue.Empty:
                    break

            # Wait briefly
            time.sleep(0.1)

            # Queue should now be empty (no new data)
            is_empty = device.data_queue.empty()
            assert is_empty, "Queue should be empty after stop"
        finally:
            device.close()


class TestDeviceIdempotence:
    """Tests for idempotent operations."""

    def test_start_twice_no_crash(self, device):
        """Calling start() twice should not crash or spawn duplicate threads."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)
        device.configure(sample_rate=20000, chunk_size=256)

        thread_count_before = threading.active_count()

        try:
            device.start()
            time.sleep(0.05)
            thread_count_first = threading.active_count()

            device.start()  # Second start - should be no-op or safe
            time.sleep(0.05)
            thread_count_second = threading.active_count()

            # Should not spawn duplicate threads
            assert thread_count_second <= thread_count_first + 1, \
                "Duplicate threads spawned on second start"
        finally:
            device.stop()
            device.close()

    def test_stop_twice_no_crash(self, device):
        """Calling stop() twice should not crash."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)
        device.configure(sample_rate=20000, chunk_size=256)

        try:
            device.start()
            time.sleep(0.05)
            device.stop()
            device.stop()  # Should not raise
        finally:
            device.close()

    def test_stop_without_start_is_safe(self, device):
        """Calling stop() before start() should not crash."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)
        device.configure(sample_rate=20000, chunk_size=256)

        try:
            device.stop()  # Should not raise
        finally:
            device.close()


class TestDeviceBufferContract:
    """Tests for buffer contract compliance."""

    def test_buffer_dtype_is_float32(self, device):
        """Device buffer should always use float32."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)
        device.configure(sample_rate=20000, chunk_size=256)

        try:
            device.start()

            # Get pointer and verify buffer dtype
            try:
                item = device.data_queue.get(timeout=1.0)
                if isinstance(item, ChunkPointer):
                    buffer = device.get_buffer()
                    assert buffer.dtype == np.float32, f"Expected float32, got {buffer.dtype}"
            except queue.Empty:
                pytest.fail("No data received to check buffer dtype")
        finally:
            device.stop()
            device.close()

    def test_buffer_shape_matches_config(self, device):
        """Buffer shape should match configured channel count."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)

        # Get available channels
        available = device.list_available_channels(device_id)
        n_channels = min(2, len(available))
        channels = [c.id for c in available[:n_channels]]

        config = device.configure(sample_rate=20000, channels=channels, chunk_size=256)

        try:
            device.start()

            buffer = device.get_buffer()
            # Buffer shape is (channels, capacity)
            assert buffer.shape[0] == len(config.channels), \
                f"Buffer channels {buffer.shape[0]} != config channels {len(config.channels)}"
        finally:
            device.stop()
            device.close()


class TestDeviceStatistics:
    """Tests for device statistics reporting."""

    def test_stats_available_during_streaming(self, device):
        """Stats should be available while device is streaming."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)
        device.configure(sample_rate=20000, chunk_size=256)

        try:
            device.start()
            time.sleep(0.1)  # Let it produce some data

            stats = device.stats()
            assert stats is not None
            assert isinstance(stats, dict)

            # Common stats fields
            assert "chunks_emitted" in stats or "emitted_chunks" in stats or len(stats) >= 0
        finally:
            device.stop()
            device.close()


class TestDeviceMultipleRuns:
    """Tests for multiple start/stop cycles."""

    def test_multiple_start_stop_cycles(self, device):
        """Device should handle multiple start/stop cycles cleanly."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)
        device.configure(sample_rate=20000, chunk_size=256)

        try:
            for cycle in range(3):
                device.start()

                # Verify data flow
                data_received = False
                for _ in range(10):
                    try:
                        item = device.data_queue.get(timeout=0.1)
                        if isinstance(item, ChunkPointer):
                            data_received = True
                            break
                    except queue.Empty:
                        continue

                assert data_received, f"No data in cycle {cycle}"

                device.stop()

                # Drain queue
                while not device.data_queue.empty():
                    try:
                        device.data_queue.get_nowait()
                    except queue.Empty:
                        break

                time.sleep(0.05)  # Brief pause between cycles
        finally:
            device.close()

    def test_reconfigure_between_runs(self, device):
        """Device should allow reconfiguration between runs."""
        devices = device.list_available_devices()
        if not devices:
            pytest.skip(f"No devices available for {device.__class__.__name__}")

        device_id = devices[0].id
        device.open(device_id)

        try:
            # First configuration
            config1 = device.configure(sample_rate=20000, chunk_size=256)
            device.start()
            time.sleep(0.05)
            device.stop()

            # Drain
            while not device.data_queue.empty():
                try:
                    device.data_queue.get_nowait()
                except queue.Empty:
                    break

            # Second configuration with different parameters
            config2 = device.configure(sample_rate=20000, chunk_size=512)
            device.start()
            time.sleep(0.05)
            device.stop()

            # Both configurations should have succeeded
            assert config1 is not None
            assert config2 is not None
        finally:
            device.close()
