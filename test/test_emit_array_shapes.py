# test/test_emit_array_shapes.py
"""Tests for BaseDevice.emit_array shape handling."""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from daq.simulated_source import SimulatedPhysiologySource
from shared.models import ChunkPointer


class TestEmitArrayShapes:
    """Tests for the strict emit_array frames-by-channels contract."""

    @pytest.fixture
    def configured_device(self):
        """Create a configured device ready for emit_array testing."""
        device = SimulatedPhysiologySource(queue_maxsize=64)
        devices = device.list_available_devices()
        device.open(devices[0].id)
        # Configure with 2 channels
        device.configure(sample_rate=20000, channels=[0, 1], chunk_size=256)
        yield device
        device.close()

    def test_frames_channels_shape(self, configured_device):
        """emit_array should handle (frames, channels) shape correctly."""
        device = configured_device
        
        # Create data with shape (frames=100, channels=2)
        frames, channels = 100, 2
        data = np.random.randn(frames, channels).astype(np.float32)
        
        # Should not raise
        pointer = device.emit_array(data)
        
        assert isinstance(pointer, ChunkPointer)
        assert pointer.length == frames

    def test_channels_frames_shape_raises(self, configured_device):
        """emit_array should reject channel-major input."""
        device = configured_device
        
        # Create data with shape (channels=2, frames=100)
        frames, channels = 100, 2
        data = np.random.randn(channels, frames).astype(np.float32)
        
        with pytest.raises(ValueError, match="does not match expected"):
            device.emit_array(data)

    def test_buffer_content_matches_frames_channels_input(self, configured_device):
        """emit_array should write frames-by-channels input as channel-major data."""
        device = configured_device
        
        # Reference data in (frames, channels) format
        frames, channels = 50, 2
        original = np.array([
            [1.0, 2.0],  # frame 0
            [3.0, 4.0],  # frame 1
            [5.0, 6.0],  # frame 2
        ], dtype=np.float32)
        
        device._reset_counters()
        ptr1 = device.emit_array(original.copy())
        buffer1 = device.ring_buffer.read(ptr1.start_index, ptr1.length)

        expected = original.T
        np.testing.assert_array_almost_equal(buffer1, expected)
        assert buffer1.shape == (channels, 3)

    def test_implicit_render_time_tracks_first_sample_origin(self, configured_device):
        """emit_array should derive chunk time from run origin and start_sample."""
        device = configured_device
        frames, channels = 8, 2
        data = np.ones((frames, channels), dtype=np.float32)
        device._reset_counters()
        device._run_start_mono = 10.0

        pointer_a = device.emit_array(data)
        pointer_b = device.emit_array(data)

        dt = frames / device.config.sample_rate
        assert pointer_a.render_time == pytest.approx(10.0)
        assert pointer_b.render_time == pytest.approx(10.0 + dt)

    def test_mismatched_dimensions_error(self, configured_device):
        """emit_array should raise on completely mismatched dimensions."""
        device = configured_device
        
        # Neither dimension matches expected 2 channels
        data = np.random.randn(5, 7).astype(np.float32)
        
        with pytest.raises(ValueError, match="does not match expected"):
            device.emit_array(data)

    def test_square_array_is_accepted_only_as_frames_channels(self, configured_device):
        """Square arrays are accepted because they still satisfy (frames, channels)."""
        # Reconfigure with equal frames and channels
        device = configured_device
        device.stop()
        device.close()
        
        # Create new device with 3 channels
        device = SimulatedPhysiologySource(queue_maxsize=64)
        devices = device.list_available_devices()
        device.open(devices[0].id)
        device.configure(sample_rate=20000, channels=[0, 1, 2], chunk_size=256)
        
        try:
            # Square array: 3x3
            data = np.random.randn(3, 3).astype(np.float32)
            
            pointer = device.emit_array(data)
            assert pointer.length == 3
        finally:
            device.close()

    def test_failing_monitor_bridge_is_disabled(self, configured_device):
        """A failing monitor bridge should be detached after the first error."""

        class _FailingBridge:
            def __init__(self) -> None:
                self.calls = 0

            def on_chunk(self, raw) -> None:
                self.calls += 1
                raise RuntimeError("bridge boom")

        device = configured_device
        bridge = _FailingBridge()
        device.register_monitor_bridge(bridge)

        data = np.random.randn(32, 2).astype(np.float32)
        device.emit_array(data)
        assert bridge.calls == 1
        assert device._monitor_bridge is None

        device.emit_array(data)
        assert bridge.calls == 1


class TestEmitArrayLockContention:
    """Finding 4.1: emit_array must not hold _state_lock across the blocking put."""

    def test_blocking_put_does_not_freeze_state_lock(self):
        """A saturated DAQ queue must not freeze stop()/state/configure().

        emit_array enqueues via the lossless "daq" policy, whose put() blocks
        when the dispatcher falls behind. If that put runs while holding
        _state_lock, every control-plane operation (stop, state, configure,
        set_active_channels) freezes with it — a user clicking Stop could not
        stop the device. The fix releases _state_lock before the put, so the
        lock stays available while the put blocks.

        On the buggy code the background emit holds the lock for the full ~10s
        put timeout, so the `device.state` read below blocks well past 1s and
        this assertion fails. On the fixed code the read returns immediately.
        """
        device = SimulatedPhysiologySource(queue_maxsize=2)
        try:
            devices = device.list_available_devices()
            device.open(devices[0].id)
            device.configure(sample_rate=20000, channels=[0, 1], chunk_size=256)

            data = np.random.randn(64, 2).astype(np.float32)

            # Saturate the data queue so the next put blocks (lossless policy).
            device.emit_array(data.copy())
            device.emit_array(data.copy())
            assert device.data_queue.full(), "queue should be saturated for the test"

            # A further emit blocks inside _safe_put. Everything before the put
            # is non-blocking, so a still-alive thread here is stuck in the put.
            blocked = threading.Thread(
                target=lambda: device.emit_array(data.copy()), daemon=True
            )
            blocked.start()
            time.sleep(0.1)  # let it pass the ring write and reach the put
            assert blocked.is_alive(), "background emit should be blocked in the put"

            # Reading state needs _state_lock; it must not wait for the put.
            start = time.perf_counter()
            _ = device.state
            elapsed = time.perf_counter() - start
            assert elapsed < 1.0, (
                f"_state_lock was held across the blocking put: "
                f"state read took {elapsed:.2f}s"
            )

            # Unblock the background emit by freeing a queue slot.
            try:
                device.data_queue.get_nowait()
            except queue.Empty:
                pass
            blocked.join(timeout=11.0)
            assert not blocked.is_alive(), "background emit should complete after drain"
        finally:
            device.close()
