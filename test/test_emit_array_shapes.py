# test/test_emit_array_shapes.py
"""Tests for BaseDevice.emit_array shape handling.

Verifies that emit_array correctly handles both (frames, channels) and
(channels, frames) data orientations.
"""
from __future__ import annotations

import queue
import numpy as np
import pytest

from daq.simulated_source import SimulatedPhysiologySource
from shared.models import ChunkPointer


class TestEmitArrayShapes:
    """Tests for emit_array shape orientation detection."""

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

    def test_channels_frames_shape(self, configured_device):
        """emit_array should handle (channels, frames) shape correctly."""
        device = configured_device
        
        # Create data with shape (channels=2, frames=100)
        frames, channels = 100, 2
        data = np.random.randn(channels, frames).astype(np.float32)
        
        # Should not raise - auto-detected as channel-major
        pointer = device.emit_array(data)
        
        assert isinstance(pointer, ChunkPointer)
        assert pointer.length == frames

    def test_buffer_content_consistency(self, configured_device):
        """Both orientations should produce identical buffer content."""
        device = configured_device
        
        # Reference data in (frames, channels) format
        frames, channels = 50, 2
        original = np.array([
            [1.0, 2.0],  # frame 0
            [3.0, 4.0],  # frame 1
            [5.0, 6.0],  # frame 2
        ], dtype=np.float32)
        
        # Test with (frames, channels) orientation
        device._reset_counters()
        ptr1 = device.emit_array(original.copy())
        buffer1 = device.ring_buffer.read(ptr1.start_index, ptr1.length)
        
        # Test with (channels, frames) orientation (transposed)
        device._reset_counters()
        ptr2 = device.emit_array(original.T.copy())
        buffer2 = device.ring_buffer.read(ptr2.start_index, ptr2.length)
        
        # Both should produce the same channel-major buffer content
        np.testing.assert_array_almost_equal(buffer1, buffer2)
        
        # Verify the expected channel-major shape: (channels, frames) 
        assert buffer1.shape == (channels, 3)

    def test_mismatched_dimensions_error(self, configured_device):
        """emit_array should raise on completely mismatched dimensions."""
        device = configured_device
        
        # Neither dimension matches expected 2 channels
        data = np.random.randn(5, 7).astype(np.float32)
        
        with pytest.raises(ValueError, match="does not match expected"):
            device.emit_array(data)

    def test_square_array_assumes_frames_channels(self, configured_device):
        """Square arrays should be treated as (frames, channels) for backward compat."""
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
            
            # Should treat as (frames=3, channels=3) and transpose
            pointer = device.emit_array(data)
            assert pointer.length == 3
        finally:
            device.close()
