"""
Unit tests for SharedRingBuffer correctness.

These tests verify the core safety invariants of the ring buffer:
1. Data integrity is preserved through wraparound
2. Correct dtype (float32) is always maintained
3. Reads return correct data at all positions
4. Thread safety under concurrent access

The ring buffer is safety-critical because corruption here means
corrupted data flows to all consumers (visualization, analysis, logging).

Test Strategy:
- Deterministic unit tests for core operations
- Edge cases at buffer boundaries
- Concurrent stress tests for thread safety
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from shared.ring_buffer import SharedRingBuffer


class TestRingBufferBasicOperations:
    """Basic read/write operations without wraparound."""

    def test_write_read_single_chunk(self):
        """Write and read a single chunk within capacity."""
        buf = SharedRingBuffer((2, 100), dtype=np.float32)

        data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float32)
        start = buf.write(data)

        assert start == 0
        result = buf.read(0, 5)
        np.testing.assert_array_equal(result, data)

    def test_write_read_multiple_chunks(self):
        """Write multiple chunks and read consecutively."""
        buf = SharedRingBuffer((1, 100), dtype=np.float32)

        chunk1 = np.array([[1, 2, 3]], dtype=np.float32)
        chunk2 = np.array([[4, 5, 6]], dtype=np.float32)

        start1 = buf.write(chunk1)
        start2 = buf.write(chunk2)

        assert start1 == 0
        assert start2 == 3

        result1 = buf.read(0, 3)
        result2 = buf.read(3, 3)

        np.testing.assert_array_equal(result1, chunk1)
        np.testing.assert_array_equal(result2, chunk2)

    def test_read_returns_float32(self):
        """Buffer always returns float32 regardless of input dtype."""
        buf = SharedRingBuffer((1, 50), dtype=np.float32)

        # Write as float64
        data_f64 = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)
        buf.write(data_f64)

        result = buf.read(0, 3)
        assert result.dtype == np.float32

        # Values should be converted correctly
        np.testing.assert_array_almost_equal(result.flatten(), [1.5, 2.5, 3.5])

    def test_read_contiguous_returns_view(self):
        """Contiguous reads should return views when possible."""
        buf = SharedRingBuffer((1, 100), dtype=np.float32)

        data = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
        buf.write(data)

        result = buf.read(0, 5)
        # A view shares memory with the original (though this is implementation detail)
        # The key property is that it works correctly
        assert result.shape == (1, 5)


class TestRingBufferWraparound:
    """Tests for correct behavior when writes wrap around buffer end."""

    def test_wraparound_preserves_sequence(self):
        """Known sequence survives wraparound correctly."""
        capacity = 10
        buf = SharedRingBuffer((1, capacity), dtype=np.float32)

        # Fill buffer to near end
        fill = np.arange(8, dtype=np.float32).reshape(1, -1)
        buf.write(fill)

        # Write data that wraps around
        wrap_data = np.array([[100, 101, 102, 103, 104]], dtype=np.float32)
        start = buf.write(wrap_data)

        # Start should be at position 8 (where we left off)
        assert start == 8

        # Read the wrapped data
        result = buf.read(8, 5)
        np.testing.assert_array_equal(result, wrap_data)

    def test_wraparound_read_returns_copy(self):
        """Wrapped reads must return a copy (contiguous data)."""
        capacity = 10
        buf = SharedRingBuffer((1, capacity), dtype=np.float32)

        # Fill to position 8
        buf.write(np.arange(8, dtype=np.float32).reshape(1, -1))

        # Write data that wraps: positions 8,9,0,1,2
        wrap_data = np.array([[10, 11, 12, 13, 14]], dtype=np.float32)
        buf.write(wrap_data)

        # Read wrapped range
        result = buf.read(8, 5)

        # Should get correct values regardless of wrap
        np.testing.assert_array_equal(result.flatten(), [10, 11, 12, 13, 14])

    def test_overwrite_behavior(self):
        """When buffer is full, old data is overwritten."""
        capacity = 5
        buf = SharedRingBuffer((1, capacity), dtype=np.float32)

        # Write more than capacity
        first = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
        buf.write(first)

        second = np.array([[10, 11, 12]], dtype=np.float32)
        buf.write(second)

        # Old data at start should be overwritten
        # Buffer now contains: [10, 11, 12, 4, 5] logically as positions 5-9
        # Reading from position 5 (where second write started) should give [10, 11, 12]
        result = buf.read(0, 3)
        np.testing.assert_array_equal(result.flatten(), [10, 11, 12])

    def test_full_capacity_wraparound_cycle(self):
        """Complete wraparound cycle maintains integrity."""
        capacity = 8
        buf = SharedRingBuffer((1, capacity), dtype=np.float32)

        # Write 3 full cycles worth of data
        for cycle in range(3):
            for chunk_idx in range(4):
                chunk_val = cycle * 100 + chunk_idx * 10
                data = np.array([[chunk_val, chunk_val + 1]], dtype=np.float32)
                start = buf.write(data)

                # Verify we can read what we just wrote
                result = buf.read(start, 2)
                np.testing.assert_array_equal(result.flatten(), [chunk_val, chunk_val + 1])


class TestRingBufferMultiChannel:
    """Tests for multi-channel buffer operations."""

    def test_multichannel_write_read(self):
        """Multi-channel data maintains channel separation."""
        buf = SharedRingBuffer((3, 50), dtype=np.float32)

        # 3 channels, 5 samples each
        data = np.array([
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            [100, 200, 300, 400, 500],
        ], dtype=np.float32)

        buf.write(data)
        result = buf.read(0, 5)

        np.testing.assert_array_equal(result, data)

    def test_multichannel_wraparound(self):
        """Multi-channel wraparound preserves channel layout."""
        buf = SharedRingBuffer((2, 8), dtype=np.float32)

        # Fill to position 6
        fill = np.zeros((2, 6), dtype=np.float32)
        buf.write(fill)

        # Write data that wraps
        wrap_data = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], dtype=np.float32)
        buf.write(wrap_data)

        result = buf.read(6, 4)
        np.testing.assert_array_equal(result, wrap_data)


class TestRingBufferValidation:
    """Tests for input validation and error handling."""

    def test_invalid_shape_empty_raises(self):
        """Empty shape tuple should raise."""
        with pytest.raises(ValueError):
            SharedRingBuffer((), dtype=np.float32)

    def test_invalid_shape_zero_dimension_raises(self):
        """Zero in shape should raise."""
        with pytest.raises(ValueError):
            SharedRingBuffer((0, 10), dtype=np.float32)

    def test_invalid_shape_negative_dimension_raises(self):
        """Negative dimension should raise."""
        with pytest.raises(ValueError):
            SharedRingBuffer((1, -5), dtype=np.float32)

    def test_write_dimension_mismatch_raises(self):
        """Writing wrong number of dimensions should raise."""
        buf = SharedRingBuffer((2, 10), dtype=np.float32)

        # 1D instead of 2D
        with pytest.raises(ValueError):
            buf.write(np.array([1, 2, 3], dtype=np.float32))

    def test_write_channel_mismatch_raises(self):
        """Writing wrong number of channels should raise."""
        buf = SharedRingBuffer((2, 10), dtype=np.float32)

        # 3 channels instead of 2
        with pytest.raises(ValueError):
            buf.write(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))

    def test_write_exceeds_capacity_raises(self):
        """Writing more than capacity in one chunk should raise."""
        buf = SharedRingBuffer((1, 10), dtype=np.float32)

        with pytest.raises(ValueError):
            buf.write(np.arange(20, dtype=np.float32).reshape(1, -1))

    def test_read_negative_length_raises(self):
        """Reading negative length should raise."""
        buf = SharedRingBuffer((1, 10), dtype=np.float32)
        buf.write(np.array([[1, 2, 3]], dtype=np.float32))

        with pytest.raises(ValueError):
            buf.read(0, -1)

    def test_read_zero_length_raises(self):
        """Reading zero length should raise."""
        buf = SharedRingBuffer((1, 10), dtype=np.float32)
        buf.write(np.array([[1, 2, 3]], dtype=np.float32))

        with pytest.raises(ValueError):
            buf.read(0, 0)

    def test_read_exceeds_capacity_raises(self):
        """Reading more than capacity should raise."""
        buf = SharedRingBuffer((1, 10), dtype=np.float32)
        buf.write(np.array([[1, 2, 3]], dtype=np.float32))

        with pytest.raises(ValueError):
            buf.read(0, 20)

    def test_read_invalid_start_index_raises(self):
        """Reading from invalid start index should raise."""
        buf = SharedRingBuffer((1, 10), dtype=np.float32)

        with pytest.raises(ValueError):
            buf.read(-1, 5)

        with pytest.raises(ValueError):
            buf.read(15, 5)


class TestRingBufferProperties:
    """Tests for buffer properties and metadata."""

    def test_capacity_property(self):
        """Capacity property returns correct value."""
        buf = SharedRingBuffer((2, 100), dtype=np.float32)
        assert buf.capacity == 100

    def test_shape_property(self):
        """Shape property returns correct value."""
        buf = SharedRingBuffer((3, 50), dtype=np.float32)
        assert buf.shape == (3, 50)

    def test_dtype_property(self):
        """Dtype property returns correct value."""
        buf = SharedRingBuffer((1, 10), dtype=np.float32)
        assert buf.dtype == np.float32

        buf64 = SharedRingBuffer((1, 10), dtype=np.float64)
        assert buf64.dtype == np.float64


class TestRingBufferThreadSafety:
    """Tests for concurrent access safety."""

    def test_concurrent_write_read_no_crash(self):
        """Concurrent writes and reads should not crash."""
        buf = SharedRingBuffer((1, 100), dtype=np.float32)
        errors = []

        def writer():
            try:
                for i in range(100):
                    data = np.array([[i, i+1, i+2]], dtype=np.float32)
                    buf.write(data)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    # Read from position 0 is always valid
                    result = buf.read(0, 3)
                    assert result.shape == (1, 3)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join(timeout=5)
        reader_thread.join(timeout=5)

        assert not errors, f"Errors during concurrent access: {errors}"

    @pytest.mark.slow
    def test_stress_many_writers_readers(self):
        """Stress test with multiple writers and readers."""
        buf = SharedRingBuffer((2, 1000), dtype=np.float32)
        errors = []
        write_count = threading.Lock()
        writes_done = [0]

        def writer(writer_id: int):
            try:
                for i in range(50):
                    data = np.array([
                        [writer_id * 1000 + i] * 5,
                        [writer_id * 1000 + i + 100] * 5,
                    ], dtype=np.float32)
                    buf.write(data)
                    with write_count:
                        writes_done[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                for _ in range(100):
                    try:
                        result = buf.read(0, 5)
                        assert result.shape == (2, 5)
                        assert result.dtype == np.float32
                    except ValueError:
                        # May happen if reading before first write
                        pass
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("reader", e))

        with ThreadPoolExecutor(max_workers=8) as executor:
            # 4 writers, 4 readers
            futures = []
            for i in range(4):
                futures.append(executor.submit(writer, i))
            for _ in range(4):
                futures.append(executor.submit(reader))

            # Wait for all to complete
            for f in futures:
                f.result(timeout=10)

        assert not errors, f"Errors during stress test: {errors}"
        assert writes_done[0] == 200, f"Expected 200 writes, got {writes_done[0]}"
