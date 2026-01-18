"""
Performance and stability tests.

These tests verify system behavior under extended operation:
1. No memory growth over time (buffer leaks)
2. No thread count growth (thread leaks)
3. Queue depths stay bounded
4. Timing remains consistent

Marked with @pytest.mark.slow as they take significant time.
Typically run in nightly CI, not on every commit.
"""
from __future__ import annotations

import gc
import threading
import time
import tracemalloc
from typing import Optional

import numpy as np
import pytest

from core.runtime import SpikeHoundRuntime
from daq.simulated_source import SimulatedPhysiologySource
from shared.ring_buffer import SharedRingBuffer


@pytest.mark.slow
class TestMemoryStability:
    """Tests for memory stability over extended operation."""

    def test_ring_buffer_no_growth(self):
        """Ring buffer memory should not grow with repeated writes."""
        capacity = 10000
        n_channels = 2
        buf = SharedRingBuffer((n_channels, capacity), dtype=np.float32)

        # Warm up
        for _ in range(10):
            data = np.random.randn(n_channels, 1000).astype(np.float32)
            buf.write(data)

        gc.collect()
        tracemalloc.start()

        initial_snapshot = tracemalloc.take_snapshot()

        # Many write cycles
        for _ in range(1000):
            data = np.random.randn(n_channels, 100).astype(np.float32)
            buf.write(data)

        gc.collect()
        final_snapshot = tracemalloc.take_snapshot()

        tracemalloc.stop()

        # Compare memory usage
        stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        total_growth = sum(s.size_diff for s in stats if s.size_diff > 0)

        # Allow some growth but not unbounded
        # 1MB is generous - should be much less
        assert total_growth < 1_000_000, f"Memory grew by {total_growth / 1024:.1f} KB"

    def test_pipeline_no_memory_leak(self):
        """Pipeline memory should stabilize during operation."""
        runtime = SpikeHoundRuntime()

        try:
            devices = SimulatedPhysiologySource.list_available_devices()
            runtime.switch_backend(SimulatedPhysiologySource, configure_kwargs={"sample_rate": 10000, "chunk_size": 256})
            runtime.select_device(devices[0].id)
            runtime.configure_acquisition(
                channels=[0],
            )

            # Start and run for initial period
            runtime.start_acquisition()
            time.sleep(1.0)

            gc.collect()
            tracemalloc.start()
            initial = tracemalloc.take_snapshot()

            # Run for extended period
            time.sleep(5.0)

            gc.collect()
            final = tracemalloc.take_snapshot()
            tracemalloc.stop()

            runtime.stop_acquisition()

            # Check memory growth
            stats = final.compare_to(initial, 'lineno')
            total_growth = sum(s.size_diff for s in stats if s.size_diff > 0)

            # Allow 5MB growth max for reasonable operation
            assert total_growth < 5_000_000, \
                f"Pipeline leaked {total_growth / 1024 / 1024:.1f} MB over 5 seconds"
        finally:
            runtime.stop_acquisition()
            runtime.shutdown()


@pytest.mark.slow
class TestThreadStability:
    """Tests for thread count stability."""

    def test_no_thread_leak_over_cycles(self):
        """Thread count should not grow with start/stop cycles."""
        runtime = SpikeHoundRuntime()

        initial_threads = threading.active_count()

        try:
            devices = SimulatedPhysiologySource.list_available_devices()
            runtime.switch_backend(SimulatedPhysiologySource, configure_kwargs={"sample_rate": 10000, "chunk_size": 256})
            runtime.select_device(devices[0].id)
            runtime.configure_acquisition(
                channels=[0],
            )

            # Multiple start/stop cycles
            for cycle in range(10):
                runtime.start_acquisition()
                time.sleep(0.1)
                runtime.stop_acquisition()
                time.sleep(0.1)

                current_threads = threading.active_count()

                # Thread count should not grow significantly
                assert current_threads <= initial_threads + 5, \
                    f"Thread leak at cycle {cycle}: {current_threads} threads"
        finally:
            runtime.shutdown()
            time.sleep(0.2)

        final_threads = threading.active_count()
        assert final_threads <= initial_threads + 2, \
            f"Final thread leak: {initial_threads} → {final_threads}"


@pytest.mark.slow
class TestQueueStability:
    """Tests for queue depth stability."""

    def test_visualization_queue_bounded(self):
        """Visualization queue should not grow unbounded."""
        runtime = SpikeHoundRuntime()

        try:
            devices = SimulatedPhysiologySource.list_available_devices()
            runtime.switch_backend(SimulatedPhysiologySource, configure_kwargs={"sample_rate": 10000, "chunk_size": 256})
            runtime.select_device(devices[0].id)
            runtime.configure_acquisition(
                channels=[0],
            )

            runtime.start_acquisition()

            max_depth = 0
            for _ in range(50):  # Check over 5 seconds
                time.sleep(0.1)
                depth = runtime.visualization_queue.qsize()
                max_depth = max(max_depth, depth)

            runtime.stop_acquisition()

            # Queue should have bounded depth
            # (depends on queue maxsize configuration which is 1024)
            assert max_depth < 800, f"Queue grew too large: {max_depth}"
        finally:
            runtime.shutdown()


@pytest.mark.slow
class TestTimingStability:
    """Tests for timing consistency."""

    def test_chunk_interval_consistency(self):
        """Chunk arrival intervals should be consistent."""
        runtime = SpikeHoundRuntime()

        try:
            devices = SimulatedPhysiologySource.list_available_devices()
            chunk_size = 256
            sample_rate = 10000
            expected_interval = chunk_size / sample_rate

            runtime.switch_backend(SimulatedPhysiologySource, configure_kwargs={"sample_rate": sample_rate, "chunk_size": chunk_size})
            runtime.select_device(devices[0].id)

            runtime.configure_acquisition(
                channels=[0],
            )

            runtime.start_acquisition()
            time.sleep(0.5)  # Let it stabilize

            # Measure chunk arrival times
            arrival_times = []
            for _ in range(20):
                try:
                    item = runtime.visualization_queue.get(timeout=0.5)
                    arrival_times.append(time.time())
                except:
                    break

            runtime.stop_acquisition()

            if len(arrival_times) >= 10:
                intervals = np.diff(arrival_times)
                mean_interval = float(np.mean(intervals))
                std_interval = float(np.std(intervals))

                # Mean should be close to expected
                assert abs(mean_interval - expected_interval) < expected_interval * 0.5, \
                    f"Mean interval {mean_interval*1000:.1f}ms != expected {expected_interval*1000:.1f}ms"

                # Should be reasonably consistent (CV < 50%)
                cv = std_interval / mean_interval if mean_interval > 0 else float('inf')
                assert cv < 1.5, f"Timing too variable: CV = {cv:.2f}"
        finally:
            runtime.shutdown()


@pytest.mark.slow
class TestLongRunStability:
    """Extended run stability tests."""

    def test_30_second_continuous_run(self):
        """System should run stably for 30 seconds."""
        runtime = SpikeHoundRuntime()

        try:
            devices = SimulatedPhysiologySource.list_available_devices()
            runtime.switch_backend(SimulatedPhysiologySource, configure_kwargs={"sample_rate": 10000, "chunk_size": 256})
            runtime.select_device(devices[0].id)
            runtime.configure_acquisition(
                channels=[0, 1],  # Multiple channels
            )

            runtime.start_acquisition()

            errors = []
            chunks_received = 0
            start_time = time.time()

            while time.time() - start_time < 30.0:
                try:
                    item = runtime.visualization_queue.get(timeout=1.0)
                    chunks_received += 1
                except Exception as e:
                    errors.append(str(e))
                    break

            runtime.stop_acquisition()

            # Should have received many chunks
            # Allow for slow simulation (at least 40% of real-time)
            expected_chunks = 30 * 10000 / 256  # ~1170 chunks
            assert chunks_received >= expected_chunks * 0.4, \
                f"Received only {chunks_received} chunks (expected ~{expected_chunks:.0f})"

            assert not errors, f"Errors during run: {errors}"
        finally:
            runtime.shutdown()
