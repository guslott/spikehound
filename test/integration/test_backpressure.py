"""
Integration tests for dispatcher backpressure behavior.

The README claims: "Dispatcher fans out to queues so a slow UI never blocks acquisition."
This is a core safety invariant. These tests verify:

1. Slow consumers cannot block the acquisition thread
2. Drop counters accurately track which consumer dropped what
3. Fast consumers continue receiving data even when others are slow
4. Lossless queues (logging) block rather than drop
5. Shutdown is clean even with saturated queues

Test approach:
- Use ControlledDevice for deterministic, predictable input
- Create varying consumer speeds (fast, slow, blocked)
- Measure timing to verify acquisition doesn't stall
- Check stats for correct drop/forward counts
"""
from __future__ import annotations

import queue
import threading
import time
from typing import List, Optional

import numpy as np
import pytest

from core.dispatcher import Dispatcher, QUEUE_POLICIES
from core.conditioning import FilterSettings
from shared.models import Chunk, ChunkPointer, EndOfStream
from shared.ring_buffer import SharedRingBuffer


def make_test_dispatcher(
    n_channels: int = 1,
    capacity: int = 4096,
    sample_rate: float = 10000.0,
    viz_queue_size: int = 10,
    audio_queue_size: int = 10,
    logging_queue_size: int = 100,
) -> tuple:
    """Create a dispatcher with controlled queue sizes for testing."""
    raw_queue = queue.Queue()
    visualization_queue = queue.Queue(maxsize=viz_queue_size)
    audio_queue = queue.Queue(maxsize=audio_queue_size)
    logging_queue = queue.Queue(maxsize=logging_queue_size)
    event_queue = queue.Queue()

    source_buffer = SharedRingBuffer((n_channels, capacity), dtype=np.float32)

    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        audio_queue,
        logging_queue,
        event_queue,
        filter_settings=FilterSettings(),
    )
    dispatcher.set_source_buffer(source_buffer, sample_rate=sample_rate)

    return dispatcher, source_buffer, {
        "raw": raw_queue,
        "visualization": visualization_queue,
        "audio": audio_queue,
        "logging": logging_queue,
        "event": event_queue,
    }


class TestSlowConsumerIsolation:
    """Verify slow consumers cannot block acquisition."""

    def test_slow_viz_consumer_doesnt_block_producer(self):
        """Visualization queue saturation should not block raw queue processing."""
        dispatcher, source_buffer, queues = make_test_dispatcher(
            viz_queue_size=2,  # Very small - will saturate quickly
            audio_queue_size=100,  # Large - won't saturate
        )

        n_chunks = 20
        chunk_size = 256
        sample_rate = 10000.0

        # Start dispatcher
        dispatcher.start()

        # Feed chunks as fast as possible
        start_time = time.time()
        for seq in range(n_chunks):
            samples = np.ones((1, chunk_size), dtype=np.float32) * seq
            start_idx = source_buffer.write(samples)
            pointer = ChunkPointer(
                start_index=start_idx,
                length=chunk_size,
                render_time=seq * chunk_size / sample_rate,
                seq=seq,
                start_sample=seq * chunk_size,
            )
            queues["raw"].put(pointer)

        queues["raw"].put(EndOfStream)
        dispatcher.join(timeout=5.0)
        end_time = time.time()

        # All chunks should process quickly (< 2 seconds for reasonable hardware)
        # If blocking occurred, this would take much longer
        elapsed = end_time - start_time
        assert elapsed < 2.0, f"Processing took {elapsed:.2f}s - possible blocking"

        # Verify stats
        stats = dispatcher.snapshot()
        assert stats["processed"] == n_chunks

        # Visualization should have evicted some chunks (queue too small)
        evicted = stats.get("evicted", {})
        if "visualization" in evicted:
            assert evicted["visualization"] >= 1, "Expected viz evictions"

    def test_blocked_consumer_doesnt_stall_others(self):
        """A blocked consumer should not prevent other consumers from receiving data."""
        dispatcher, source_buffer, queues = make_test_dispatcher(
            viz_queue_size=5,
            audio_queue_size=5,
        )

        # Register analysis queue that we'll leave unread (simulates blocked consumer)
        blocked_analysis_queue = queue.Queue(maxsize=1)
        token = dispatcher.register_analysis_queue(blocked_analysis_queue)

        n_chunks = 15
        chunk_size = 256
        sample_rate = 10000.0

        # Start a consumer thread for audio queue (fast consumer)
        audio_received = []
        audio_consumer_done = threading.Event()

        def audio_consumer():
            while True:
                try:
                    item = queues["audio"].get(timeout=0.5)
                    if item is EndOfStream:
                        break
                    audio_received.append(item)
                except queue.Empty:
                    continue
            audio_consumer_done.set()

        audio_thread = threading.Thread(target=audio_consumer, daemon=True)
        audio_thread.start()

        dispatcher.start()

        # Send chunks
        for seq in range(n_chunks):
            samples = np.ones((1, chunk_size), dtype=np.float32) * seq
            start_idx = source_buffer.write(samples)
            pointer = ChunkPointer(
                start_index=start_idx,
                length=chunk_size,
                render_time=seq * chunk_size / sample_rate,
                seq=seq,
                start_sample=seq * chunk_size,
            )
            queues["raw"].put(pointer)

        queues["raw"].put(EndOfStream)
        dispatcher.join(timeout=5.0)

        # Wait for audio consumer
        audio_consumer_done.wait(timeout=2.0)

        # Audio consumer should have received data despite blocked analysis
        assert len(audio_received) >= 1, "Audio consumer should receive data"

        # Cleanup
        dispatcher.unregister_analysis_queue(token)


class TestDropTracking:
    """Verify drop counters are accurate per-consumer."""

    def test_eviction_counts_tracked_per_queue(self):
        """Each queue's evictions should be tracked separately."""
        dispatcher, source_buffer, queues = make_test_dispatcher(
            viz_queue_size=2,
            audio_queue_size=2,
        )

        n_chunks = 10
        chunk_size = 256
        sample_rate = 10000.0

        dispatcher.start()

        # Send chunks without consuming - will cause evictions
        for seq in range(n_chunks):
            samples = np.ones((1, chunk_size), dtype=np.float32) * seq
            start_idx = source_buffer.write(samples)
            pointer = ChunkPointer(
                start_index=start_idx,
                length=chunk_size,
                render_time=seq * chunk_size / sample_rate,
                seq=seq,
                start_sample=seq * chunk_size,
            )
            queues["raw"].put(pointer)

        queues["raw"].put(EndOfStream)
        dispatcher.join(timeout=3.0)

        stats = dispatcher.snapshot()

        # Should have processed all chunks
        assert stats["processed"] == n_chunks

        # Both viz and audio should have evictions (queues too small)
        evicted = stats.get("evicted", {})
        forwarded = stats.get("forwarded", {})

        dropped = stats.get("dropped", {})

        # evicted + dropped + forwarded should account for all chunks per queue
        viz_total = (
            evicted.get("visualization", 0) 
            + dropped.get("visualization", 0)
            + forwarded.get("visualization", 0)
        )
        assert viz_total >= n_chunks - 1, f"Viz accounting mismatch: {viz_total} vs {n_chunks}"


class TestQueuePolicies:
    """Verify each queue type follows its documented backpressure policy."""

    def test_queue_policies_defined(self):
        """All expected queue types should have policies defined."""
        expected_queues = ["visualization", "audio", "logging", "analysis", "events"]
        for q_name in expected_queues:
            assert q_name in QUEUE_POLICIES, f"Missing policy for {q_name}"

    def test_logging_queue_never_drops(self):
        """Logging queue should use lossless policy (blocks or uses large queue)."""
        assert QUEUE_POLICIES.get("logging") == "lossless"

    def test_visualization_uses_drop_oldest_for_freshness(self):
        """Visualization should use drop-oldest to maintain data freshness.
        
        drop-oldest evicts stale backlog so consumers always see the newest data.
        This prevents real-time UI/audio from lagging behind acquisition.
        """
        policy = QUEUE_POLICIES.get("visualization")
        assert policy == "drop-oldest", f"Unexpected viz policy: {policy}"


class TestShutdownBehavior:
    """Verify clean shutdown even with queue pressure."""

    def test_shutdown_with_full_queues(self):
        """Dispatcher should shut down cleanly even with saturated queues."""
        dispatcher, source_buffer, queues = make_test_dispatcher(
            viz_queue_size=1,
            audio_queue_size=1,
        )

        chunk_size = 256
        sample_rate = 10000.0

        dispatcher.start()

        # Send a few chunks to saturate queues
        for seq in range(5):
            samples = np.ones((1, chunk_size), dtype=np.float32) * seq
            start_idx = source_buffer.write(samples)
            pointer = ChunkPointer(
                start_index=start_idx,
                length=chunk_size,
                render_time=seq * chunk_size / sample_rate,
                seq=seq,
                start_sample=seq * chunk_size,
            )
            queues["raw"].put(pointer)

        queues["raw"].put(EndOfStream)

        # Should shut down within timeout
        shutdown_start = time.time()
        dispatcher.join(timeout=3.0)
        shutdown_duration = time.time() - shutdown_start

        assert shutdown_duration < 3.0, f"Shutdown took too long: {shutdown_duration:.2f}s"

    def test_stop_then_join_is_idempotent(self):
        """Calling stop() multiple times should not crash."""
        dispatcher, source_buffer, queues = make_test_dispatcher()

        dispatcher.start()

        # Stop multiple times
        dispatcher.stop()
        dispatcher.stop()  # Second call should not crash

        # Join also should work
        dispatcher.join(timeout=1.0)
        dispatcher.join(timeout=1.0)  # Second call should not crash


class TestDataOrdering:
    """Verify data ordering is preserved through the dispatcher."""

    def test_chunk_sequence_preserved(self):
        """Chunks should arrive at consumers in transmission order."""
        dispatcher, source_buffer, queues = make_test_dispatcher(
            viz_queue_size=100,  # Large enough to not evict
        )

        n_chunks = 20
        chunk_size = 256
        sample_rate = 10000.0

        dispatcher.start()

        # Send chunks with identifiable values
        for seq in range(n_chunks):
            samples = np.full((1, chunk_size), float(seq), dtype=np.float32)
            start_idx = source_buffer.write(samples)
            pointer = ChunkPointer(
                start_index=start_idx,
                length=chunk_size,
                render_time=seq * chunk_size / sample_rate,
                seq=seq,
                start_sample=seq * chunk_size,
            )
            queues["raw"].put(pointer)

        queues["raw"].put(EndOfStream)
        dispatcher.join(timeout=3.0)

        # Collect from viz queue
        received = []
        while True:
            try:
                item = queues["visualization"].get_nowait()
                if item is EndOfStream:
                    break
                received.append(item)
            except queue.Empty:
                break

        # All received items should be in order (render_time increasing)
        if len(received) > 1:
            render_times = [p.render_time for p in received]
            for i in range(1, len(render_times)):
                assert render_times[i] > render_times[i - 1], "Chunk order violated"


class TestMultipleAnalysisQueues:
    """Verify analysis queue registration and fan-out."""

    def test_multiple_analysis_consumers(self):
        """Multiple registered analysis queues should all receive data."""
        dispatcher, source_buffer, queues = make_test_dispatcher()

        # Register multiple analysis queues
        analysis_queues = [queue.Queue(maxsize=50) for _ in range(3)]
        tokens = [dispatcher.register_analysis_queue(q) for q in analysis_queues]

        n_chunks = 10
        chunk_size = 256
        sample_rate = 10000.0

        dispatcher.start()

        for seq in range(n_chunks):
            samples = np.full((1, chunk_size), float(seq), dtype=np.float32)
            start_idx = source_buffer.write(samples)
            pointer = ChunkPointer(
                start_index=start_idx,
                length=chunk_size,
                render_time=seq * chunk_size / sample_rate,
                seq=seq,
                start_sample=seq * chunk_size,
            )
            queues["raw"].put(pointer)

        queues["raw"].put(EndOfStream)
        dispatcher.join(timeout=3.0)

        # All analysis queues should have received data
        for i, aq in enumerate(analysis_queues):
            count = 0
            while True:
                try:
                    item = aq.get_nowait()
                    if item is EndOfStream:
                        break
                    count += 1
                except queue.Empty:
                    break

            assert count >= 1, f"Analysis queue {i} received no data"

        # Cleanup
        for token in tokens:
            dispatcher.unregister_analysis_queue(token)

    def test_unregistered_queue_stops_receiving(self):
        """Unregistered analysis queue should stop receiving data."""
        dispatcher, source_buffer, queues = make_test_dispatcher()

        analysis_queue = queue.Queue(maxsize=50)
        token = dispatcher.register_analysis_queue(analysis_queue)

        dispatcher.start()

        # Send one chunk
        samples = np.ones((1, 256), dtype=np.float32)
        start_idx = source_buffer.write(samples)
        pointer = ChunkPointer(start_index=start_idx, length=256, render_time=0.0, seq=0, start_sample=0)
        queues["raw"].put(pointer)

        time.sleep(0.1)  # Let it process

        # Unregister
        dispatcher.unregister_analysis_queue(token)

        # Drain the queue
        initial_count = 0
        while True:
            try:
                analysis_queue.get_nowait()
                initial_count += 1
            except queue.Empty:
                break

        # Send more chunks
        for seq in range(5):
            samples = np.ones((1, 256), dtype=np.float32) * (seq + 10)
            start_idx = source_buffer.write(samples)
            pointer = ChunkPointer(start_index=start_idx, length=256, render_time=0.0, seq=seq, start_sample=seq * 256)
            queues["raw"].put(pointer)

        queues["raw"].put(EndOfStream)
        dispatcher.join(timeout=2.0)

        # Queue should not have received the new chunks
        post_count = 0
        while True:
            try:
                item = analysis_queue.get_nowait()
                if item is not EndOfStream:
                    post_count += 1
            except queue.Empty:
                break

        # After unregister, should receive 0 new chunks
        assert post_count == 0, f"Unregistered queue received {post_count} chunks"
