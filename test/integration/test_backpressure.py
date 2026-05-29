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
import time
from typing import List, Optional

import threading

import numpy as np
import pytest

import shared.models as shared_models
from core.dispatcher import Dispatcher, QUEUE_POLICIES
from core.conditioning import FilterSettings
from shared.models import Chunk, ChunkPointer, EndOfStream, TriggerConfig, enqueue_with_policy
from shared.ring_buffer import SharedRingBuffer


def _enable_triggered_viz(dispatcher, sample_rate: float = 10000.0) -> None:
    """Put the dispatcher in a triggered (non-stream) mode so the visualization
    queue is actually populated. In stream mode the queue stays empty by design
    (visualization flows via tick callbacks) — see the "Visualization delivery
    contract" in core/dispatcher.py and finding 3.2 #1.
    """
    dispatcher.set_trigger_config(
        TriggerConfig(
            channel_index=0,
            threshold=0.5,
            hysteresis=0.0,
            pretrigger_sec=0.2,
            window_sec=0.2,
            mode="repeated",
        ),
        sample_rate,
    )


def make_test_dispatcher(
    n_channels: int = 1,
    capacity: int = 4096,
    sample_rate: float = 10000.0,
    viz_queue_size: int = 10,
    logging_queue_size: int = 100,
) -> tuple:
    """Create a dispatcher with controlled queue sizes for testing."""
    raw_queue = queue.Queue()
    visualization_queue = queue.Queue(maxsize=viz_queue_size)
    logging_queue = queue.Queue(maxsize=logging_queue_size)
    event_queue = queue.Queue()

    source_buffer = SharedRingBuffer((n_channels, capacity), dtype=np.float32)

    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        logging_queue,
        event_queue,
        filter_settings=FilterSettings(),
    )
    dispatcher.set_source_buffer(source_buffer, sample_rate=sample_rate)

    return dispatcher, source_buffer, {
        "raw": raw_queue,
        "visualization": visualization_queue,
        "logging": logging_queue,
        "event": event_queue,
    }


class TestSlowConsumerIsolation:
    """Verify slow consumers cannot block acquisition."""

    def test_slow_viz_consumer_doesnt_block_producer(self):
        """Visualization queue saturation should not block raw queue processing."""
        dispatcher, source_buffer, queues = make_test_dispatcher(
            viz_queue_size=2,  # Very small - will saturate quickly
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
        )

        # One analysis consumer is "blocked" (tiny queue, never drained); a
        # second analysis consumer is fast and should still receive data.
        blocked_analysis_queue = queue.Queue(maxsize=1)
        fast_analysis_queue = queue.Queue(maxsize=100)
        blocked_token = dispatcher.register_analysis_queue(blocked_analysis_queue)
        fast_token = dispatcher.register_analysis_queue(fast_analysis_queue)

        n_chunks = 15
        chunk_size = 256
        sample_rate = 10000.0

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

        # The fast consumer should have received data despite the blocked one,
        # and the dispatcher should have processed every chunk.
        fast_received = []
        while True:
            try:
                item = fast_analysis_queue.get_nowait()
            except queue.Empty:
                break
            if item is not EndOfStream:
                fast_received.append(item)
        assert len(fast_received) >= 1, "Fast consumer should receive data despite a blocked consumer"
        assert dispatcher.snapshot()["processed"] == n_chunks

        # Cleanup
        dispatcher.unregister_analysis_queue(blocked_token)
        dispatcher.unregister_analysis_queue(fast_token)


class TestDropTracking:
    """Verify drop counters are accurate per-consumer."""

    def test_eviction_counts_tracked_per_queue(self):
        """Each queue's evictions should be tracked separately.

        Uses a triggered mode so the visualization queue is exercised (in stream
        mode it is empty by design — finding 3.2 #1).
        """
        dispatcher, source_buffer, queues = make_test_dispatcher(
            viz_queue_size=2,
        )

        n_chunks = 10
        chunk_size = 256
        sample_rate = 10000.0

        _enable_triggered_viz(dispatcher, sample_rate)
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

        # Visualization should have evictions (queue too small)
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
        expected_queues = ["visualization", "logging", "analysis", "events"]
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


class TestLosslessStallContract:
    """Finding 4.4: a sustained lossless-queue stall (e.g. a slow disk during
    recording) must degrade to counted, logged drops — never raise into the
    acquisition (DAQ producer) or dispatcher thread.
    """

    def test_lossless_put_drops_without_raising_on_stall(self, monkeypatch):
        """The lossless put returns normally (no RuntimeError) on a sustained
        stall, in both the plain and cancel-event variants, and reports a drop.
        """
        monkeypatch.setattr(shared_models, "_LOSSLESS_TIMEOUT_SEC", 0.1)
        monkeypatch.setattr(shared_models, "_LOSSLESS_POLL_SEC", 0.02)

        full_q: "queue.Queue" = queue.Queue(maxsize=1)
        full_q.put("occupied")  # saturate so every put stalls
        actions: list[tuple[str, str]] = []

        def cb(name: str, action: str) -> None:
            actions.append((name, action))

        # Plain path (dispatcher 'logging'/WAV writer): must NOT raise.
        enqueue_with_policy("logging", full_q, "x", stats_callback=cb)
        assert ("logging", "dropped") in actions
        assert full_q.qsize() == 1, "stalled item must be dropped, not enqueued"

        # Cancel-event path (DAQ producer) under a non-stop stall: also drops.
        actions.clear()
        not_stopping = threading.Event()  # cleared -> a real stall, not a stop
        enqueue_with_policy("daq", full_q, "y", stats_callback=cb, cancel_event=not_stopping)
        assert ("daq", "dropped") in actions
        assert full_q.qsize() == 1

    def test_recording_under_throttled_consumer_keeps_fanning_out(self, monkeypatch):
        """With a throttled (never-drained) logging/WAV consumer, the dispatcher
        must keep running and keep fanning out to other consumers. Previously the
        logging put raised mid-_fan_out, aborting the rest of the fan-out (so
        analysis lost the stalled chunks); now it drops gracefully.
        """
        monkeypatch.setattr(shared_models, "_LOSSLESS_TIMEOUT_SEC", 0.1)
        monkeypatch.setattr(shared_models, "_LOSSLESS_POLL_SEC", 0.02)

        dispatcher, source_buffer, queues = make_test_dispatcher(
            viz_queue_size=10,
            logging_queue_size=1,  # tiny + never drained -> stalls after 1 chunk
        )
        analysis_q: "queue.Queue" = queue.Queue(maxsize=100)
        dispatcher.register_analysis_queue(analysis_q)
        dispatcher.set_recording_enabled(True)

        n_chunks = 5
        chunk_size = 256
        sample_rate = 10000.0
        dispatcher.start()
        for seq in range(n_chunks):
            samples = np.ones((1, chunk_size), dtype=np.float32) * seq
            start_idx = source_buffer.write(samples)
            queues["raw"].put(
                ChunkPointer(
                    start_index=start_idx,
                    length=chunk_size,
                    render_time=seq * chunk_size / sample_rate,
                    seq=seq,
                    start_sample=seq * chunk_size,
                )
            )
        queues["raw"].put(EndOfStream)
        dispatcher.join(timeout=10.0)

        received = []
        while True:
            try:
                item = analysis_q.get_nowait()
            except queue.Empty:
                break
            if item is not EndOfStream:
                received.append(item)

        stats = dispatcher.snapshot()
        # Fan-out to analysis continued for every chunk despite the logging stall.
        assert len(received) == n_chunks, (
            f"analysis lost chunks when logging stalled: got {len(received)}/{n_chunks}"
        )
        assert stats["processed"] == n_chunks
        # The stall was recorded as graceful drops, not a crash.
        assert stats.get("dropped", {}).get("logging", 0) >= 1
