"""
Tests for Dispatcher with ChunkPointer/SharedRingBuffer API.

The Dispatcher reads raw samples via ChunkPointer objects that reference
data in a SharedRingBuffer. Tests must:
1. Create a SharedRingBuffer and write sample data
2. Create ChunkPointer referencing the written data
3. Call dispatcher.set_source_buffer() to link the buffer
4. Push ChunkPointers to the raw_queue
"""
import math
import queue
import time

import numpy as np
import pytest
from core import (
    ChannelFilterSettings,
    Chunk,
    Dispatcher,
    EndOfStream,
    FilterSettings,
    SignalConditioner,
)
from shared.models import ChunkPointer
from shared.ring_buffer import SharedRingBuffer


def _drain_chunks(target_queue: "queue.Queue"):
    """Drain all items from a queue, filtering out EndOfStream."""
    items = []
    while True:
        try:
            item = target_queue.get_nowait()
        except queue.Empty:
            break
        if item is EndOfStream:
            continue
        items.append(item)
    return items


def _make_dispatcher_with_buffer(
    n_channels: int,
    capacity: int,
    sample_rate: float,
    filter_settings: FilterSettings,
):
    """Create a Dispatcher with linked SharedRingBuffer for testing."""
    raw_queue = queue.Queue()
    visualization_queue = queue.Queue()
    audio_queue = queue.Queue()
    logging_queue = queue.Queue()
    event_queue = queue.Queue()
    
    # Create source buffer (channels x samples)
    source_buffer = SharedRingBuffer((n_channels, capacity), dtype=np.float32)
    
    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        audio_queue,
        logging_queue,
        event_queue,
        filter_settings=filter_settings,
    )
    
    # Link the source buffer
    dispatcher.set_source_buffer(source_buffer, sample_rate=sample_rate)
    
    return dispatcher, source_buffer, {
        "raw": raw_queue,
        "visualization": visualization_queue,
        "audio": audio_queue,
        "logging": logging_queue,
        "event": event_queue,
    }


def test_dispatcher_filters_dc_and_notch_at_target_frequency():
    """Test that AC coupling removes DC offset and notch filter attenuates target frequency."""
    fs = 1_000.0
    dt = 1.0 / fs
    duration = 1.0
    frames = int(duration * fs)
    t = np.arange(frames, dtype=np.float32) * dt
    
    dc_offset = 0.75
    notch_freq = 60.0
    sine = np.sin(2.0 * math.pi * notch_freq * t)
    raw_signal = (dc_offset + sine).astype(np.float32)
    
    # Shape: (1 channel, frames samples)
    samples = raw_signal.reshape(1, -1)
    
    settings = FilterSettings(
        default=ChannelFilterSettings(
            ac_couple=True,
            ac_cutoff_hz=1.0,
            notch_enabled=True,
            notch_freq_hz=notch_freq,
            notch_q=35.0,
        )
    )
    
    dispatcher, source_buffer, queues = _make_dispatcher_with_buffer(
        n_channels=1, capacity=frames * 2, sample_rate=fs, filter_settings=settings
    )
    
    # Write samples to the source buffer
    start_index = source_buffer.write(samples)
    
    # Create ChunkPointer referencing the written data
    pointer = ChunkPointer(start_index=start_index, length=frames, render_time=0.0, seq=0, start_sample=0)
    
    dispatcher.start()
    queues["raw"].put(pointer)
    queues["raw"].put(EndOfStream)
    dispatcher.join(timeout=2.0)
    
    # Visualization queue should contain ChunkPointers (to viz_buffer)
    viz_items = _drain_chunks(queues["visualization"])
    assert len(viz_items) == 1
    assert isinstance(viz_items[0], ChunkPointer)
    
    # Read filtered data from viz_buffer
    viz_pointer = viz_items[0]
    filtered = dispatcher.viz_buffer.read(viz_pointer.start_index, viz_pointer.length)
    
    # AC coupling should remove DC - mean should be near zero
    mean_val = abs(float(np.mean(filtered)))
    assert mean_val < 0.05, f"DC not removed: mean={mean_val}"
    
    # Notch filter should attenuate 60 Hz
    # Compare against signal processed without notch
    conditioner_no_notch = SignalConditioner(
        FilterSettings(default=ChannelFilterSettings(ac_couple=True, ac_cutoff_hz=1.0))
    )
    chunk_for_comparison = Chunk(
        samples=samples, start_time=0.0, dt=dt, seq=0,
        channel_names=("ch0",), units="V", meta={}
    )
    baseline = conditioner_no_notch.process(chunk_for_comparison)[0]
    baseline -= np.mean(baseline)
    baseline_rms = float(np.sqrt(np.mean(baseline**2)))
    
    filtered_flat = filtered.flatten() if filtered.ndim > 1 else filtered
    filtered_rms = float(np.sqrt(np.mean(filtered_flat**2)))
    
    # Notch should reduce RMS significantly (< 50% of without notch)
    assert filtered_rms < baseline_rms * 0.5, f"Notch ineffective: {filtered_rms} vs {baseline_rms}"
    
    # Note: Logging queue requires set_recording_enabled(True)


def test_dispatcher_preserves_filter_state_across_chunks():
    """Test that filter state is preserved across chunk boundaries."""
    fs = 2_000.0
    dt = 1.0 / fs
    total_frames = 200
    
    # Create impulse signal
    impulse = np.zeros(total_frames, dtype=np.float32)
    impulse[10] = 1.0
    
    first_len = 120
    second_len = total_frames - first_len
    
    settings = FilterSettings(
        default=ChannelFilterSettings(lowpass_hz=200.0, lowpass_order=4)
    )
    
    dispatcher, source_buffer, queues = _make_dispatcher_with_buffer(
        n_channels=1, capacity=total_frames * 2, sample_rate=fs, filter_settings=settings
    )
    
    # Write first chunk
    chunk1_samples = impulse[:first_len].reshape(1, -1)
    start1 = source_buffer.write(chunk1_samples)
    pointer1 = ChunkPointer(start_index=start1, length=first_len, render_time=0.0, seq=0, start_sample=0)
    
    # Write second chunk
    chunk2_samples = impulse[first_len:].reshape(1, -1)
    start2 = source_buffer.write(chunk2_samples)
    pointer2 = ChunkPointer(start_index=start2, length=second_len, render_time=first_len * dt, seq=1, start_sample=first_len)
    
    dispatcher.start()
    queues["raw"].put(pointer1)
    queues["raw"].put(pointer2)
    queues["raw"].put(EndOfStream)
    dispatcher.join(timeout=2.0)
    
    # Get filtered outputs
    viz_items = _drain_chunks(queues["visualization"])
    assert len(viz_items) == 2
    
    # Read and combine filtered data
    combined = np.concatenate([
        dispatcher.viz_buffer.read(p.start_index, p.length).flatten()
        for p in viz_items
    ])
    
    # Compare to single-chunk processing
    conditioner = SignalConditioner(settings)
    full_chunk = Chunk(
        samples=impulse.reshape(1, -1), start_time=0.0, dt=dt, seq=0,
        channel_names=("ch0",), units="V", meta={}
    )
    expected = conditioner.process(full_chunk).flatten()
    
    # Should match closely (filter state preserved)
    assert np.allclose(combined, expected, atol=1e-4), "Filter state not preserved across chunks"


def test_dispatcher_fan_out_and_backpressure_tracking():
    """Test that dispatcher fans out to all queues and tracks backpressure."""
    settings = FilterSettings()
    
    # Use small queue sizes to trigger backpressure
    raw_queue = queue.Queue()
    visualization_queue = queue.Queue(maxsize=1)
    audio_queue = queue.Queue(maxsize=1)
    logging_queue = queue.Queue()
    event_queue = queue.Queue()
    
    fs = 1_000.0
    frames = 16
    n_chunks = 3
    
    source_buffer = SharedRingBuffer((1, frames * n_chunks * 2), dtype=np.float32)
    
    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        audio_queue,
        logging_queue,
        event_queue,
        filter_settings=settings,
    )
    dispatcher.set_source_buffer(source_buffer, sample_rate=fs)
    
    # Register analysis queue
    # Use maxsize=2 so we can retain at least one chunk + EndOfStream
    # (If maxsize=1, the final EOS will evict the last data chunk)
    analysis_queue = queue.Queue(maxsize=2)
    token = dispatcher.register_analysis_queue(analysis_queue)
    
    dispatcher.start()
    
    # Send multiple chunks
    pointers = []
    for seq in range(n_chunks):
        value = float(seq + 1)
        samples = np.ones((1, frames), dtype=np.float32) * value
        start = source_buffer.write(samples)
        pointer = ChunkPointer(start_index=start, length=frames, render_time=seq * frames / fs, seq=seq, start_sample=seq * frames)
        pointers.append(pointer)
        raw_queue.put(pointer)
    
    raw_queue.put(EndOfStream)
    dispatcher.join(timeout=2.0)
    
    stats = dispatcher.snapshot()
    assert stats["received"] == n_chunks
    assert stats["processed"] == n_chunks
    
    forwarded = stats["forwarded"]
    dropped = stats["dropped"]
    
    # With small queues (maxsize=1), we expect:
    # - Some chunks to be forwarded
    # - Some chunks to be evicted (visualization uses drop-oldest policy)
    # - Dropped count should be 0 for visualization (as it evicts instead of drops)
    assert forwarded.get("visualization", 0) >= 1, "Should forward at least 1 viz chunk"
    assert stats["evicted"].get("visualization", 0) >= 1, "Should evict at least 1 viz chunk (backpressure)"

    
    # Analysis queue receives all data chunks + EOS (EOS forwarded via drop-oldest)
    # n_chunks data items + 1 EOS = n_chunks + 1 forwarded
    assert forwarded.get("analysis") == n_chunks + 1
    
    # Verify data was received by analysis (at least 1 chunk - queue may evict with maxsize=2)
    analysis_received = _drain_chunks(analysis_queue)
    assert len(analysis_received) >= 1, "Should receive at least the last chunk"
    assert isinstance(analysis_received[0], Chunk), "Analysis queue should receive Chunk objects, not Pointers"
    
    # Note: Logging queue requires set_recording_enabled(True)
    
    dispatcher.unregister_analysis_queue(token)


def test_dispatcher_eos_force_delivery_to_analysis():
    """Test that EndOfStream is delivered to analysis queue even if full, by evicting old items."""
    settings = FilterSettings()
    raw_queue = queue.Queue()
    # Dummy queues
    visualization_queue = queue.Queue()
    audio_queue = queue.Queue()
    logging_queue = queue.Queue()
    event_queue = queue.Queue()

    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        audio_queue,
        logging_queue,
        event_queue,
        filter_settings=settings,
    )
    
    # Create an analysis queue with very limited capacity
    analysis_queue = queue.Queue(maxsize=1)
    dispatcher.register_analysis_queue(analysis_queue)
    
    # Fill the queue manually
    analysis_queue.put(Chunk(
        samples=np.zeros((1, 10), dtype=np.float32),
        start_time=0.0,
        dt=0.01,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={}
    ))
    assert analysis_queue.full()
    
    # Call internal _broadcast_end_of_stream directly to verify logic without thread timing noise
    # (Though dispatcher.stop() calls this too)
    dispatcher._broadcast_end_of_stream()
    
    # Verify behavior:
    # 1. The old item should have been evicted (or at least EOS should be there)
    # Since maxsize=1, the queue presumably now contains ONLY EndOfStream if it evicted 1 and put 1.
    # If it evicted multiple times or logic differs, we just ensure we can get EndOfStream.
    
    item = analysis_queue.get_nowait()
    assert item is EndOfStream
    
    # Verify stats
    stats = dispatcher.snapshot()
    # Should have at least 1 eviction in 'analysis'
    assert stats["evicted"].get("analysis", 0) >= 1


def test_dispatcher_eos_delivery_on_unregister_full_queue():
    """Test that EndOfStream is delivered when unregistering a full analysis queue."""
    settings = FilterSettings()
    raw_queue = queue.Queue()
    visualization_queue = queue.Queue()
    audio_queue = queue.Queue()
    logging_queue = queue.Queue()
    event_queue = queue.Queue()

    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        audio_queue,
        logging_queue,
        event_queue,
        filter_settings=settings,
    )
    
    # Create analysis queue with maxsize=1
    analysis_queue = queue.Queue(maxsize=1)
    token = dispatcher.register_analysis_queue(analysis_queue)
    
    # Fill it
    analysis_queue.put(Chunk(
        samples=np.zeros((1, 10), dtype=np.float32),
        start_time=0.0,
        dt=0.01,
        seq=0,
        channel_names=("ch0",),
        units="V",
        meta={}
    ))
    assert analysis_queue.full()
    
    # Unregister - should force EOS
    dispatcher.unregister_analysis_queue(token)
    
    # Verify EOS is prioritized (evicting data)
    item = analysis_queue.get_nowait()
    assert item is EndOfStream
    
    stats = dispatcher.snapshot()
    assert stats["evicted"].get("analysis", 0) >= 1

def test_eos_delivery_with_drop_policy():
    """Test that EndOfStream is delivered to drop-oldest queues (e.g. visualization) even if full."""
    settings = FilterSettings()
    raw_queue = queue.Queue()
    # Use maxsize=2 to verify we keep some data (newest) but still deliver EOS
    visualization_queue = queue.Queue(maxsize=2)
    audio_queue = queue.Queue()
    logging_queue = queue.Queue()
    event_queue = queue.Queue()

    # Verify policy is indeed drop-oldest
    from shared.models import QUEUE_POLICIES
    assert QUEUE_POLICIES["visualization"] == "drop-oldest"

    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        audio_queue,
        logging_queue,
        event_queue,
        filter_settings=settings,
    )
    
    # Fill queue manually so it is full: [Old, New]
    visualization_queue.put(ChunkPointer(0, 100, 0.0, 0, 0)) # Old
    visualization_queue.put(ChunkPointer(100, 100, 0.1, 1, 100)) # New
    assert visualization_queue.full()
    
    # Broadcast EOS
    # Should force drop-oldest behavior: Evict 'Old', keep 'New', add 'EOS'
    # Since visualization is now drop-oldest natively, this happens automatically
    dispatcher._broadcast_end_of_stream()
    
    # 1. First item should be 'New' (seq=1)
    item1 = visualization_queue.get_nowait()
    assert isinstance(item1, ChunkPointer)
    assert item1.seq == 1
    
    # 2. Second item should be EndOfStream
    item2 = visualization_queue.get_nowait()
    assert item2 is EndOfStream


def test_eos_delivery_to_events_queue_when_full():
    """Test that EndOfStream is delivered to events queue (drop-oldest) even when full.
    
    The events queue uses drop-oldest policy, which natively supports EOS delivery
    by evicting the oldest item to make room for EOS. This ensures EOS is always
    delivered for clean shutdown.
    """
    settings = FilterSettings()
    raw_queue = queue.Queue()
    visualization_queue = queue.Queue()
    audio_queue = queue.Queue()
    logging_queue = queue.Queue()
    # Small events queue to test backpressure
    event_queue = queue.Queue(maxsize=2)

    # Verify policy is indeed drop-oldest (for freshness)
    from shared.models import QUEUE_POLICIES, DetectionEvent
    assert QUEUE_POLICIES["events"] == "drop-oldest"

    dispatcher = Dispatcher(
        raw_queue,
        visualization_queue,
        audio_queue,
        logging_queue,
        event_queue,
        filter_settings=settings,
    )
    
    # Fill events queue with detection events
    dummy_event1 = DetectionEvent(t=0.0, chan=0, window=np.zeros(10))
    dummy_event2 = DetectionEvent(t=0.1, chan=0, window=np.zeros(10))
    event_queue.put(dummy_event1)
    event_queue.put(dummy_event2)
    assert event_queue.full()
    
    # Broadcast EOS - should force drop-oldest behavior for events queue
    dispatcher._broadcast_end_of_stream()
    
    # EOS should be in the queue (oldest event evicted)
    # Queue should contain: [dummy_event2, EndOfStream]
    item1 = event_queue.get_nowait()
    assert isinstance(item1, DetectionEvent)
    assert item1.t == 0.1  # This is dummy_event2 (newest kept)
    
    item2 = event_queue.get_nowait()
    assert item2 is EndOfStream
