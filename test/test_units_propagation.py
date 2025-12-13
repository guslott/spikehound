
import queue
import numpy as np
import pytest
from core import Dispatcher, FilterSettings, Chunk, EndOfStream
from shared.models import ChunkPointer
from shared.ring_buffer import SharedRingBuffer

def _make_dispatcher(n_channels=1, sample_rate=1000.0):
    raw_queue = queue.Queue()
    viz_queue = queue.Queue()
    audio_queue = queue.Queue()
    log_queue = queue.Queue()
    event_queue = queue.Queue()
    
    buf = SharedRingBuffer((n_channels, 1000), dtype=np.float32)
    dispatcher = Dispatcher(
        raw_queue, viz_queue, audio_queue, log_queue, event_queue,
        filter_settings=FilterSettings()
    )
    dispatcher.set_source_buffer(buf, sample_rate=sample_rate)
    return dispatcher, buf, raw_queue, viz_queue

def test_chunk_units_preserved():
    """Test that uniform units are propagated to Chunk objects."""
    dispatcher, buf, raw_q, viz_q = _make_dispatcher(n_channels=1)
    
    # Set layout with "mV"
    dispatcher.set_channel_layout([0], ["ch0"], ["mV"])
    
    analysis_q = queue.Queue()
    dispatcher.register_analysis_queue(analysis_q)
    
    # Push data
    start = buf.write(np.zeros((1, 10), dtype=np.float32))
    ptr = ChunkPointer(start_index=start, length=10, render_time=0.0)
    
    dispatcher.start()
    raw_q.put(ptr)
    raw_q.put(EndOfStream)
    dispatcher.join(timeout=2.0)
    
    # Check analysis queue for Chunk
    try:
        chunk = analysis_q.get(timeout=1.0)
        assert isinstance(chunk, Chunk)
        assert chunk.units == "mV"
    except queue.Empty:
        pytest.fail("Analysis queue empty")

def test_chunk_units_mixed():
    """Test that mixed units result in 'mixed' unit string."""
    dispatcher, buf, raw_q, viz_q = _make_dispatcher(n_channels=2)
    
    # Set layout with mixed units
    dispatcher.set_channel_layout([0, 1], ["ch0", "ch1"], ["V", "mV"])
    
    analysis_q = queue.Queue()
    dispatcher.register_analysis_queue(analysis_q)
    
    # Push data
    start = buf.write(np.zeros((2, 10), dtype=np.float32))
    ptr = ChunkPointer(start_index=start, length=10, render_time=0.0)
    
    dispatcher.start()
    raw_q.put(ptr)
    raw_q.put(EndOfStream)
    dispatcher.join(timeout=2.0)
    
    try:
        chunk = analysis_q.get(timeout=1.0)
        assert isinstance(chunk, Chunk)
        assert chunk.units == "mixed"
    except queue.Empty:
        pytest.fail("Analysis queue empty")

def test_chunk_units_default():
    """Test that missing units fall back to 'unknown'."""
    dispatcher, buf, raw_q, viz_q = _make_dispatcher(n_channels=1)
    
    # Set layout without units (should use default None -> unknown)
    dispatcher.set_channel_layout([0], ["ch0"])
    
    analysis_q = queue.Queue()
    dispatcher.register_analysis_queue(analysis_q)
    
    start = buf.write(np.zeros((1, 10), dtype=np.float32))
    ptr = ChunkPointer(start_index=start, length=10, render_time=0.0)
    
    dispatcher.start()
    raw_q.put(ptr)
    raw_q.put(EndOfStream)
    dispatcher.join(timeout=2.0)
    
    try:
        chunk = analysis_q.get(timeout=1.0)
        assert isinstance(chunk, Chunk)
        assert chunk.units == "unknown"
    except queue.Empty:
        pytest.fail("Analysis queue empty")
