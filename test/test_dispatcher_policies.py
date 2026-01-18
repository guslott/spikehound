
import pytest
import queue
import numpy as np
from core import Dispatcher, FilterSettings
from shared.models import ChunkPointer
from shared.ring_buffer import SharedRingBuffer

def _make_dispatcher(min_capacity=100, strict=False, gap_policy="crash"):
    raw_q = queue.Queue()
    viz_q = queue.Queue()
    aud_q = queue.Queue()
    log_q = queue.Queue()
    evt_q = queue.Queue()
    
    d = Dispatcher(
        raw_q, viz_q, aud_q, log_q, evt_q,
        strict_invariants=strict,
        gap_policy=gap_policy
    )
    return d, raw_q

def _setup_buffer(dispatcher, n_channels=1, capacity=1000, sr=1000.0):
    buf = SharedRingBuffer((n_channels, capacity), dtype=np.float32)
    dispatcher.set_source_buffer(buf, sample_rate=sr)
    return buf

def test_strict_invariants_raises():
    """Test that strict_invariants=True raises RuntimeError when source buffer is missing."""
    d, raw_q = _make_dispatcher(strict=True)
    
    # Do NOT set source buffer
    
    pointer = ChunkPointer(start_index=0, length=10, render_time=0.0, seq=0, start_sample=0)
    
    # Using internal method to avoid dealing with threads/queues for this unit test
    with pytest.raises(RuntimeError, match="Dispatcher Error: No source buffer linked"):
        d._process_pointer(pointer)

def test_loose_invariants_logs_only():
    """Test that strict_invariants=False (default) returns None when source buffer is missing."""
    d, raw_q = _make_dispatcher(strict=False)
    
    # Do NOT set source buffer
    
    pointer = ChunkPointer(start_index=0, length=10, render_time=0.0, seq=0, start_sample=0)
    
    result = d._process_pointer(pointer)
    assert result is None

def test_gap_policy_crash():
    """Test that gap_policy='crash' raises RuntimeError on sample gap."""
    d, raw_q = _make_dispatcher(gap_policy="crash")
    buf = _setup_buffer(d)
    
    # Process first chunk
    buf.write(np.zeros((1, 10), dtype=np.float32))
    ptr1 = ChunkPointer(start_index=0, length=10, render_time=0.0, seq=0, start_sample=0)
    d._process_pointer(ptr1)
    
    # Process second chunk with a GAP (start_sample=20 instead of 10)
    buf.write(np.zeros((1, 10), dtype=np.float32))
    ptr2 = ChunkPointer(start_index=10, length=10, render_time=0.02, seq=1, start_sample=20)
    
    with pytest.raises(RuntimeError, match="Sample gap detected"):
        d._process_pointer(ptr2)

def test_gap_policy_reset():
    """Test that gap_policy='reset' resets counters and continues."""
    d, raw_q = _make_dispatcher(gap_policy="reset")
    buf = _setup_buffer(d)
    
    # Process first chunk
    buf.write(np.zeros((1, 10), dtype=np.float32))
    ptr1 = ChunkPointer(start_index=0, length=10, render_time=0.0, seq=0, start_sample=0)
    d._process_pointer(ptr1)
    
    assert d._filled == 10
    
    # Process second chunk with a GAP
    buf.write(np.ones((1, 10), dtype=np.float32))
    ptr2 = ChunkPointer(start_index=10, length=10, render_time=0.02, seq=1, start_sample=20)
    
    # Should NOT raise
    d._process_pointer(ptr2)
    
    # Because of reset, previous 10 samples are gone. New 10 samples added.
    # _filled should be 10 (just the new chunk)
    assert d._filled == 10
    assert d._write_idx == 10  # effectively index 10 in output buffer now (wrote 10 starting at 0)
    assert d._stats.sample_gaps == 1

def test_gap_policy_ignore():
    """Test that gap_policy='ignore' continues without reset."""
    d, raw_q = _make_dispatcher(gap_policy="ignore")
    buf = _setup_buffer(d)
    
    # Process first chunk
    buf.write(np.zeros((1, 10), dtype=np.float32))
    ptr1 = ChunkPointer(start_index=0, length=10, render_time=0.0, seq=0, start_sample=0)
    d._process_pointer(ptr1)
    
    assert d._filled == 10
    
    # Process second chunk with a GAP
    buf.write(np.zeros((1, 10), dtype=np.float32))
    ptr2 = ChunkPointer(start_index=10, length=10, render_time=0.02, seq=1, start_sample=20)
    
    d._process_pointer(ptr2)
    
    # Should just append. 10 + 10 = 20 filled
    assert d._filled == 20
    assert d._stats.sample_gaps == 1
