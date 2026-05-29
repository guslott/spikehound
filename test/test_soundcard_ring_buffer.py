"""Regression tests for LocalRingBuffer overrun accounting (finding 6.1)."""
from __future__ import annotations

import numpy as np

from daq.soundcard_source import LocalRingBuffer


def _block(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def test_no_overrun_returns_zero_and_no_drops() -> None:
    buf = LocalRingBuffer(capacity=10, channels=1, dtype=np.dtype(np.float32))
    assert buf.write(_block(range(8))) == 0
    assert buf.dropped_frames == 0
    assert buf.overruns == 0
    assert buf.filled == 8


def test_overrun_counts_dropped_frames_and_keeps_newest() -> None:
    """When the producer outpaces the consumer, the overwritten unread frames
    must be counted (not dropped silently), and the buffer must retain the most
    recent `capacity` frames."""
    buf = LocalRingBuffer(capacity=10, channels=1, dtype=np.dtype(np.float32))
    buf.write(_block([1, 1, 1, 1, 1, 1, 1, 1]))  # filled = 8

    # 8 + 5 = 13 > capacity(10): the 3 oldest unread frames are overwritten.
    dropped = buf.write(_block([10, 11, 12, 13, 14]))
    assert dropped == 3
    assert buf.overruns == 1
    assert buf.dropped_frames == 3
    assert buf.filled == 10

    # The retained 10 frames are the most recent: 5 surviving 1's + the 5 new.
    out = buf.read(10).reshape(-1)
    np.testing.assert_array_equal(
        out, np.array([1, 1, 1, 1, 1, 10, 11, 12, 13, 14], dtype=np.float32)
    )


def test_oversized_block_counts_all_lost_frames() -> None:
    """A single block larger than the whole buffer drops every buffered frame
    plus the head of the block that doesn't fit."""
    buf = LocalRingBuffer(capacity=10, channels=1, dtype=np.dtype(np.float32))
    buf.write(_block(range(6)))  # filled = 6

    # 25-frame block, capacity 10: dropped = filled(6) + (25 - 10) = 21.
    dropped = buf.write(_block(range(25)))
    assert dropped == 21
    assert buf.overruns == 1
    assert buf.dropped_frames == 21
    assert buf.filled == 10

    # Only the last 10 frames of the oversized block survive.
    out = buf.read(10).reshape(-1)
    np.testing.assert_array_equal(out, np.arange(15, 25, dtype=np.float32))


def test_overruns_accumulate_across_writes() -> None:
    buf = LocalRingBuffer(capacity=4, channels=1, dtype=np.dtype(np.float32))
    buf.write(_block([1, 2, 3, 4]))      # full, no drop
    assert buf.write(_block([5, 6])) == 2  # overrun: 4+2-4 = 2 dropped
    assert buf.write(_block([7, 8])) == 2  # overrun again
    assert buf.overruns == 2
    assert buf.dropped_frames == 4
