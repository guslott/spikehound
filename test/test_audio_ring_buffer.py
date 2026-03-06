"""
Tests for the AudioPlayer output ring buffer.

The ring buffer (audio/player.py) must correctly distinguish a *full* buffer
from an *empty* one.  The original code used a sentinel-slot approach
(``_ring_space`` subtracted 1) which worked in practice but:

  * Wasted one slot of usable capacity.
  * Made ``space + available`` equal ``size - 1`` rather than ``size``,
    creating an invisible asymmetry.
  * Could misreport a completely-full buffer as empty if ``_r_head``
    wrapped to equal ``_r_tail`` — an ambiguity the sentinel avoids by
    construction, but which is extremely non-obvious to future contributors.

The fix replaces the sentinel with an explicit ``_r_count`` fill-level
counter.  These tests verify:

  1. Basic read / write invariants.
  2. The critical full-vs-empty distinction (regression for the original bug).
  3. Wrap-around correctness.
  4. Drop-oldest semantics when the caller overflows the ring.
  5. Data integrity across the full lifecycle.
  6. Thread-safety smoke test.
"""
from __future__ import annotations

import threading
import queue
import unittest.mock as mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: extract ring-buffer internals without starting audio hardware
# ---------------------------------------------------------------------------

def _make_player(ring_size: int = 16) -> "AudioPlayerRingOnly":
    """
    Return an AudioPlayer-like object whose ring buffer has exactly
    *ring_size* slots.  We skip miniaudio entirely (it may not be present
    in CI) by patching the import guard and calling __init__ with a real
    queue but never calling .run().
    """
    import sys
    import types

    # Stub out miniaudio so the import inside player.py succeeds without hardware.
    if "miniaudio" not in sys.modules:
        stub = types.ModuleType("miniaudio")
        stub.PlaybackDevice = mock.MagicMock()
        stub.SampleFormat = mock.MagicMock()
        stub.Devices = mock.MagicMock()
        sys.modules["miniaudio"] = stub

    from audio.player import AudioPlayer, AudioConfig

    cfg = AudioConfig(ring_seconds=0.0)   # will be overridden by blocksize
    # Force ring_len == ring_size by setting blocksize = ring_size // 4 and
    # in_sr small enough that int(in_sr * ring_seconds) < blocksize * 4.
    blocksize = ring_size // 4 if ring_size >= 8 else ring_size
    cfg.blocksize = blocksize
    cfg.ring_seconds = 0.0          # so int(in_sr * 0.0) = 0 < blocksize * 4
    in_sr = 1000

    player = AudioPlayer(
        audio_queue=queue.Queue(),
        input_samplerate=in_sr,
        config=cfg,
    )
    # Directly override ring to be exactly ring_size slots.
    player._ring = np.zeros(ring_size, dtype=np.float32)
    player._r_head = 0
    player._r_tail = 0
    player._r_count = 0
    return player


# ---------------------------------------------------------------------------
# Group 1 — Basic invariants
# ---------------------------------------------------------------------------

class TestRingBufferBasicInvariants:
    """The ring's capacity, space, and available counters must always be
    consistent: space + available == ring.size."""

    def test_initial_state_empty(self):
        p = _make_player(16)
        assert p._ring_available() == 0
        assert p._ring_space() == 16

    def test_space_plus_available_equals_size_after_write(self):
        p = _make_player(16)
        data = np.ones(7, dtype=np.float32)
        p._ring_write(data)
        assert p._ring_space() + p._ring_available() == p._ring.size

    def test_space_plus_available_equals_size_after_partial_read(self):
        p = _make_player(16)
        p._ring_write(np.ones(10, dtype=np.float32))
        p._ring_read(4)
        assert p._ring_space() + p._ring_available() == p._ring.size

    def test_write_then_read_full_roundtrip(self):
        p = _make_player(16)
        data = np.arange(10, dtype=np.float32)
        p._ring_write(data)
        out = p._ring_read(10)
        np.testing.assert_array_equal(out, data)
        assert p._ring_available() == 0
        assert p._ring_space() == 16

    def test_read_from_empty_returns_empty_array(self):
        p = _make_player(16)
        out = p._ring_read(8)
        assert out.size == 0

    def test_read_more_than_available_returns_only_available(self):
        p = _make_player(16)
        p._ring_write(np.ones(5, dtype=np.float32))
        out = p._ring_read(100)
        assert out.size == 5

    def test_write_zero_samples_is_noop(self):
        p = _make_player(16)
        p._ring_write(np.array([], dtype=np.float32))
        assert p._ring_available() == 0
        assert p._ring_space() == 16


# ---------------------------------------------------------------------------
# Group 2 — Critical: full-vs-empty distinction  (regression for the bug)
# ---------------------------------------------------------------------------

class TestFullVsEmptyDistinction:
    """
    The original sentinel approach prevented head == tail when full by
    construction, at the cost of wasting one slot and creating an asymmetry
    that was invisible to readers of the code.  The count-based fix must
    handle the pathological case explicitly: write exactly *ring.size* samples
    so that _r_head wraps back to equal _r_tail, and verify that the buffer
    reports full (available == size), not empty (available == 0).
    """

    def test_fill_to_exact_capacity_reports_full_not_empty(self):
        size = 16
        p = _make_player(size)
        p._ring_write(np.ones(size, dtype=np.float32))
        # head has wrapped to equal tail; count must say "full", not "empty"
        assert p._r_head == p._r_tail, "precondition: head == tail after full fill"
        assert p._ring_available() == size, (
            "full buffer with head==tail was misreported as empty (the bug)"
        )
        assert p._ring_space() == 0

    def test_fill_to_capacity_all_data_readable(self):
        size = 16
        p = _make_player(size)
        data = np.arange(size, dtype=np.float32)
        p._ring_write(data)
        out = p._ring_read(size)
        np.testing.assert_array_equal(out, data)

    def test_drain_full_buffer_reports_empty(self):
        size = 16
        p = _make_player(size)
        p._ring_write(np.ones(size, dtype=np.float32))
        p._ring_read(size)
        assert p._ring_available() == 0
        assert p._ring_space() == size
        assert p._r_head == p._r_tail  # still equal, but now empty

    def test_head_equals_tail_is_not_misidentified_after_wrap(self):
        """
        Interleave writes and reads so that _r_head wraps past the end and
        catches up to _r_tail.  Verify that _r_count always reflects the true
        fill level regardless of whether head == tail at that instant.
        """
        size = 8
        p = _make_player(size)
        # Fill completely so head wraps to equal tail (== 0 after full fill).
        p._ring_write(np.ones(size, dtype=np.float32))
        assert p._r_head == p._r_tail, "precondition: full ring has head == tail"
        assert p._ring_available() == size, "full ring: available must be size, not 0"
        assert p._ring_space() == 0

        # Now read half, then write to a state where head != tail — confirm count.
        p._ring_read(4)
        assert p._ring_available() == 4
        p._ring_write(np.full(2, 2.0, dtype=np.float32))
        assert p._ring_available() == 6   # 4 + 2 = 6 unread samples
        assert p._ring_space() == 2       # 2 slots still free


# ---------------------------------------------------------------------------
# Group 3 — Wrap-around correctness
# ---------------------------------------------------------------------------

class TestWrapAround:
    """Data must survive the ring wrapping past index 0."""

    def test_data_survives_head_wrap(self):
        size = 8
        p = _make_player(size)
        # Prime head near the end of the array.
        p._ring_write(np.zeros(6, dtype=np.float32))
        p._ring_read(6)                          # head=6, tail=6
        sentinel = np.array([1, 2, 3, 4], dtype=np.float32)
        p._ring_write(sentinel)                  # wraps: writes [6,7,0,1]
        out = p._ring_read(4)
        np.testing.assert_array_equal(out, sentinel)

    def test_multiple_wrap_cycles(self):
        size = 8
        p = _make_player(size)
        for cycle in range(5):
            data = np.full(5, float(cycle), dtype=np.float32)
            p._ring_write(data)
            out = p._ring_read(5)
            np.testing.assert_array_equal(out, data, err_msg=f"cycle {cycle}")

    def test_write_wraps_tail_past_end(self):
        """
        Bring tail close to the end of the array, then overfill to force a
        drop-oldest that wraps tail past index 0.
        """
        size = 8
        p = _make_player(size)
        p._ring_write(np.zeros(6, dtype=np.float32))
        p._ring_read(6)                          # head=6, tail=6, empty
        # Write 9 samples (> size=8): clamped to 8, then drop-oldest advances tail.
        payload = np.arange(9, dtype=np.float32)
        p._ring_write(payload)
        assert p._ring_available() == size
        # The data in the ring must be the *last* size samples of payload.
        out = p._ring_read(size)
        np.testing.assert_array_equal(out, payload[-size:])


# ---------------------------------------------------------------------------
# Group 4 — Drop-oldest semantics
# ---------------------------------------------------------------------------

class TestDropOldest:
    """When the ring overflows, the oldest samples are discarded; the newest
    samples arrive intact."""

    def test_overflow_by_one_drops_oldest_one(self):
        size = 8
        p = _make_player(size)
        first = np.arange(8, dtype=np.float32)
        p._ring_write(first)                     # fill exactly
        extra = np.array([99.0], dtype=np.float32)
        p._ring_write(extra)                     # overflow by 1 → drop oldest
        out = p._ring_read(size)
        # Ring should contain first[1:] + [99.0]
        expected = np.concatenate([first[1:], extra])
        np.testing.assert_array_equal(out, expected)

    def test_overflow_keeps_newest_samples(self):
        size = 8
        p = _make_player(size)
        old = np.zeros(8, dtype=np.float32)
        new = np.arange(1, 9, dtype=np.float32)  # 8 fresh samples
        p._ring_write(old)
        p._ring_write(new)                       # overwrites everything
        out = p._ring_read(size)
        np.testing.assert_array_equal(out, new)

    def test_oversized_write_clamped_to_ring_size(self):
        """A write larger than ring.size is trimmed to the *last* ring.size
        samples — not silently corrupted."""
        size = 8
        p = _make_player(size)
        big = np.arange(20, dtype=np.float32)
        p._ring_write(big)
        out = p._ring_read(size)
        np.testing.assert_array_equal(out, big[-size:])

    def test_count_is_correct_after_drop_oldest(self):
        size = 8
        p = _make_player(size)
        p._ring_write(np.ones(6, dtype=np.float32))
        p._ring_write(np.ones(5, dtype=np.float32))  # total 11 > 8 → drop 3
        assert p._ring_available() == size
        assert p._ring_space() == 0
        assert p._r_count == size

    def test_repeated_overflow_count_never_exceeds_size(self):
        size = 8
        p = _make_player(size)
        for _ in range(20):
            p._ring_write(np.ones(size, dtype=np.float32))
            assert p._r_count == size
            assert p._ring_space() == 0


# ---------------------------------------------------------------------------
# Group 5 — Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    """Sample values are preserved exactly through the ring."""

    def test_sequential_values_preserved(self):
        size = 16
        p = _make_player(size)
        data = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
        p._ring_write(data)
        out = p._ring_read(12)
        np.testing.assert_array_almost_equal(out, data)

    def test_partial_reads_preserve_order(self):
        size = 16
        p = _make_player(size)
        data = np.arange(12, dtype=np.float32)
        p._ring_write(data)
        chunk1 = p._ring_read(4)
        chunk2 = p._ring_read(4)
        chunk3 = p._ring_read(4)
        np.testing.assert_array_equal(np.concatenate([chunk1, chunk2, chunk3]), data)

    def test_interleaved_writes_and_reads(self):
        """
        Exercise write/read interleaving without overflow: keep writes small
        enough that the ring never drops samples, then verify exact data order.
        """
        size = 16
        p = _make_player(size)
        rng = np.random.default_rng(42)
        reference: list[float] = []

        for _ in range(200):
            # Only write when there is guaranteed space (≤ 4 samples, leave slack).
            n_write = int(rng.integers(1, 4))
            if p._ring_space() >= n_write + 4:   # keep headroom to avoid drops
                chunk = rng.random(n_write).astype(np.float32)
                reference.extend(chunk.tolist())
                p._ring_write(chunk)
            # Read a small batch and verify against reference.
            if rng.random() > 0.4 and p._ring_available() >= 3:
                n_read = int(rng.integers(1, 4))
                out = p._ring_read(n_read)
                expected = reference[:out.size]
                del reference[:out.size]
                np.testing.assert_array_almost_equal(
                    out, expected, decimal=6,
                    err_msg="Sample order corrupted during interleaved read/write"
                )


# ---------------------------------------------------------------------------
# Group 6 — Thread-safety smoke test
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """One writer thread and one reader thread running concurrently must not
    corrupt the counter or produce out-of-range values."""

    def test_concurrent_write_read_no_corruption(self):
        size = 32
        p = _make_player(size)
        errors: list[str] = []

        def writer():
            for _ in range(500):
                p._ring_write(np.ones(4, dtype=np.float32))

        def reader():
            for _ in range(500):
                out = p._ring_read(4)
                if out.size > 0 and not np.all(np.isfinite(out)):
                    errors.append("non-finite sample detected")
                with p._r_lock:
                    if p._r_count < 0:
                        errors.append(f"negative count: {p._r_count}")
                    if p._r_count > p._ring.size:
                        errors.append(f"count exceeds size: {p._r_count}")

        t_w = threading.Thread(target=writer)
        t_r = threading.Thread(target=reader)
        t_w.start(); t_r.start()
        t_w.join(); t_r.join()
        assert errors == [], f"Thread-safety violations: {errors}"

    def test_count_invariant_never_violated_under_load(self):
        """_r_count must stay in [0, ring.size] at all times."""
        size = 16
        p = _make_player(size)
        violations: list[str] = []

        def stress():
            rng = np.random.default_rng()
            for _ in range(1000):
                if rng.random() > 0.5:
                    p._ring_write(rng.random(int(rng.integers(1, 8))).astype(np.float32))
                else:
                    p._ring_read(int(rng.integers(1, 8)))
                with p._r_lock:
                    if not (0 <= p._r_count <= p._ring.size):
                        violations.append(f"count={p._r_count} out of [0, {p._ring.size}]")

        threads = [threading.Thread(target=stress) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert violations == [], f"Count invariant violated: {violations[:5]}"
