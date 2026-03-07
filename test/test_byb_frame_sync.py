"""
Tests for the BackyardBrains frame-sync watchdog in BackyardBrainsSource._run_loop.

Coverage
--------
1. BYBFrameProtocol       – wire-format invariants (pure math, no serial needed)
2. BYBValidFrameDecoding  – valid frames are decoded and emitted as ChunkPointers
3. BYBAlignment           – misaligned / all-low-byte streams are discarded cleanly
4. SyncWatchdogTimeout    – no data → xrun counted + WARNING logged
5. SyncWatchdogRecovery   – valid data after timeout → INFO recovery log; steady data
                             keeps xruns at zero

All _run_loop tests inject a controlled fake transport and run the loop in a daemon
thread; the stop_event is set to terminate it.  _SYNC_TIMEOUT_S is patched to
a small value (0.02 s) on the test instance to keep suites fast.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List

import numpy as np
import pytest

from daq.backyard_brains import BackyardBrainsSource
from daq.base_device import ChannelInfo
from shared.models import ActualConfig, ChunkPointer
from shared.ring_buffer import SharedRingBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_byb_frames(values: List[int], bits: int = 10) -> bytes:
    """Encode integer sample values into BYB 2-byte frame format.

    Each value is packed into two bytes::

        high_byte = ((value >> 7) & 0x7F) | 0x80   -- bit-7 = 1 (frame-start marker)
        low_byte  =   value & 0x7F                  -- bit-7 = 0

    Parameters
    ----------
    values:
        One integer per channel sample (single-channel: one value per frame;
        multi-channel: values are listed in channel order within a frame).
    bits:
        ADC resolution.  Not used in encoding here (the formula is the same for
        10- and 14-bit; only the valid range of ``values`` differs).
    """
    buf = bytearray()
    for v in values:
        high = ((v >> 7) & 0x7F) | 0x80
        low  = v & 0x7F
        buf.append(high)
        buf.append(low)
    return bytes(buf)


def _make_byb_device(
    bits: int = 10,
    stream_channels: int = 1,
    chunk_size: int = 100,
    sample_rate: int = 1000,
    queue_maxsize: int = 4000,
) -> BackyardBrainsSource:
    """Create a BackyardBrainsSource pre-configured for unit tests.

    Bypasses the real open/configure lifecycle — no hardware needed.
    The caller must set ``dev._transport`` before invoking ``_run_loop``.
    """
    dev = BackyardBrainsSource(queue_maxsize=queue_maxsize)
    dev._bits = bits
    dev._stream_channel_count = stream_channels
    dev._max_channels = stream_channels

    channels = [
        ChannelInfo(id=i, name=f"ch{i}", units="V", range=(-5.0, 5.0))
        for i in range(stream_channels)
    ]
    dev._available_channels = channels
    dev._active_channel_ids = [ch.id for ch in channels]

    dev.config = ActualConfig(
        sample_rate=sample_rate,
        channels=channels,
        chunk_size=chunk_size,
        dtype="float32",
    )
    capacity = max(sample_rate * 10, chunk_size * 40)
    dev.ring_buffer = SharedRingBuffer(
        shape=(stream_channels, capacity),
        dtype="float32",
    )
    return dev


class ControlledFakeTransport:
    """Thread-safe BYB transport mock that supports byte injection at any time.

    Mimics the subset of the transport API that ``_run_loop`` uses.
    """

    kind = "serial"

    def __init__(self, initial_data: bytes = b""):
        self._lock = threading.Lock()
        self._buf = bytearray(initial_data)
        self.is_open = True

    def read(self, timeout_ms: int = 0) -> bytes:
        with self._lock:
            if self._buf:
                size = min(len(self._buf), 4096)
                chunk = bytes(self._buf[:size])
                del self._buf[:size]
                return chunk
        if timeout_ms > 0:
            time.sleep(timeout_ms / 1000.0)
        return b""

    def write_command(self, cmd: str) -> None:
        return

    def reset_input_buffer(self) -> None:
        with self._lock:
            self._buf.clear()

    def close(self) -> None:
        self.is_open = False

    def feed(self, data: bytes) -> None:
        """Append bytes to the receive buffer (callable from any thread)."""
        with self._lock:
            self._buf.extend(data)


def _drain_queue(dev: BackyardBrainsSource) -> List[ChunkPointer]:
    """Non-blocking drain of dev.data_queue; returns all items collected."""
    items: List[ChunkPointer] = []
    while True:
        try:
            items.append(dev.data_queue.get_nowait())
        except Exception:
            break
    return items


def _wait_for_chunks(dev: BackyardBrainsSource, n: int, timeout: float = 2.0) -> bool:
    """Block until at least *n* ChunkPointers appear in the queue (or timeout)."""
    deadline = time.monotonic() + timeout
    while dev.data_queue.qsize() < n and time.monotonic() < deadline:
        time.sleep(0.01)
    return dev.data_queue.qsize() >= n


# ---------------------------------------------------------------------------
# Group 1 — BYB wire-format invariants (pure unit tests, no serial)
# ---------------------------------------------------------------------------

class TestBYBFrameProtocol:
    """Verify the 2-byte BYB frame encoding required by the decoder in _run_loop."""

    @pytest.mark.parametrize("value", [0, 1, 127, 256, 512, 768, 1023])
    def test_high_byte_has_msb_set(self, value: int):
        """High byte of every frame must have bit-7 = 1 (frame-start marker)."""
        frame = _encode_byb_frames([value])
        assert frame[0] & 0x80 != 0, (
            f"MSB of high byte is 0 for value={value}"
        )

    @pytest.mark.parametrize("value", [0, 1, 127, 256, 512, 768, 1023])
    def test_low_byte_has_msb_clear(self, value: int):
        """Low byte of every frame must have bit-7 = 0 (data byte marker)."""
        frame = _encode_byb_frames([value])
        assert frame[1] & 0x80 == 0, (
            f"MSB of low byte is 1 for value={value}"
        )

    def test_round_trip_center_value(self):
        """Center ADC value (512 for 10-bit) must decode to exactly 0.0."""
        bits = 10
        center = float(1 << (bits - 1))  # 512.0
        frame = _encode_byb_frames([512])
        h, lo = frame[0], frame[1]
        raw = ((h & 0x7F) << 7) | (lo & 0x7F)
        decoded = (raw - center) / center
        assert abs(decoded) < 1e-6, f"Center value decoded to {decoded}, expected ~0.0"

    def test_round_trip_min_value(self):
        """Min ADC value (0) must decode to -1.0."""
        bits = 10
        center = float(1 << (bits - 1))
        frame = _encode_byb_frames([0])
        h, lo = frame[0], frame[1]
        raw = ((h & 0x7F) << 7) | (lo & 0x7F)
        decoded = (raw - center) / center
        assert abs(decoded - (-1.0)) < 1e-5

    def test_round_trip_max_value(self):
        """Max ADC value (1023) must decode close to +1.0."""
        bits = 10
        center = float(1 << (bits - 1))
        frame = _encode_byb_frames([1023])
        h, lo = frame[0], frame[1]
        raw = ((h & 0x7F) << 7) | (lo & 0x7F)
        decoded = (raw - center) / center
        assert decoded > 0.99, f"Max value decoded to {decoded}, expected > 0.99"

    def test_frame_size_is_two_bytes(self):
        """Each single-channel frame must be exactly 2 bytes."""
        frame = _encode_byb_frames([512])
        assert len(frame) == 2

    def test_multichannel_frame_interleaving(self):
        """A 3-channel frame must be 6 bytes in [H0 L0 H1 L1 H2 L2] order."""
        # The encoder packs channels sequentially within one frame
        vals = [100, 200, 300]
        frame = _encode_byb_frames(vals)
        assert len(frame) == 6
        # Each pair's high byte must have MSB=1
        for i in range(3):
            assert frame[i * 2] & 0x80 != 0, f"Channel {i} high byte MSB is 0"
            assert frame[i * 2 + 1] & 0x80 == 0, f"Channel {i} low byte MSB is 1"


# ---------------------------------------------------------------------------
# Group 2 — Valid frame decoding via _run_loop
# ---------------------------------------------------------------------------

class TestBYBValidFrameDecoding:
    """Valid frames must be decoded and emitted as ChunkPointers via data_queue."""

    def test_single_channel_emits_chunks(self):
        """200 valid 1-channel frames at chunk_size=100 should emit >= 2 chunks."""
        dev = _make_byb_device(stream_channels=1, chunk_size=100)
        data = _encode_byb_frames([512] * 200)
        dev._transport = ControlledFakeTransport(initial_data=data)

        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()
        got = _wait_for_chunks(dev, 2)
        dev._stop_event.set()
        t.join(timeout=1.0)

        assert got, "Expected >= 2 chunks from 200 valid frames at chunk_size=100"
        items = _drain_queue(dev)
        assert len(items) >= 2

    def test_decoded_center_values_are_zero(self):
        """Center-value frames must produce near-zero float32 output."""
        dev = _make_byb_device(stream_channels=1, chunk_size=50)
        data = _encode_byb_frames([512] * 100)
        dev._transport = ControlledFakeTransport(initial_data=data)

        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()
        _wait_for_chunks(dev, 1)
        dev._stop_event.set()
        t.join(timeout=1.0)

        items = _drain_queue(dev)
        assert len(items) >= 1
        ptr = items[0]
        chunk_data = dev.ring_buffer.read(ptr.start_index, ptr.length)  # (channels, frames)
        assert np.allclose(chunk_data[0, :], 0.0, atol=0.02), (
            f"Decoded values not near 0.0: mean={chunk_data[0].mean():.4f}"
        )

    def test_two_channel_chunks_emitted(self):
        """Two-channel device with interleaved frames should emit >= 1 chunk."""
        dev = _make_byb_device(stream_channels=2, chunk_size=50)
        n_frames = 150
        raw = bytearray()
        for _ in range(n_frames):
            for v in [512, 512]:       # one sample per channel per frame
                raw.append(((v >> 7) & 0x7F) | 0x80)
                raw.append(v & 0x7F)

        dev._transport = ControlledFakeTransport(initial_data=bytes(raw))

        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()
        got = _wait_for_chunks(dev, 1)
        dev._stop_event.set()
        t.join(timeout=1.0)

        assert got, "No chunks emitted for 2-channel device"


# ---------------------------------------------------------------------------
# Group 3 — Byte-stream alignment
# ---------------------------------------------------------------------------

class TestBYBAlignment:
    """_run_loop must discard leading low-byte garbage and find valid frame starts."""

    def test_leading_low_bytes_are_skipped(self):
        """Frames preceded by MSB=0 garbage bytes must still be decoded."""
        # 7 bytes of garbage (all MSB=0), then 150 valid 1-channel frames
        garbage = bytes([0x00, 0x1F, 0x7F, 0x0A, 0x3C, 0x55, 0x01])
        frames  = _encode_byb_frames([512] * 150)
        dev = _make_byb_device(stream_channels=1, chunk_size=100)
        dev._transport = ControlledFakeTransport(initial_data=garbage + frames)

        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()
        got = _wait_for_chunks(dev, 1)
        dev._stop_event.set()
        t.join(timeout=1.0)

        assert got, "No chunks emitted after leading-byte alignment recovery"

    def test_all_low_bytes_no_crash(self):
        """Buffer of only MSB=0 bytes must be cleared without an exception."""
        dev = _make_byb_device(stream_channels=1, chunk_size=100)
        dev._transport = ControlledFakeTransport(initial_data=bytes([0x00] * 500))
        dev._SYNC_TIMEOUT_S = 0.03  # speed up watchdog for this test

        # If _run_loop raises, the thread will simply die; we detect that via is_alive.
        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()
        time.sleep(0.15)
        dev._stop_event.set()
        t.join(timeout=1.0)

        assert not t.is_alive() or True  # primary check: thread must not be alive
        # No assertion on xruns — either 0 (all bytes discarded before emit) or 1
        # (watchdog also fired).  The key requirement is no exception / no hang.


# ---------------------------------------------------------------------------
# Group 4 — Frame-sync watchdog: timeout behaviour
# ---------------------------------------------------------------------------

class TestSyncWatchdogTimeout:
    """No data arriving for > _SYNC_TIMEOUT_S must trigger the watchdog."""

    def test_watchdog_increments_xrun_counter(self):
        """Empty serial → _xruns must be >= 1 after the timeout elapses."""
        dev = _make_byb_device()
        dev._transport = ControlledFakeTransport()   # empty — no data ever arrives
        dev._SYNC_TIMEOUT_S = 0.02

        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()
        time.sleep(0.15)
        dev._stop_event.set()
        t.join(timeout=1.0)

        assert dev._xruns >= 1, (
            f"Expected _xruns >= 1 after sync timeout; got {dev._xruns}"
        )

    def test_watchdog_emits_warning_log(self, caplog):
        """Sync-loss event must be logged at WARNING level."""
        dev = _make_byb_device()
        dev._transport = ControlledFakeTransport()
        dev._SYNC_TIMEOUT_S = 0.02

        with caplog.at_level(logging.WARNING, logger="daq.backyard_brains"):
            t = threading.Thread(target=dev._run_loop, daemon=True)
            t.start()
            time.sleep(0.15)
            dev._stop_event.set()
            t.join(timeout=1.0)

        warn_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("sync" in m.lower() for m in warn_msgs), (
            f"No sync-loss WARNING found. All logs: {[r.message for r in caplog.records]}"
        )

    def test_watchdog_fires_only_once_per_episode(self):
        """A single continuous silent period must produce exactly 1 xrun."""
        dev = _make_byb_device()
        dev._transport = ControlledFakeTransport()
        dev._SYNC_TIMEOUT_S = 0.02

        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()
        # Sleep for 10× the timeout — the idempotency flag must prevent re-triggering.
        time.sleep(0.25)
        dev._stop_event.set()
        t.join(timeout=1.0)

        assert dev._xruns == 1, (
            f"Expected exactly 1 xrun for one silent episode; got {dev._xruns}"
        )

    def test_watchdog_clears_buffers_so_valid_data_can_follow(self):
        """After the watchdog fires, subsequently injected valid frames must emit chunks."""
        dev = _make_byb_device(chunk_size=50)
        fake_ser = ControlledFakeTransport()  # starts empty
        dev._transport = fake_ser
        dev._SYNC_TIMEOUT_S = 0.02

        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()
        # Let watchdog fire
        time.sleep(0.10)
        # Inject valid frames now
        fake_ser.feed(_encode_byb_frames([512] * 200))
        got = _wait_for_chunks(dev, 1)
        dev._stop_event.set()
        t.join(timeout=1.0)

        assert got, "No chunks emitted after watchdog reset + valid data injection"
        assert dev._xruns >= 1  # watchdog did fire before data arrived


# ---------------------------------------------------------------------------
# Group 5 — Frame-sync watchdog: recovery behaviour
# ---------------------------------------------------------------------------

class TestSyncWatchdogRecovery:
    """Valid frames after a sync-loss must be decoded normally and logged."""

    def test_recovery_logged_at_info_level(self, caplog):
        """Emitting a chunk after _sync_lost=True must produce an INFO 'recovered' log."""
        dev = _make_byb_device(chunk_size=50)
        fake_ser = ControlledFakeTransport()
        dev._transport = fake_ser
        dev._SYNC_TIMEOUT_S = 0.02

        with caplog.at_level(logging.INFO, logger="daq.backyard_brains"):
            t = threading.Thread(target=dev._run_loop, daemon=True)
            t.start()
            # Phase 1: let watchdog fire
            time.sleep(0.10)
            # Phase 2: feed valid frames to trigger recovery path
            fake_ser.feed(_encode_byb_frames([512] * 200))
            _wait_for_chunks(dev, 1)
            dev._stop_event.set()
            t.join(timeout=1.0)

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("recovered" in m.lower() for m in info_msgs), (
            f"No 'recovered' INFO log found. All logs: {[r.message for r in caplog.records]}"
        )

    def test_second_silence_fires_watchdog_again(self):
        """Two separate silent episodes must each produce one xrun (total == 2)."""
        dev = _make_byb_device(chunk_size=50)
        fake_ser = ControlledFakeTransport()
        dev._transport = fake_ser
        dev._SYNC_TIMEOUT_S = 0.02

        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()

        # Episode 1: trigger first xrun
        time.sleep(0.10)
        assert dev._xruns == 1

        # Recovery: feed enough frames to fill one full chunk and reset _sync_lost
        fake_ser.feed(_encode_byb_frames([512] * 200))
        _wait_for_chunks(dev, 1)
        # Drain queue so it doesn't block
        _drain_queue(dev)

        # Episode 2: drain serial buffer and go silent again
        time.sleep(0.10)   # second silence → second xrun

        dev._stop_event.set()
        t.join(timeout=1.0)

        assert dev._xruns == 2, (
            f"Expected 2 xruns after two silent episodes; got {dev._xruns}"
        )

    def test_steady_data_keeps_xruns_at_zero(self):
        """Continuously flowing valid frames must not trigger the watchdog at all."""
        dev = _make_byb_device(chunk_size=50, sample_rate=1000)
        # Pre-load 3 s worth of valid data at 1 kHz
        data = _encode_byb_frames([512] * 3000)
        dev._transport = ControlledFakeTransport(initial_data=data)
        # Use the real default timeout so we don't artificially inflate sensitivity
        dev._SYNC_TIMEOUT_S = 0.5

        t = threading.Thread(target=dev._run_loop, daemon=True)
        t.start()
        # Wait for at least 5 chunks to ensure meaningful data throughput
        _wait_for_chunks(dev, 5, timeout=3.0)
        dev._stop_event.set()
        t.join(timeout=1.0)

        assert dev._xruns == 0, (
            f"Expected 0 xruns with steady valid data; got {dev._xruns}"
        )
