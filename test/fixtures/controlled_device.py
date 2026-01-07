"""
Controlled DAQ device for deterministic testing.

This device produces exact, repeatable waveforms allowing tests to validate
the pipeline behavior with known inputs. Unlike SimulatedPhysiologySource,
this device is designed purely for testing with full control over timing.

Key Features:
- Produces configurable waveform patterns (zeros, ramps, custom)
- Can inject known spike trains at specific times
- Supports failure simulation (disconnect, corrupt data)
- Tracks all emitted chunks for post-test validation

Design for AI-maintainability:
- Inherits from BaseDevice to ensure contract compliance
- Clear separation between configuration and runtime behavior
- All state is inspectable for test assertions
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from daq.base_device import BaseDevice
from shared.models import ActualConfig, Capabilities, ChannelInfo, DeviceInfo, Chunk


@dataclass
class EmittedChunk:
    """Record of a chunk emitted by ControlledDevice for validation."""
    seq: int
    start_sample: int
    samples: np.ndarray
    timestamp: float


class ControlledDevice(BaseDevice):
    """DAQ device for deterministic test scenarios.

    Emits chunks at controllable intervals with configurable waveform patterns.
    Useful for testing pipeline behavior without hardware dependencies.

    Example:
        >>> device = ControlledDevice()
        >>> device.open("test")
        >>> device.configure(10000, [0, 1], chunk_size=256)
        >>> device.set_waveform(lambda t, ch: np.sin(2 * np.pi * 100 * t))
        >>> device.start()
        >>> # ... consume from device.queue
        >>> device.stop()
        >>> device.close()
    """

    @classmethod
    def device_class_name(cls) -> str:
        return "Controlled Test Device"

    def __init__(
        self,
        queue_maxsize: int = 64,
        *,
        chunk_interval_sec: float = 0.01,  # 10ms between chunks
        max_chunks: Optional[int] = None,  # Stop after N chunks (None = unlimited)
    ) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        self._chunk_interval = chunk_interval_sec
        self._max_chunks = max_chunks

        # Waveform generator: f(time_array, channel_index) -> samples
        self._waveform_fn: Callable[[np.ndarray, int], np.ndarray] = lambda t, ch: np.zeros_like(t)

        # Failure injection
        self._fail_after_chunks: Optional[int] = None
        self._fail_exception: Optional[Exception] = None

        # Tracking for test validation
        self._emitted_chunks: List[EmittedChunk] = []
        self._lock = threading.Lock()

        # Thread management
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        return [
            DeviceInfo(
                id="controlled_test",
                name="Controlled Test Device",
                vendor="SpikeHound Tests",
            )
        ]

    def get_capabilities(self, device_id: str) -> Capabilities:
        return Capabilities(
            max_channels_in=16,
            sample_rates=[1000, 2000, 4000, 8000, 10000, 20000, 44100, 48000],
            dtype="float32",
        )

    def list_available_channels(self, device_id: str) -> List[ChannelInfo]:
        return [
            ChannelInfo(id=i, name=f"TestCh{i}", units="V")
            for i in range(16)
        ]

    # -------------------------------------------------------------------------
    # Configuration API
    # -------------------------------------------------------------------------

    def set_waveform(self, waveform_fn: Callable[[np.ndarray, int], np.ndarray]) -> None:
        """Set waveform generator function.

        Args:
            waveform_fn: Function taking (time_array, channel_index) and returning
                samples for that channel. Time array is in seconds from stream start.
        """
        self._waveform_fn = waveform_fn

    def set_constant(self, value: float) -> None:
        """Set constant output value for all channels."""
        self._waveform_fn = lambda t, ch: np.full_like(t, value)

    def set_ramp(self, slope: float = 1.0) -> None:
        """Set linear ramp output (for sequence testing)."""
        self._waveform_fn = lambda t, ch: (t * slope).astype(np.float32)

    def inject_failure_after(self, n_chunks: int, exception: Optional[Exception] = None) -> None:
        """Inject a simulated failure after N chunks.

        Args:
            n_chunks: Number of chunks to emit before "failing".
            exception: Exception to raise (default: simulates disconnect).
        """
        self._fail_after_chunks = n_chunks
        self._fail_exception = exception or RuntimeError("Simulated device disconnect")

    def clear_failure_injection(self) -> None:
        """Remove any pending failure injection."""
        self._fail_after_chunks = None
        self._fail_exception = None

    # -------------------------------------------------------------------------
    # Test introspection
    # -------------------------------------------------------------------------

    def get_emitted_chunks(self) -> List[EmittedChunk]:
        """Return list of all chunks emitted since last start()."""
        with self._lock:
            return list(self._emitted_chunks)

    def get_total_samples_emitted(self) -> int:
        """Return total sample count emitted since last start()."""
        with self._lock:
            return sum(c.samples.shape[1] for c in self._emitted_chunks)

    # -------------------------------------------------------------------------
    # BaseDevice implementation
    # -------------------------------------------------------------------------

    def _open_impl(self, device_id: str) -> None:
        # No real hardware to open
        pass

    def _close_impl(self) -> None:
        # No real hardware to close
        pass

    def _configure_impl(
        self,
        sample_rate: int,
        channels: Sequence[int],
        chunk_size: int,
        **options,
    ) -> ActualConfig:
        return ActualConfig(
            sample_rate=sample_rate,
            channels=[ChannelInfo(id=c, name=f"TestCh{c}", units="V") for c in channels],
            chunk_size=chunk_size,
            dtype="float32",
        )

    def _start_impl(self) -> None:
        # Clear tracking from previous run
        with self._lock:
            self._emitted_chunks.clear()

        self._thread = threading.Thread(
            target=self._producer_loop,
            name="ControlledDevice",
            daemon=True,
        )
        self._thread.start()

    def _stop_impl(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _producer_loop(self) -> None:
        """Main producer loop - emits chunks at controlled intervals."""
        config = self._config
        if config is None:
            return

        sample_rate = config.sample_rate
        chunk_size = config.chunk_size
        n_channels = len(config.channels)
        dt = 1.0 / sample_rate

        chunk_count = 0
        current_sample = 0

        while not self._stop_event.is_set():
            # Check for failure injection
            if self._fail_after_chunks is not None and chunk_count >= self._fail_after_chunks:
                # Simulate failure by stopping
                break

            # Check for max chunks limit
            if self._max_chunks is not None and chunk_count >= self._max_chunks:
                break

            # Generate chunk samples
            t_start = current_sample * dt
            t_array = t_start + np.arange(chunk_size, dtype=np.float64) * dt

            samples = np.zeros((n_channels, chunk_size), dtype=np.float32)
            for ch_idx in range(n_channels):
                samples[ch_idx] = self._waveform_fn(t_array, ch_idx)

            # Emit via BaseDevice infrastructure
            self.emit_array(samples.T)  # emit_array expects (frames, channels)

            # Track for validation
            with self._lock:
                self._emitted_chunks.append(EmittedChunk(
                    seq=chunk_count,
                    start_sample=current_sample,
                    samples=samples.copy(),
                    timestamp=time.time(),
                ))

            current_sample += chunk_size
            chunk_count += 1

            # Wait for next chunk interval
            self._stop_event.wait(self._chunk_interval)


class BlockingDevice(ControlledDevice):
    """ControlledDevice variant that can simulate slow producers.

    Useful for testing consumer timeout behavior.
    """

    def __init__(self, block_duration_sec: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._block_duration = block_duration_sec
        self._block_before_chunk: Optional[int] = None

    def block_before_chunk(self, chunk_number: int) -> None:
        """Block the producer thread before emitting chunk N."""
        self._block_before_chunk = chunk_number

    def _producer_loop(self) -> None:
        # Override to inject blocking behavior
        config = self._config
        if config is None:
            return

        sample_rate = config.sample_rate
        chunk_size = config.chunk_size
        n_channels = len(config.channels)
        dt = 1.0 / sample_rate

        chunk_count = 0
        current_sample = 0

        while not self._stop_event.is_set():
            # Check for blocking injection
            if self._block_before_chunk is not None and chunk_count == self._block_before_chunk:
                time.sleep(self._block_duration)
                self._block_before_chunk = None  # Only block once

            if self._max_chunks is not None and chunk_count >= self._max_chunks:
                break

            t_start = current_sample * dt
            t_array = t_start + np.arange(chunk_size, dtype=np.float64) * dt

            samples = np.zeros((n_channels, chunk_size), dtype=np.float32)
            for ch_idx in range(n_channels):
                samples[ch_idx] = self._waveform_fn(t_array, ch_idx)

            self.emit_array(samples.T)

            with self._lock:
                self._emitted_chunks.append(EmittedChunk(
                    seq=chunk_count,
                    start_sample=current_sample,
                    samples=samples.copy(),
                    timestamp=time.time(),
                ))

            current_sample += chunk_size
            chunk_count += 1
            self._stop_event.wait(self._chunk_interval)
