# DAQ Module

A small, composable framework for **multi-channel, real-time data acquisition** (DAQ).
Designed “scope-first”: a GUI can treat every hardware backend the same—consume `Chunk`s from a queue and draw.

* Minimal, boring contract
* Clean lifecycle: `open → configure → start/stop → close`
* Bounded queue with **drop-oldest** backpressure
* Uniform **float32 (frames, channels)** data
* Dual time bases: host `mono_time` and optional hardware `device_time`
* A base class that centralizes the tricky bits so drivers stay tiny

> This README covers the DAQ module only (base + drivers), not the visualization demo.

---

## Quick start

### Simulated input

```python
from daq.simulated_source import SimulatedPhysiologySource

src = SimulatedPhysiologySource()
devs = src.list_available_devices()
src.open(devs[0].id)
src.configure(sample_rate=20_000, channels=None, chunk_size=1024)
src.start()

# Consume data (e.g., in your processing thread)
while some_condition:
    try:
        # 1. Get pointer from queue (ChunkPointer | EndOfStream)
        ptr = src.data_queue.get(timeout=1.0)
        
        # 2. Access data from shared ring buffer
        # Shape: (channels, frames), dtype: float32
        samples = src.get_buffer().read(ptr.start_index, ptr.length)
        
        print(f"Got {ptr.length} frames at t={ptr.render_time}")
    except:
        break

src.stop()
src.close()
```

### Sound card input (cross-platform)

```python
from daq.soundcard_source import SoundCardSource

src = SoundCardSource()
print("Devices:")
for d in src.list_available_devices():
    print(f" - {d.id}: {d.name}")

src.open(device_id="default")
caps = src.get_capabilities("default")
chans = src.list_available_channels("default")

# pick first two channels
if len(chans) >= 2:
    src.configure(sample_rate=48_000, channels=[chans[0].id, chans[1].id], chunk_size=1024)
    src.start()
    
    # consume data as shown above...
    src.stop()
src.close()
```

---

## Architecture & lifecycle

Every input source is a subclass of `BaseDevice` and obeys the same lifecycle:

```
list_available_devices()
open(device_id)
  ├─ get_capabilities(device_id)
  └─ list_available_channels(device_id)
configure(sample_rate, channels, chunk_size, **options) -> ActualConfig
start()
  [ driver thread emits array -> BaseDevice writes Buffer + emits ChunkPointer ]
stop()
close()
```

State machine: `"closed" → "open" → "running" → "open" → "closed"`
Illegal transitions raise `RuntimeError` early (e.g., `start()` without `configure()`).

---

## Data model

The DAQ system uses standard data types defined in `shared.models`.

### Device Metadata

```python
@dataclass(frozen=True)
class DeviceInfo:
    id: str           # Driver-specific identifier
    name: str         # Human-friendly name
    vendor: str|None = None
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ChannelInfo:
    id: int
    name: str
    units: str = "V"
    range: tuple[float, float] | None = None
```

### Config & Capabilities

```python
@dataclass(frozen=True)
class Capabilities:
    max_channels_in: int
    sample_rates: list[int] | None  # None => continuous range
    dtype: str = "float32"

@dataclass(frozen=True)
class ActualConfig:
    sample_rate: int
    channels: list[ChannelInfo]
    chunk_size: int
    dtype: str = "float32"
```

### Streaming Data

Drivers produce `Chunk` objects internally (or write directly to the buffer), but consumers receive `ChunkPointer`s to zero-copy data in the ring buffer.

```python
@dataclass(frozen=True)
class Chunk:
    """Atomic unit of streaming data passed explicitly (e.g. from tests)."""
    samples: np.ndarray          # Shape: (channels, frames)
    start_time: float            # Host monotonic time
    dt: float                    # Sample period (1/rate)
    seq: int
    channel_names: tuple[str, ...]
    units: str

@dataclass(frozen=True)
class ChunkPointer:
    """Pointer to data in the SharedRingBuffer."""
    start_index: int             # Index in ring buffer
    length: int                  # Number of frames
    render_time: float           # Approximate time for visualization
```

**Shape & dtype**

* **Ring Buffer**: Stores `float32` data as `(channels, frames)`.
* **Drivers**: Emit `float32` arrays to the buffer.


---

## BaseDevice API (for consumers)

```python
class BaseDevice(ABC):
    # Device discovery
    def list_available_devices(self) -> list[DeviceInfo]: ...
    def get_capabilities(self, device_id: str) -> Capabilities: ...
    def list_available_channels(self, device_id: str) -> list[ChannelInfo]: ...

    # Lifecycle
    def open(self, device_id: str) -> None: ...
    def configure(self, sample_rate: int, channels: Sequence[int] | None, chunk_size: int, **options) -> ActualConfig: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...

    # Channel selection helpers
    def set_active_channels(self, channel_ids: Sequence[int]) -> None: ...
    def get_active_channels(self) -> list[ChannelInfo]: ...

    # Queue of pointers to consume (ChunkPointer | EndOfStream)
    data_queue: "queue.Queue[ChunkPointer]"

    # Introspection
    @property
    def state(self) -> Literal["closed", "open", "running"]: ...
    @property
    def running(self) -> bool: ...
    def stats(self) -> dict[str, Any]: ...
    def get_buffer(self) -> SharedRingBuffer: ...
```

**Backpressure policy**

* `data_queue` uses **blocking backpressure** (not drop-oldest).
* The driver writes to a `SharedRingBuffer` and enqueues a pointer.
* If the queue fills up, the driver blocks (up to 10s) to enforce lossless transmission.
* Drop-oldest behavior is implemented downstream (in the Dispatcher) if the visualization/analysis threads cannot keep up, but the core acquisition pipeline remains lossless.

**Threading model**

* The driver owns the producer (thread or audio callback).
* Your GUI should **never block** the producer; consume on the UI side using non-blocking or short timeouts and do any heavier processing off the UI thread.

---

## Consumer patterns

### Smooth UI updates (fixed frame rate)

Draw at \~30–60 Hz regardless of DAQ chunk cadence:

```python
import time
buf = []

def drain(q):
    drained = 0
    while True:
        try:
            buf.append(q.get_nowait())
            drained += 1
        except Exception:
            break
    return drained

last_draw = time.monotonic()
while src.running:
    drained = drain(src.data_queue)
    now = time.monotonic()
    if now - last_draw >= 1/60 or drained == 0:
        # render everything currently in buf (or keep a rolling window)
        buf.clear()
        last_draw = now
    time.sleep(0.001)  # be polite
```

### Switching channels at runtime

```python
src.stop()
src.configure(sample_rate=src.config.sample_rate,
              channels=[0, 3, 4],             # your chosen ids
              chunk_size=src.config.chunk_size)
src.start()
```

---

## Writing a new driver

Implement a subclass with five driver hooks and use the base utilities to emit data. Most drivers are 100–200 lines.

### Minimal skeleton

```python
# mydevice_source.py
from base_device import BaseDevice, DeviceInfo, ChannelInfo, Capabilities, ActualConfig
import numpy as np
import threading

class MyDeviceSource(BaseDevice):
    def __init__(self, queue_maxsize: int = 64):
        super().__init__(queue_maxsize)
        self._thread = None
        self._hw = None  # your hardware/session handle

    # 1) Discovery ------------------------------------------------------------
    def list_available_devices(self):
        # Probe your SDK; return a few fields that help users pick a device
        return [DeviceInfo(id="Dev0", name="MyDAQ USB", vendor="ACME")]

    def get_capabilities(self, device_id: str):
        return Capabilities(
            max_channels_in=8,
            sample_rates=None,   # None = continuous range
            dtype="float32",
            notes="±10V, 16-bit ADC, internally normalized to float32"
        )

    def list_available_channels(self, device_id: str):
        return [ChannelInfo(id=i, name=f"AI{i}", units="V", range=(-10, 10)) for i in range(8)]

    # 2) Lifecycle ------------------------------------------------------------
    def _open_impl(self, device_id: str):
        # Open hardware/session without starting acquisition
        self._hw = ...  # open your device handle

    def _close_impl(self) -> None:
        # Release hardware/session
        self._hw = None

    def _configure_impl(self, sample_rate: int, channels: list[int], chunk_size: int, **opts) -> ActualConfig:
        # Apply configuration to hardware (don’t start yet)
        # Optionally clamp chunk_size to hardware block sizes.
        self._sample_rate = sample_rate
        self._channels = list(channels)
        self._chunk_size = chunk_size
        # Return the actual configuration (echo what the hardware will do)
        chosen = [c for c in self._available_channels if c.id in self._channels]
        return ActualConfig(sample_rate=sample_rate, channels=chosen, chunk_size=chunk_size, latency_s=None, dtype="float32")

    def _start_impl(self) -> None:
        # Spawn a producer thread or start a hardware callback
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _stop_impl(self) -> None:
        # Cooperatively stop; BaseDevice.stop_event has already been set
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # 3) Producer -------------------------------------------------------------
    def _run_loop(self):
        sr = self.config.sample_rate
        frame_block = self.config.chunk_size
        chans = len(self.config.channels)

        # Example: pull/convert from SDK into float32 (frames, channels)
        while not self.stop_event.is_set():
            # Replace this with your SDK read; keep it non-blocking or short timeout.
            block = np.zeros((frame_block, chans), dtype=np.float32)
            # device_time = ... (seconds) if your SDK provides a timestamp
            self.emit_array(block, device_time=None)  # stamps seq/start_sample/mono_time
```

### Required behaviors

* Emit **float32** arrays shaped `(frames, channels)` in the **active channel order**.
* Call `self.emit_array(...)` for each produced chunk. It:

  * validates shape and dtype,
  * slices a superset down to the active channels (when possible),
  * stamps `seq`, `start_sample`, `mono_time` (+ optional `device_time`),
  * writes to the `SharedRingBuffer` and enqueues a pointer (blocking if full).

### Timestamps

* If your backend exposes a reliable timestamp for the **first sample** of each block, pass it as `device_time=` to `emit_array`. Keep it in **seconds** (float).
* If not, omit it; downstream code must treat it as `None`.

### Channel mapping

* Ideal: configure the backend to **only deliver** the active channels, in order.
* Acceptable: deliver a fixed superset in hardware order; `emit_array` will slice it to the active set if `chans == len(available_channels)`.

### Chunk size and latency

* Prefer powers of two (`256–4096`) for real-time smoothness.
* Return the actual `chunk_size` you achieve and, if available, a nominal `latency_s` in `ActualConfig` so UIs can communicate expectations.

### Error handling & robustness

* Raise early (e.g., invalid device/channel IDs).
* If the backend signals over/underruns (XRUNs), call `self.note_xrun()`; UIs can surface `stats()["xruns"]`.
* Use the provided `stop_event` for cooperative shutdown; avoid hard blocking.

---

## Driver checklist

* [ ] `list_available_devices()`: returns at least one `DeviceInfo`
* [ ] `get_capabilities()`: sane `max_channels_in`, `sample_rates`, `dtype`
* [ ] `list_available_channels()`: complete, stable channel list
* [ ] `open(device_id)`: resources allocated; state is `"open"`
* [ ] `configure(...) -> ActualConfig`: echoes what you’ll actually run
* [ ] `start()/_start_impl()`: producer runs; emits via `emit_array(...)`
* [ ] `stop()/_stop_impl()`: producer stops within \~2s
* [ ] `close()/_close_impl()`: resources released; state is `"closed"`
* [ ] Emits **float32** (frames, channels) via `emit_array`
* [ ] Optional `device_time` passed when available
* [ ] Helper `emit_array` takes care of ring buffer writes and queueing

---

## Stats & introspection

```python
src.stats()
# => {
#   'state': 'running',
#   'queue_size': 3,
#   'queue_maxsize': 64,
#   'xruns': 0,
#   'drops': 2,
#   'next_seq': 128,
#   'next_start_sample': 131072,
#   'sample_rate': 48000,
#   'active_channels': [0, 1]
# }
```

Use this sparingly (e.g., once per second) to drive a small “health” indicator in your UI.

---

## Best practices

* **Never block** the producer. If your SDK requires blocking reads, use short timeouts and check `stop_event`.
* Keep conversions cheap: write directly into **float32** buffers when possible; avoid `vstack`/growing arrays in hot paths.
* Treat `device_time` as a luxury: use it when present, but keep logic tolerant to `None`.
* Prefer **single-writer, single-reader** patterns (one producer thread, UI drains queue).
* Normalize units: if your backend is in counts, scale to engineering units (Volts) before emitting and document in `ChannelInfo.units`.

---

## FAQ

**Q: What happens if the consumer is too slow?**
The source will block (up to 10s) and then raise an error. We enforce **lossless** transmission for the core pipeline. If you see high latency or errors, ensure your consumer drains the queue faster than real-time. Drop-oldest policies are applied downstream (e.g. in the `Dispatcher` visualization queues) to protect the UI, but the raw data acquisition is never compromised.

**Q: What chunk size should I start with?**
Start at `1024` frames. Go smaller (`256–512`) if you want snappier interactivity (at the cost of more callbacks), or larger (`2048–4096`) if you want fewer calls and can tolerate extra latency.

**Q: How do I align multiple devices?**
Use `device_time` if both devices supply accurate hardware timestamps. Otherwise, treat `mono_time` as a rough guide and apply post-hoc alignment in analysis.

---

## Module layout (suggested)

```
daq/
  base_device.py          # (this contract)
  simulated_source.py     # reference implementation
  soundcard_source.py     # miniaudio-based driver
  __init__.py             # re-exports public classes for convenience
```

`__init__.py` example:

```python
from .base_device import BaseDevice, DeviceInfo, ChannelInfo, Capabilities, ActualConfig, Chunk
from .simulated_source import SimulatedPhysiologySource
from .soundcard_source import SoundCardSource

__all__ = [
    "BaseDevice", "DeviceInfo", "ChannelInfo", "Capabilities", "ActualConfig", "Chunk",
    "SimulatedPhysiologySource", "SoundCardSource",
]
```

---

## Compatibility notes

* **SoundCardSource** uses `miniaudio`; it supports macOS, Windows, and Linux.

  * Prefer `sample_rate` of `44_100` or `48_000` unless you’ve verified others on the target OS/driver.
  * When available, the driver stamps `device_time` with PortAudio’s `input_buffer_adc_time`.