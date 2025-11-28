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
from simulated_source import SimulatedSource

src = SimulatedSource()
devs = src.list_available_devices()
src.open(devs[0].id)
src.configure(sample_rate=20_000, channels=None, chunk_size=1024)  # channels=None => all
src.start()

# Consume chunks (e.g., in your GUI thread)
while some_condition:
    try:
        ch = src.data_queue.get(timeout=0.050)  # ch is a Chunk
        # ch.data shape: (frames, channels), dtype=float32
        # use ch.mono_time, ch.device_time (optional), ch.seq, ch.start_sample
    except Exception:
        pass

src.stop()
src.close()
```

### Sound card input (cross-platform)

```python
from soundcard_source import SoundCardSource

src = SoundCardSource()
for d in src.list_available_devices():
    print(d.id, d.name)

src.open(device_id="default")  # or choose an ID from the printed list
caps = src.get_capabilities("default")
chans = src.list_available_channels("default")

# pick first two channels
src.configure(sample_rate=48_000, channels=[chans[0].id, chans[1].id], chunk_size=1024)
src.start()

# drain chunks as above...
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
  [ driver thread/callback emits Chunk -> BaseDevice.data_queue ]
stop()
close()
```

State machine: `"closed" → "open" → "running" → "open" → "closed"`
Illegal transitions raise `RuntimeError` early (e.g., `start()` without `configure()`).

---

## Data model

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

@dataclass(frozen=True)
class Capabilities:
    max_channels_in: int
    sample_rates: list[int] | None  # None => continuous range
    dtype: str = "float32"
    notes: str | None = None

@dataclass(frozen=True)
class ActualConfig:
    sample_rate: int
    channels: list[ChannelInfo]
    chunk_size: int
    latency_s: float | None = None
    dtype: str = "float32"

@dataclass(frozen=True)
class Chunk:
    start_sample: int            # 0-based from start()
    mono_time: float             # host monotonic time (seconds)
    seq: int                     # chunk sequence (0,1,2,…)
    data: np.ndarray             # shape: (frames, channels), dtype=float32 by default
    device_time: float | None = None  # hardware ADC clock time (if available)
```

**Timebases**

* `mono_time`: host‐side `_time.monotonic()` at the first sample of the chunk.
* `device_time`: (optional) ADC/driver timestamp for the first sample (e.g., PortAudio’s `input_buffer_adc_time`). Use this to align cross-device streams or estimate drift/jitter. If absent, treat as `None`.

**Shape & dtype**

* All drivers emit **float32** by default.
* `Chunk.data` is always `(frames, channels)` in the **user-selected channel order**.

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

    # Queue of data to consume (non-blocking producer)
    data_queue: "queue.Queue[Chunk]"

    # Introspection
    @property
    def state(self) -> Literal["closed", "open", "running"]: ...
    @property
    def running(self) -> bool: ...
    def stats(self) -> dict[str, Any]: ...
```

**Backpressure policy**

* `data_queue` is bounded (default size 64).
* On overflow, the **oldest** chunk is evicted and a counter increments (`"drops"` in `stats()`).
* This keeps the UI responsive and memory bounded.

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
  * enqueues with the module’s **drop-oldest** backpressure policy.

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
* [ ] Emits **float32**, `(frames, channels)`, correct channel order
* [ ] Optional `device_time` passed when available
* [ ] Handles backpressure implicitly via base class

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

**Q: Why do I see “drops” in `stats()`?**
Your consumer isn’t draining fast enough for bursts. That’s OK for live scope use—the policy drops the **oldest** chunk to keep latency low. If you need lossless capture, attach a file writer thread behind the queue with a larger buffer and disk I/O.

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
  soundcard_source.py     # PortAudio/sounddevice-based driver
  __init__.py             # re-exports public classes for convenience
```

`__init__.py` example:

```python
from .base_device import BaseDevice, DeviceInfo, ChannelInfo, Capabilities, ActualConfig, Chunk
from .simulated_source import SimulatedSource
from .soundcard_source import SoundCardSource

__all__ = [
    "BaseDevice", "DeviceInfo", "ChannelInfo", "Capabilities", "ActualConfig", "Chunk",
    "SimulatedSource", "SoundCardSource",
]
```

---

## Compatibility notes

* **SoundCardSource** uses PortAudio via `sounddevice`; it supports macOS, Windows, and Linux.

  * Prefer `sample_rate` of `44_100` or `48_000` unless you’ve verified others on the target OS/driver.
  * When available, the driver stamps `device_time` with PortAudio’s `input_buffer_adc_time`.