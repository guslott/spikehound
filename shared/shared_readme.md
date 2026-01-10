# Shared Module

Common data types and utilities used throughout SpikeHound. This module defines the immutable dataclasses that flow between components.

---

## Data Type Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     Device Layer (daq/)                         │
│                                                                 │
│   DeviceInfo     ChannelInfo     Capabilities     ActualConfig  │
│   (discovery)    (channels)     (device caps)    (config result)│
└─────────────────────────────┬───────────────────────────────────┘
                              │ produces
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streaming Layer (core/)                      │
│                                                                 │
│        Chunk                ChunkPointer              │
│   (raw samples + metadata)    (pointer to ring buffer)          │
└─────────────────────────────┬───────────────────────────────────┘
                              │ triggers
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Detection Layer (core/detection/)             │
│                                                                 │
│                      DetectionEvent                             │
│              (simple: t, chan, window, properties)              │
└─────────────────────────────┬───────────────────────────────────┘
                              │ enriched to
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis Layer (analysis/, gui/)             │
│                                                                 │
│                      AnalysisEvent                              │
│        (detailed: timing metadata, computed metrics)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## When to Use Each Type

### Device Types (`shared/models.py`)

| Type | Use When | Example |
|------|----------|---------|
| `DeviceInfo` | Discovering available devices | `list_available_devices()` |
| `ChannelInfo` | Listing device channels | `list_available_channels()` |
| `Capabilities` | Querying device limits | `get_capabilities()` |
| `ActualConfig` | Storing configured state | `configure()` return value |

### Streaming Types (`shared/models.py`)

| Type | Use When | Example |
|------|----------|---------|
| `Chunk` | Passing raw samples between threads | DAQ → Dispatcher |
| `ChunkPointer` | Referencing data in ring buffer | Dispatcher → Visualization |

### Event Types

| Type | Location | Use When |
|------|----------|----------|
| `DetectionEvent` | `shared/models.py` | Low-level detection output |
| `AnalysisEvent` | `shared/types.py` | GUI display with full timing |

**Key Difference:**
- `DetectionEvent`: Simple, lightweight (`t`, `chan`, `window`). Canonical output of the detection layer.
- `AnalysisEvent`: Rich metadata (timing, sample rate, pre/post windows). Enriched version for analysis and UI.

#### Event Conversion Contract

To keep the codebase clean and avoid confusion, we follow a strict conversion contract:

1. **Detectors** (in `core/detection/`) ONLY emit `DetectionEvent`.
2. **Analysis Workers** (in `analysis/`) receive `DetectionEvent` and convert it to `AnalysisEvent`.
3. **The GUI** and **Analysis Tabs** consume `AnalysisEvent` for display and metric reporting.

The canonical conversion function is `analysis.analysis_worker.detection_to_analysis_event`.

---

## Type Definitions

### DetectionEvent (`shared/models.py`)

Emitted by threshold detectors in the core layer:

```python
from shared.models import DetectionEvent

event = DetectionEvent(
    t=1.234,                    # Seconds since stream start
    chan=0,                     # Channel index
    window=waveform_array,      # Samples around detection
    properties={"amplitude": 0.5},
    params={"threshold": 0.3},
)
```

### AnalysisEvent (`shared/types.py`)

Detailed event for GUI display:

```python
from shared.types import AnalysisEvent

event = AnalysisEvent(
    id=42,                      # Unique event ID
    channelId=0,                # Channel index
    thresholdValue=0.3,         # Threshold crossed
    crossingIndex=12345,        # Absolute sample index
    crossingTimeSec=1.234,      # Timestamp of crossing
    firstSampleTimeSec=1.232,   # Start of window
    sampleRateHz=10000.0,       # For time calculations
    windowMs=5.0,               # Total window duration
    preMs=2.0,                  # Before crossing
    postMs=3.0,                 # After crossing
    samples=waveform_array,     # The waveform
    properties={"energy": 1.5}, # Computed metrics
    intervalSinceLastSec=0.15,  # ISI
)
```

### TriggerConfig (`shared/models.py`)

Trigger settings shared between GUI and core:

```python
from shared.models import TriggerConfig

config = TriggerConfig(
    channel_index=0,
    threshold=0.5,
    hysteresis=0.0,
    pretrigger_frac=0.2,
    window_sec=1.0,
    mode="continuous",  # "stream", "single", "continuous"
)
```

---

## Other Utilities

### SharedRingBuffer (`ring_buffer.py`)

Thread-safe ring buffer with fast path:

```python
from shared.ring_buffer import SharedRingBuffer

buffer = SharedRingBuffer(capacity=1_000_000, n_channels=4)
buffer.write(samples)  # Write from producer
data = buffer.read(start, length)  # Read from consumers
```

### EventRingBuffer (`event_buffer.py`)

Thread-safe event storage with ID-based retrieval:

```python
from shared.event_buffer import EventRingBuffer

buffer = EventRingBuffer(capacity=1000)
buffer.push(event)
events = buffer.pull_since(last_id)
```

### AppSettingsStore (`app_settings.py`)

Persistent application settings with observer pattern:

```python
from shared.app_settings import AppSettingsStore

store = AppSettingsStore()
store.update(audio_device="device_1")
settings = store.get()
```

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `models.py` | 245 | Core dataclasses (Chunk, DetectionEvent, etc.) |
| `types.py` | 88 | AnalysisEvent type |
| `ring_buffer.py` | 100 | SharedRingBuffer |
| `event_buffer.py` | 84 | EventRingBuffer |
| `app_settings.py` | 140 | AppSettingsStore |

---

## Immutability

All data types are **frozen dataclasses** with validation:

```python
@dataclass(frozen=True)
class TriggerConfig:
    channel_index: int
    threshold: float
    # ...
```

Benefits:
- **Thread-safe**: Can pass between threads without copying
- **Hashable**: Can use as dict keys or in sets
- **Validated**: `__post_init__` checks constraints
