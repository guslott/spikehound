# Core Module

The pipeline orchestration layer for SpikeHound. Manages data flow from DAQ sources through conditioning, triggering, and fan-out to visualization, analysis, and audio consumers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SpikeHoundRuntime                          │
│              (Headless Orchestrator - runtime.py)               │
│                                                                 │
│  Responsibilities:                                              │
│  • Device attachment (attach_source)                            │
│  • Acquisition lifecycle (start/stop)                           │
│  • Analysis worker management                                   │
│  • Health metrics & settings persistence                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ owns
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PipelineController                           │
│              (Pipeline Lifecycle - controller.py)               │
│                                                                 │
│  Responsibilities:                                              │
│  • Source switching (swap DAQ devices)                          │
│  • Dispatcher management                                        │
│  • Filter/trigger configuration                                 │
│  • Recording control                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │ owns
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Dispatcher                               │
│              (Data Router - dispatcher.py)                      │
│                                                                 │
│  Responsibilities:                                              │
│  • Pull from DAQ source queue                                   │
│  • Apply signal conditioning (filters)                          │
│  • Fan-out to consumer queues                                   │
│  • Trigger detection & event windowing                          │
└────────────────┬────────────┬────────────┬─────────────────────┘
                 │            │            │
        ┌────────▼───┐  ┌─────▼────┐  ┌────▼─────┐
        │ Viz Queue  │  │Audio Queue│  │Log Queue │
        │(UI Plots)  │  │(Speakers) │  │(WAV File)│
        └────────────┘  └──────────┘  └──────────┘
```

---

## Key Components

### SpikeHoundRuntime (`runtime.py`)

Headless orchestrator that UIs or CLI tools interact with:

```python
from core.runtime import SpikeHoundRuntime

runtime = SpikeHoundRuntime()

# Configure the pipeline and start
controller = runtime.controller
assert controller is not None
controller.attach_source(driver, sample_rate=20000, channels=channels)
controller.set_active_channels([ch.id for ch in channels])
runtime.start_acquisition()

# Get health metrics
print(runtime.health_snapshot())

# Stop
runtime.stop_acquisition()
```

**Public API:**
| Method | Purpose |
|--------|---------|
| `start_acquisition()` / `stop_acquisition()` | Control streaming |
| `health_snapshot()` | Get dispatcher stats, queue depths |
| `open_analysis_stream(channel, sr)` | Start analysis worker |

See `runtime_readme.md` for detailed usage.

---

### PipelineController (`controller.py`)

Manages the acquisition pipeline lifecycle:

```python
from core.controller import PipelineController

controller = PipelineController()

# Switch to a new source
controller.switch_source(SimulatedPhysiologySource, configure_kwargs={...})

# Start/stop
controller.start()
controller.stop()

# Update configuration while running
controller.update_filter_settings(settings)
controller.update_trigger_config(trigger_config)
```

**Key Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `running` | `bool` | Is pipeline streaming? |
| `source` | `BaseDevice` | Current DAQ source |
| `dispatcher` | `Dispatcher` | Data router |
| `sample_rate` | `float` | Current sample rate |
| `visualization_queue` | `Queue[ChunkPointer]` | For UI plots |
| `audio_queue` | `Queue[ChunkPointer]` | For speakers |
| `logging_queue` | `Queue[Chunk]` | For WAV recording |

---

### Dispatcher (`dispatcher.py`)

The central data router. Pulls from DAQ, applies conditioning, fans out to consumers.

See `doc/dispatcher_readme.md` for details.

**Queue Semantics:**
| Queue | Policy | Behavior |
|-------|--------|----------|
| Visualization | Eviction | Evicts oldest if full |
| Audio | Eviction | Evicts oldest if full |
| Logging | Lossless | Blocks up to 10s |
| Analysis | Eviction | Evicts oldest if full |

**Note:** `EndOfStream` signals are guaranteed to be delivered. `drop-newest` queues (if any) are temporarily treated as `drop-oldest` for EOS messages.


---

### SignalConditioner (`conditioning.py`)

Per-channel digital signal processing:

```python
from core.conditioning import FilterSettings, ChannelFilterSettings

settings = FilterSettings(
    default=ChannelFilterSettings(ac_couple=True, ac_cutoff_hz=1.0),
    overrides={
        "ch0": ChannelFilterSettings(
            notch_enabled=True,
            notch_freq_hz=60.0,
            highpass_hz=10.0,
        ),
    },
)

controller.update_filter_settings(settings)
```

**Filter Types:**
| Filter | Parameter | Effect |
|--------|-----------|--------|
| Notch | `notch_freq_hz` | Remove mains hum (50/60 Hz) |
| High-pass | `highpass_hz` | Remove DC offset, drift |
| Low-pass | `lowpass_hz` | Anti-aliasing, noise reduction |

---

## Data Flow

1. **DAQ Source** produces `ChunkPointer` objects (referencing raw ring buffer data)
2. **Dispatcher** reads from source's `data_queue`
3. **SignalConditioner** applies per-channel filters
4. **Trigger Logic** detects threshold crossings, captures windows
5. **Fan-out** distributes to consumer queues:
   - `visualization_queue` → GUI plots
   - `audio_queue` → Audio player
   - `logging_queue` → WAV writer
   - `analysis_queue` → Analysis workers

---

## Adding New Functionality

### Adding a New Filter Type

1. Add parameters to `ChannelFilterSettings` in `conditioning.py`
2. Implement filter construction/application in `SignalConditioner._ensure_filters()` and `SignalConditioner.process()`
3. See `/.agent/workflows/add-filter-type.md` for complete steps

### Adding a New Consumer

To add a new downstream processor:

```python
# 1. In controller.py, register a new queue:
self._my_queue = queue.Queue(maxsize=256)

# 2. In shared/models.py, add policy to QUEUE_POLICIES dict:
QUEUE_POLICIES["my_queue"] = "drop-newest"  # or "lossless" / "drop-oldest"

# 3. In dispatcher.py _fan_out(), enqueue using the queue name:
self._enqueue_with_policy("my_queue", self._my_queue, chunk_pointer)
```

The `_enqueue_with_policy()` method uses the queue name to look up the policy from `QUEUE_POLICIES`.

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `dispatcher.py` | 820 | Central data router |
| `controller.py` | 690 | Pipeline lifecycle |
| `runtime.py` | 337 | Headless orchestrator |
| `device_registry.py` | 341 | Device discovery & management |
| `audio_manager.py` | 308 | Audio monitoring |
| `conditioning.py` | 276 | Signal filters |
| `runtime_readme.md` | 44 | Runtime usage docs |

---

## Thread Safety

- **Dispatcher** runs in its own thread, pulls from source thread
- **All queues** are thread-safe (`queue.Queue`)
- **Settings** use locks for concurrent access
- **Ring buffers** use `RLock` for thread-safe access (not lock-free)
