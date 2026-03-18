# Analysis Module

Real-time spike detection and event analysis for neural signals. Processes streaming data from the DAQ pipeline, detects threshold crossings, and computes event metrics.

* Simple threshold detection with auto-thresholding
* Refractory period enforcement
* Waveform capture around detection points
* Centralized metric computation (envelope, frequency, amplitude)
* Thread-safe settings with observer pattern

> This README covers analysis patterns for adding new detection algorithms and metrics.

---

## Quick Start

### Adding a new metric

```python
# In analysis/metrics.py

def my_metric(samples: np.ndarray, sr: float) -> float:
    """Compute my custom metric.
    
    Args:
        samples: 1D array of waveform samples around event
        sr: Sample rate in Hz
        
    Returns:
        Computed metric value
    """
    if samples.size == 0 or sr <= 0:
        return 0.0
    # Your computation here
    return float(result)
```

### Using a metric in the analysis worker

```python
# In analysis/analysis_worker.py, within _detect_events():

from .metrics import my_metric

# After extracting waveform:
my_value = my_metric(waveform_samples, sample_rate)
event.properties["my_metric"] = my_value
```

---

## Architecture

### Data Flow

```
DAQ Source
    ↓
SharedRingBuffer (raw samples)
    ↓
Dispatcher (filtering)
    ↓
analysis_queue (filtered Chunk objects)
    ↓
┌─────────────────────────────────────┐
│         Analysis Layer              │
│                                     │
│  AnalysisWorker                     │
│  (supported per-channel worker)     │
│         ↓                           │
│  AmpThresholdDetector               │
│  (manual / auto thresholding)       │
│         ↓                           │
│  Event detection + metrics          │
│         ↓                           │
│  event_queue → GUI (AnalysisTab)    │
└─────────────────────────────────────┘
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `AnalysisWorker` | `analysis_worker.py` | Supported per-channel analysis worker for GUI |
| `AmpThresholdDetector` | `../core/detection/threshold.py` | Supported threshold detector used by `AnalysisWorker` |
| `AnalysisBatch` | `batch.py` | Worker-to-GUI payload carrying a chunk and its detected events |
| `AnalysisSettings` | `settings.py` | Thread-safe settings with observers |
| `metrics.py` | `metrics.py` | Centralized metric functions |

---

## Data Types

### AnalysisBatch (analysis/batch.py)

Worker output passed to the GUI:

```python
@dataclass(frozen=True)
class AnalysisBatch:
    chunk: Chunk
    events: Sequence[AnalysisEvent]
```

### AnalysisEvent (shared/types.py)

Detailed event for GUI display (analysis layer):

```python
@dataclass(frozen=True)
class AnalysisEvent:
    id: int                       # Unique event ID
    channelId: int                # Source channel
    thresholdValue: float         # Threshold that was crossed
    crossingIndex: int            # Absolute sample index
    crossingTimeSec: float        # Timestamp of crossing
    firstSampleTimeSec: float     # Timestamp of window start
    sampleRateHz: float           # For time conversions
    windowMs: float               # Total window duration
    preMs: float                  # Pre-crossing portion
    postMs: float                 # Post-crossing portion
    samples: np.ndarray           # Waveform around crossing
    properties: Dict[str, float]  # Computed metrics
    intervalSinceLastSec: float   # Time since previous event
```

### AnalysisSettings (analysis/settings.py)

Runtime-configurable parameters:

```python
@dataclass(frozen=True)
class AnalysisSettings:
    event_window_ms: float = 10.0  # Default event window width
```

---

## Metric Functions

All metrics are **pure functions** in `analysis/metrics.py`:

| Function | Signature | Description |
|----------|-----------|-------------|
| `baseline` | `(samples, pre_samples) → float` | Median baseline from pre-event samples |
| `envelope` | `(samples) → float` | Signal envelope (max - min) |
| `min_max` | `(samples) → (max, min)` | Peak amplitude extraction |
| `peak_frequency_sinc` | `(samples, sr, min_hz, center_index) → float` | FFT + sinc interpolation frequency |
| `autocorr_frequency` | `(segment, sr, min_hz, max_hz) → float` | Autocorrelation frequency fallback |

### Adding a New Metric

1. **Create the function in `metrics.py`**:

```python
def spike_width(samples: np.ndarray, sr: float, threshold_frac: float = 0.5) -> float:
    """Compute spike width at half-maximum.
    
    Args:
        samples: 1D waveform array
        sr: Sample rate in Hz
        threshold_frac: Fraction of peak for width measurement
        
    Returns:
        Width in milliseconds
    """
    if samples.size < 3 or sr <= 0:
        return 0.0
    
    peak_idx = int(np.argmax(np.abs(samples)))
    peak_val = np.abs(samples[peak_idx])
    threshold = peak_val * threshold_frac
    
    # Find crossings
    above = np.abs(samples) >= threshold
    # ... (compute width)
    
    return float(width_samples / sr * 1000.0)  # Convert to ms
```

2. **Add to `__all__` export** (if module has one):

```python
__all__ = ["baseline", "envelope", "min_max", "peak_frequency_sinc", "spike_width"]
```

3. **Use in AnalysisWorker._detect_events()**:

```python
from .metrics import spike_width

# In event detection loop:
width = spike_width(waveform, sample_rate)
properties["spike_width_ms"] = width
```

4. **Display in AnalysisTab** (if desired):

```python
# In gui/analysis_tab.py, add to metrics display
self.width_label.setText(f"Width: {event.properties.get('spike_width_ms', 0):.2f} ms")
```

---

## Detection Patterns

### Threshold Detection (Basic)

```python
def detect_threshold_crossings(
    samples: np.ndarray,
    threshold: float,
    polarity: str = "neg",
    refractory_samples: int = 30,
) -> list[int]:
    """Detect threshold crossings with refractory enforcement.
    
    Args:
        samples: 1D signal array
        threshold: Detection threshold (absolute value for neg/pos)
        polarity: "neg" (below), "pos" (above), or "both"
        refractory_samples: Minimum samples between detections
        
    Returns:
        List of crossing indices
    """
    crossings = []
    last_crossing = -refractory_samples
    
    for i in range(1, len(samples)):
        # Check refractory
        if i - last_crossing < refractory_samples:
            continue
            
        # Check crossing based on polarity
        if polarity == "neg":
            crossed = samples[i-1] > -threshold and samples[i] <= -threshold
        elif polarity == "pos":
            crossed = samples[i-1] < threshold and samples[i] >= threshold
        else:  # both
            crossed = (abs(samples[i-1]) < threshold and 
                      abs(samples[i]) >= threshold)
        
        if crossed:
            crossings.append(i)
            last_crossing = i
    
    return crossings
```

### Auto-Thresholding (MAD-based)

```python
def compute_auto_threshold(samples: np.ndarray, k_sigma: float = 5.0) -> float:
    """Compute threshold from noise estimate using Median Absolute Deviation.
    
    MAD is robust to outliers (spikes) unlike standard deviation.
    sigma ≈ 1.4826 * MAD for Gaussian noise.
    
    Args:
        samples: 1D signal array (should be longer than typical spike)
        k_sigma: Multiplier for noise estimate
        
    Returns:
        Absolute threshold value
    """
    median = np.median(samples)
    mad = np.median(np.abs(samples - median))
    sigma = 1.4826 * mad
    return k_sigma * sigma
```

## AnalysisSettingsStore Pattern

Thread-safe settings with observer notifications:

```python
from analysis.settings import AnalysisSettings, AnalysisSettingsStore

# Create store
store = AnalysisSettingsStore()

# Subscribe to changes
def on_settings_changed(settings: AnalysisSettings):
    print(f"Window changed to {settings.event_window_ms} ms")

unsubscribe = store.subscribe(on_settings_changed, replay=True)

# Update settings (notifies all subscribers)
store.update(event_window_ms=10.0)

# Get current settings
current = store.get()

# Unsubscribe when done
unsubscribe()
```

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `analysis_worker.py` | 674 | Per-channel analysis worker for GUI integration |
| `batch.py` | 18 | Worker-to-GUI batch payload |
| `metrics.py` | 168 | Centralized metric functions |
| `settings.py` | 63 | Thread-safe settings store |
| `__init__.py` | 4 | Package exports |

---

## Checklist for New Detection Algorithm

- [ ] Create detector class or function
- [ ] Handle edge cases (empty arrays, invalid sample rate)
- [ ] Enforce refractory period between detections
- [ ] Create AnalysisEvent objects with proper timing metadata
- [ ] Compute and attach relevant metrics to `event.properties`
- [ ] Add tests with synthetic signals
- [ ] Document expected input/output formats

---

## Checklist for New Metric

- [ ] Create pure function in `metrics.py`
- [ ] Accept `samples: np.ndarray` and `sr: float` at minimum
- [ ] Handle edge cases (empty array, zero sample rate)
- [ ] Return `float` (use 0.0 for invalid inputs)
- [ ] Add to event's `properties` dict in `_detect_events()`
- [ ] Display in GUI if user-facing
- [ ] Add unit test with known input/output

---

## FAQ

**Q: What is the supported event-detection path?**

The supported path is `AnalysisWorker` with `AmpThresholdDetector`, wired through the GUI and `PipelineController`.

**Q: How do I add a new detection algorithm?**

Create a new detector class following `AmpThresholdDetector` (in `core/detection.py`), register it in `DETECTOR_REGISTRY`, and configure it via `AnalysisWorker.configure_threshold()`.

**Q: Why are metrics pure functions?**

Pure functions are easier to test, don't require state management, and can be composed or called from multiple locations (worker, GUI, tests) without side effects.

**Q: How do I tune auto-threshold sensitivity?**

In the supported path, tune the detector factor configured through `AnalysisWorker.configure_threshold()`; the canonical default is `5.0` sigma.
