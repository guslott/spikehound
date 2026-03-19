# Analysis Module

Real-time event analysis for filtered DAQ data. The supported path is intentionally single-track:

`Dispatcher -> AnalysisWorker -> AnalysisBatch -> AnalysisTab`

This guide is the source of truth for adding analysis features. If a change does not fit this pipeline, it is probably reintroducing a legacy pattern.

## Supported Pipeline

### Data Flow

```text
DAQ Source
    ↓
SharedRingBuffer (raw samples)
    ↓
Dispatcher (filtering + per-channel fan-out)
    ↓
registered analysis queue (filtered Chunk objects)
    ↓
AnalysisWorker
    ↓
detection_to_analysis_event()
    ↓
AnalysisBatch(chunk, events)
    ↓
AnalysisTab
    ↓
metric records / bounded event details / scatter plot / user classes / export / STA
```

### Contract Rules

- `AnalysisWorker.output_queue` emits `AnalysisBatch` objects only.
- `AnalysisBatch` is the sole worker-to-UI payload for analysis tabs.
- `detection_to_analysis_event()` is the canonical conversion point from `DetectionEvent` to `AnalysisEvent`.
- `AnalysisTab` owns the bounded UI-side state derived from those events: overlays, scatter metrics, class labels, waveform detail cache, exports, and STA inputs.
- Do not add a second event-delivery bus, hidden chunk metadata fallback, or tab-side polling path.

## Where To Change Code

### Add a waveform-derived metric

1. Add a pure function to [metrics.py](/Users/guslott/Code/SpikeHound2/spikehound/analysis/metrics.py).
2. Compute it in [analysis_worker.py](/Users/guslott/Code/SpikeHound2/spikehound/analysis/analysis_worker.py) inside `detection_to_analysis_event()`.
3. Store it on `event.properties`.
4. If it should appear in the UI, wire it into [analysis_tab.py](/Users/guslott/Code/SpikeHound2/spikehound/gui/analysis_tab.py) selectors, labels, export, or class logic.

### Add or change a detector

1. Implement or extend a detector in `core/detection`.
2. Register it in `DETECTOR_REGISTRY`.
3. Configure it from `AnalysisWorker.configure_threshold()`.
4. Keep `detection_to_analysis_event()` as the only `AnalysisEvent` creation path.

### Add a new analysis-tab feature

Build it from existing `AnalysisTab` state instead of creating a parallel cache:

- `_metric_events` for bounded scatter/history records
- `_event_details` for bounded waveform/detail retention
- `_event_cluster_labels` for current class membership
- `_clusters` for user-defined ROIs bound to metric axes

## Quick Start

### Adding a new metric

```python
# In analysis/metrics.py

def spike_asymmetry(samples: np.ndarray, sr: float) -> float:
    if samples.size < 2 or sr <= 0:
        return 0.0
    peak = float(np.max(samples))
    trough = float(np.min(samples))
    denom = abs(peak) + abs(trough)
    if denom <= 1e-12:
        return 0.0
    return float((peak + trough) / denom)
```

```python
# In analysis/analysis_worker.py inside detection_to_analysis_event()

from .metrics import spike_asymmetry

props["spike_asymmetry"] = spike_asymmetry(wf, sr)
```

### Surfacing a metric in the analysis tab

Common touch points in [analysis_tab.py](/Users/guslott/Code/SpikeHound2/spikehound/gui/analysis_tab.py):

- metric selector labels for scatter axes
- metric extraction when building point records
- export columns
- details panel text

If the metric affects class membership or export behavior, update the relevant `AnalysisTab` helpers instead of creating a side channel.

## Core Types

### AnalysisBatch

```python
@dataclass(frozen=True)
class AnalysisBatch:
    chunk: Chunk
    events: Sequence[AnalysisEvent]
```

### AnalysisEvent

```python
@dataclass(frozen=True)
class AnalysisEvent:
    id: int
    channelId: int
    thresholdValue: float
    crossingIndex: int
    crossingTimeSec: float
    firstSampleTimeSec: float
    sampleRateHz: float
    windowMs: float
    preMs: float
    postMs: float
    samples: np.ndarray
    properties: Dict[str, float]
    intervalSinceLastSec: float
```

## Files That Matter

| File | Responsibility |
|------|----------------|
| [analysis_worker.py](/Users/guslott/Code/SpikeHound2/spikehound/analysis/analysis_worker.py) | Per-channel event detection and `AnalysisBatch` emission |
| [batch.py](/Users/guslott/Code/SpikeHound2/spikehound/analysis/batch.py) | Worker-to-UI payload contract |
| [metrics.py](/Users/guslott/Code/SpikeHound2/spikehound/analysis/metrics.py) | Pure metric functions |
| [analysis_tab.py](/Users/guslott/Code/SpikeHound2/spikehound/gui/analysis_tab.py) | Bounded UI state, scatter plot, classes, overlays, export, STA |
| [test_analysis_worker.py](/Users/guslott/Code/SpikeHound2/spikehound/test/test_analysis_worker.py) | Worker contract and waveform/metric correctness |
| [test_analysis_tab_event_pipeline.py](/Users/guslott/Code/SpikeHound2/spikehound/test/test_analysis_tab_event_pipeline.py) | End-to-end tab behavior from batch ingestion to classes/export/STA |

## Agent Workflow

When an agent adds an analysis feature, use this order:

1. Decide whether the feature is waveform-derived, detector-derived, or UI-derived.
2. Put waveform math in [metrics.py](/Users/guslott/Code/SpikeHound2/spikehound/analysis/metrics.py) as a pure function.
3. Wire event-level properties in `detection_to_analysis_event()` so every downstream consumer sees the same value.
4. Extend [analysis_tab.py](/Users/guslott/Code/SpikeHound2/spikehound/gui/analysis_tab.py) only for presentation, classification, export, or STA behavior.
5. Add worker coverage in [test_analysis_worker.py](/Users/guslott/Code/SpikeHound2/spikehound/test/test_analysis_worker.py).
6. Add tab/pipeline coverage in [test_analysis_tab_event_pipeline.py](/Users/guslott/Code/SpikeHound2/spikehound/test/test_analysis_tab_event_pipeline.py) if the change affects scatter points, classes, overlays, export, or STA.
7. Run the focused headless suite:

```bash
QT_QPA_PLATFORM=offscreen pytest -q test/test_analysis_worker.py test/test_analysis_tab_event_pipeline.py
```

## Checklists

### New Metric

- [ ] Pure function added to `metrics.py`
- [ ] Metric attached in `detection_to_analysis_event()`
- [ ] UI wiring added only where needed
- [ ] Export/class/STA behavior updated if applicable
- [ ] Worker tests added
- [ ] Tab pipeline tests added if user-visible

### New Detector

- [ ] Detector added under `core/detection`
- [ ] Registered in `DETECTOR_REGISTRY`
- [ ] Configurable through `AnalysisWorker`
- [ ] Still creates `AnalysisEvent` objects only through `detection_to_analysis_event()`
- [ ] Threshold/manual/auto edge cases covered by tests

## FAQ

**Q: Where should a new event property be computed?**

If it comes from the waveform and should be shared by scatter/classes/export/STA, compute it in `detection_to_analysis_event()`.

**Q: Should I store extra events in chunk metadata or a side queue?**

No. Use `AnalysisBatch.events`. That is the supported contract.

**Q: Where should class-aware behavior live?**

In `AnalysisTab`, using `_event_cluster_labels` and the bounded event-detail/metric state already maintained there.
