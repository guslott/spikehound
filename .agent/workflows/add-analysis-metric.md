---
description: Add a new analysis metric for event characterization
---

# Add a New Analysis Metric

Follow these steps to add a new metric that will be computed for each detected event.

## Prerequisites
- Understanding of the metric computation (formula, input requirements)
- Review `analysis/analysis_readme.md` for patterns

## Steps

1. **Create the metric function in `analysis/metrics.py`**
   ```python
   def my_metric(samples: np.ndarray, sr: float) -> float:
       """Compute my custom metric.
       
       Args:
           samples: 1D array of waveform samples around event
           sr: Sample rate in Hz
           
       Returns:
           Computed metric value (0.0 for invalid inputs)
       """
       # Handle edge cases
       if samples.size == 0 or sr <= 0:
           return 0.0
       
       # Your computation
       result = ...
       
       return float(result)
   ```

2. **Add the metric to event detection in `analysis/analysis_worker.py`**
   - Find the `_detect_events()` method
   - Import your metric: `from .metrics import my_metric`
   - Compute and store in event properties:
   ```python
   my_value = my_metric(waveform_samples, sample_rate)
   properties["my_metric"] = my_value
   ```

3. **(Optional) Display in GUI**
   - In `gui/analysis_tab.py`, add a label or display for the metric
   - Access via `event.properties.get("my_metric", 0.0)`

4. **Test the metric**
   ```python
   # In test file or interactive:
   from analysis.metrics import my_metric
   import numpy as np
   
   # Test with known signal
   samples = np.sin(np.linspace(0, 2*np.pi, 100))
   result = my_metric(samples, 10000.0)
   assert result > 0  # or expected value
   ```

## Verification Checklist
- [ ] Function handles empty arrays gracefully
- [ ] Function handles zero/negative sample rate
- [ ] Returns float type
- [ ] Metric appears in event.properties
- [ ] (If GUI display added) Metric displays correctly
