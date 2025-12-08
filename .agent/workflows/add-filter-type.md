---
description: Add a new signal filter type for conditioning
---

# Add a New Signal Filter

Follow these steps to add a new filter type to the signal conditioning pipeline.

## Prerequisites
- Understanding of the filter design (scipy.signal or custom)
- Review `core/conditioning.py` for existing patterns

## Steps

1. **Add filter parameters to `ChannelFilterSettings`**
   
   In `core/conditioning.py`:
   ```python
   @dataclass
   class ChannelFilterSettings:
       # Existing fields...
       highpass_enabled: bool = False
       highpass_freq: float = 10.0
       
       # Add your new filter:
       myfilter_enabled: bool = False
       myfilter_param1: float = 100.0
       myfilter_param2: float = 0.5
   ```

2. **Add filter computation in `SignalConditioner`**
   
   In `core/conditioning.py`, find `_apply_channel_filters()`:
   ```python
   def _apply_channel_filters(self, data: np.ndarray, settings: ChannelFilterSettings) -> np.ndarray:
       # Existing filters...
       
       # Add your filter:
       if settings.myfilter_enabled:
           data = self._apply_myfilter(data, settings.myfilter_param1, settings.myfilter_param2)
       
       return data
   
   def _apply_myfilter(self, data: np.ndarray, param1: float, param2: float) -> np.ndarray:
       """Apply my custom filter.
       
       Args:
           data: Input signal array (samples,)
           param1: First parameter
           param2: Second parameter
           
       Returns:
           Filtered signal
       """
       # Design filter (cache if possible)
       # sos = scipy.signal.butter(order, freq, btype, fs=self._sample_rate, output='sos')
       
       # Apply filter
       # return scipy.signal.sosfiltfilt(sos, data)
       
       return data  # Replace with actual implementation
   ```

3. **Add UI controls in `ChannelDetailPanel`**
   
   In `gui/channel_controls_widget.py`:
   
   a. Add to `ChannelConfig` in `gui/types.py`:
   ```python
   @dataclass
   class ChannelConfig:
       # Existing fields...
       myfilter_enabled: bool = False
       myfilter_param1: float = 100.0
   ```
   
   b. Add controls in `_build_ui()`:
   ```python
   self.myfilter_checkbox = QtWidgets.QCheckBox("My Filter")
   self.myfilter_spin = QtWidgets.QDoubleSpinBox()
   self.myfilter_spin.setRange(1.0, 1000.0)
   layout.addWidget(self.myfilter_checkbox)
   layout.addWidget(self.myfilter_spin)
   ```
   
   c. Connect signals in `_connect_signals()`:
   ```python
   self.myfilter_checkbox.toggled.connect(self._on_myfilter_toggled)
   self.myfilter_spin.valueChanged.connect(self._on_widgets_changed)
   ```
   
   d. Add handler:
   ```python
   def _on_myfilter_toggled(self, checked: bool) -> None:
       self._on_widgets_changed()
   ```
   
   e. Include in `_on_widgets_changed()`:
   ```python
   self._config = replace(self._config,
       myfilter_enabled=self.myfilter_checkbox.isChecked(),
       myfilter_param1=self.myfilter_spin.value(),
   )
   ```

4. **Wire through to dispatcher**
   
   Ensure `FilterSettings` construction in `MainWindow._sync_filter_settings()` includes your new parameters.

5. **Test the filter**
   - Enable filter on a channel
   - Verify signal is modified as expected
   - Check for latency/artifacts

## Verification Checklist
- [ ] Filter parameters added to dataclass
- [ ] Filter implementation works correctly
- [ ] UI controls appear and function
- [ ] Filter applies to live signal
- [ ] No audio glitches or visual artifacts
- [ ] Settings persist across sessions (if using config save/load)
