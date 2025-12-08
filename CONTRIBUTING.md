# Contributing to SpikeHound

> **SpikeHound as a Platform**: This guide focuses on using SpikeHound as a development platform for custom experiments, lessons, and specialized applications. Whether you're adding digital I/O, custom analysis, or entirely new experiment workflows—with or without contributing back to the main codebase.

---

## 🤖 AI-Assisted Development Workflow

SpikeHound is designed for **AI-assisted development**. The codebase includes extensive documentation, consistent patterns, and workflow files that enable AI coding assistants to understand and extend the system.

### The Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   You describe  │ ──▶ │   AI reads the  │ ──▶ │  AI implements  │
│   what you want │     │  documentation  │     │   the feature   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │  You verify it  │
                                              │     works       │
                                              └─────────────────┘
```

### Quick Reference: Slash Commands

Point your AI assistant to these workflow files:

| Command | Use Case |
|---------|----------|
| `/add-gui-tab` | Create a new experiment control tab |
| `/add-daq-driver` | Add support for new hardware |
| `/add-analysis-metric` | Add measurement calculations |
| `/add-filter-type` | Add signal conditioning options |

### Documentation Locations

| Document | Path | Purpose |
|----------|------|---------|
| Core Pipeline | `core/core_readme.md` | Data flow architecture |
| Data Types | `shared/shared_readme.md` | Chunk, Event, and Config types |
| Runtime API | `core/runtime_readme.md` | High-level orchestration |
| DAQ Drivers | `daq/daq_readme.md` | Hardware device patterns |
| Analysis | `analysis/analysis_readme.md` | Event detection and metrics |
| GUI Widgets | `gui/gui_readme.md` | User interface patterns |
| Dispatcher | `doc/dispatcher_readme.md` | Data routing details |

---

## 🧪 Creating Custom Experiment Tabs

The tab system is the primary extension point for custom experiments. Each tab can receive real-time data from the pipeline and provide specialized visualization, analysis, or control interfaces.

### Architecture: How Tabs Connect to Data

```
┌────────────────────────────────────────────────────────────┐
│                     MainWindow                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    TabWidget                          │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────────┐ │  │
│  │  │  Scope  │ │Analysis │ │Settings │ │ YOUR TAB   │ │  │
│  │  └────┬────┘ └────┬────┘ └─────────┘ └─────┬──────┘ │  │
│  └───────│───────────│────────────────────────│────────┘  │
│          │           │                        │            │
│          ▼           ▼                        ▼            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         PipelineController (via MainWindow)          │  │
│  │  ┌────────────────┐  ┌────────────────┐             │  │
│  │  │ Visualization  │  │    Analysis    │             │  │
│  │  │    Queue       │  │     Queue      │             │  │
│  │  └────────────────┘  └────────────────┘             │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Step-by-Step: Create an Experiment Tab

#### 1. Create Your Tab File

```python
# gui/my_experiment_tab.py
"""Custom experiment control tab."""
from __future__ import annotations

from typing import Optional
from PySide6 import QtCore, QtWidgets, QtGui
import numpy as np
import pyqtgraph as pg


class MyExperimentTab(QtWidgets.QWidget):
    """Tab for [your experiment description].
    
    This tab receives real-time data from the pipeline and provides
    custom visualization and control for your specific experiment.
    
    Signals:
        triggerRequested: Emitted when user wants to trigger an event
        parameterChanged: Emitted when experiment parameters change
    """
    
    # Qt signals for communicating with MainWindow
    triggerRequested = QtCore.Signal()
    parameterChanged = QtCore.Signal(dict)  # {param_name: value}
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._data_buffer: list[np.ndarray] = []
        self._sample_rate: float = 20000.0
        self._setup_ui()
        self._connect_signals()
        self._setup_timers()
    
    def _setup_ui(self) -> None:
        """Build the tab layout."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- Control Panel ---
        controls = QtWidgets.QHBoxLayout()
        
        self.start_btn = QtWidgets.QPushButton("Start Experiment")
        self.start_btn.setCheckable(True)
        controls.addWidget(self.start_btn)
        
        self.param_spin = QtWidgets.QDoubleSpinBox()
        self.param_spin.setRange(0.0, 100.0)
        self.param_spin.setValue(50.0)
        self.param_spin.setPrefix("Threshold: ")
        controls.addWidget(self.param_spin)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # --- Visualization Area ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude', units='V')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.curve = self.plot_widget.plot(pen='y')
        layout.addWidget(self.plot_widget)
        
        # --- Results Panel ---
        results = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QFormLayout(results)
        self.count_label = QtWidgets.QLabel("0")
        self.rate_label = QtWidgets.QLabel("0.0 Hz")
        results_layout.addRow("Events:", self.count_label)
        results_layout.addRow("Rate:", self.rate_label)
        layout.addWidget(results)
    
    def _connect_signals(self) -> None:
        """Wire internal signals to handlers."""
        self.start_btn.toggled.connect(self._on_start_toggled)
        self.param_spin.valueChanged.connect(self._on_param_changed)
    
    def _setup_timers(self) -> None:
        """Set up periodic update timers."""
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.timeout.connect(self._update_display)
        self._update_timer.setInterval(50)  # 20 Hz update
    
    # --- Public API (called by MainWindow) ---
    
    def set_sample_rate(self, rate: float) -> None:
        """Update the sample rate for time calculations."""
        self._sample_rate = rate
    
    def handle_chunk(self, chunk) -> None:
        """Process incoming data chunk from pipeline.
        
        Called by MainWindow when new data arrives.
        The chunk contains:
          - chunk.samples: numpy array of shape (channels, samples)
          - chunk.dt: sample interval in seconds
          - chunk.channel_names: tuple of channel names
        """
        # Buffer recent data for display
        self._data_buffer.append(chunk.samples[0])  # First channel
        # Keep last N seconds
        max_samples = int(2.0 * self._sample_rate)
        while sum(len(d) for d in self._data_buffer) > max_samples:
            self._data_buffer.pop(0)
    
    def handle_event(self, event) -> None:
        """Process detected events from analysis.
        
        Called by MainWindow when events are detected.
        """
        # Update event count
        count = int(self.count_label.text()) + 1
        self.count_label.setText(str(count))
    
    # --- Private handlers ---
    
    def _on_start_toggled(self, checked: bool) -> None:
        """Handle experiment start/stop."""
        if checked:
            self.start_btn.setText("Stop Experiment")
            self._update_timer.start()
            self._data_buffer.clear()
        else:
            self.start_btn.setText("Start Experiment")
            self._update_timer.stop()
    
    def _on_param_changed(self, value: float) -> None:
        """Handle parameter changes."""
        self.parameterChanged.emit({"threshold": value})
    
    def _update_display(self) -> None:
        """Periodic display update."""
        if not self._data_buffer:
            return
        
        # Concatenate buffered data
        data = np.concatenate(self._data_buffer)
        time = np.arange(len(data)) / self._sample_rate
        
        # Update plot
        self.curve.setData(time, data)
```

#### 2. Register in MainWindow

In `gui/main_window.py`:

```python
# At top of file
from .my_experiment_tab import MyExperimentTab

# In _init_ui(), where tabs are created
self.experiment_tab = MyExperimentTab()
self.tab_widget.addTab(self.experiment_tab, "My Experiment")

# Wire up signals
self.experiment_tab.parameterChanged.connect(self._on_experiment_param_changed)

# Add handler method
def _on_experiment_param_changed(self, params: dict) -> None:
    """Handle parameter changes from experiment tab."""
    # Apply parameters to pipeline, trigger, etc.
    if "threshold" in params:
        self._pipeline.update_trigger_config(
            threshold=params["threshold"]
        )
```

#### 3. Connect to Data Flow

To receive real-time data in your tab, subscribe to the dispatcher:

```python
# In MainWindow._bind_dispatcher_signals() or similar
def _on_visualization_chunk(self, chunk) -> None:
    # Forward to experiment tab
    self.experiment_tab.handle_chunk(chunk)
    
# Or use QTimer to poll the queue directly in your tab
```

---

## 🔌 Adding Custom Hardware / Digital I/O

SpikeHound's DAQ layer is designed for easy extension. Each device driver follows the same pattern, making it straightforward to add support for new hardware.

### Common Extension Scenarios

| Scenario | Approach |
|----------|----------|
| Digital outputs for stimulation | Add methods to your DAQ driver |
| External trigger inputs | Use the device's callback/interrupt system |
| Multi-device synchronization | Coordinate via shared timing source |
| Custom analog outputs | Extend the driver with output capabilities |

### Example: Adding Digital I/O to a DAQ Driver

```python
# daq/my_daq_with_dio.py
"""DAQ driver with digital I/O support."""
from __future__ import annotations

import numpy as np
from .base_device import BaseDevice

class MyDAQWithDIO(BaseDevice):
    """DAQ device with digital input/output capabilities.
    
    Extends the standard BaseDevice pattern to include:
    - Digital output for stimulus control
    - Digital input for external triggers
    """
    
    def __init__(self) -> None:
        super().__init__("My DAQ", "My DAQ with Digital I/O")
        self._dio_state: int = 0
        self._trigger_callback = None
    
    # --- Standard DAQ hooks ---
    
    def _open_impl(self, device_id: str) -> None:
        # Open device handle
        self._handle = my_sdk.open(device_id)
    
    def _configure_impl(self, sample_rate, channels, chunk_size, **opts):
        # Configure analog input
        my_sdk.configure_ai(self._handle, sample_rate, channels, chunk_size)
        # Configure digital lines
        my_sdk.configure_dio(self._handle, lines=[0, 1, 2, 3])
        return sample_rate
    
    def _start_impl(self) -> None:
        # Start streaming thread
        self._running = True
        self._thread = threading.Thread(target=self._producer_loop)
        self._thread.start()
    
    def _stop_impl(self) -> None:
        self._running = False
        self._thread.join()
    
    def _close_impl(self) -> None:
        my_sdk.close(self._handle)
    
    # --- Digital I/O Extensions ---
    
    def set_digital_output(self, line: int, state: bool) -> None:
        """Set a digital output line high or low.
        
        Use for controlling stimulation, LED indicators, or
        external equipment synchronization.
        
        Args:
            line: Digital output line number (0-3)
            state: True for high, False for low
        """
        if state:
            self._dio_state |= (1 << line)
        else:
            self._dio_state &= ~(1 << line)
        my_sdk.write_dio(self._handle, self._dio_state)
    
    def pulse_digital_output(self, line: int, duration_ms: float) -> None:
        """Generate a brief pulse on a digital output line.
        
        Useful for triggering oscilloscopes, cameras, or stimulators.
        """
        self.set_digital_output(line, True)
        time.sleep(duration_ms / 1000.0)
        self.set_digital_output(line, False)
    
    def register_trigger_callback(self, callback) -> None:
        """Register a callback for external trigger events.
        
        The callback receives the timestamp when the trigger occurred.
        """
        self._trigger_callback = callback
        my_sdk.enable_trigger_interrupt(self._handle, self._on_trigger)
    
    def _on_trigger(self, timestamp: float) -> None:
        """Internal: handle trigger interrupt."""
        if self._trigger_callback:
            self._trigger_callback(timestamp)
```

### Exposing Device Features in Your Tab

```python
class StimulusControlTab(QtWidgets.QWidget):
    """Control tab for experiment with external stimulation."""
    
    def __init__(self, device) -> None:
        super().__init__()
        self._device = device  # Reference to DAQ with DIO
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        
        self.stim_btn = QtWidgets.QPushButton("Deliver Stimulus")
        self.stim_btn.clicked.connect(self._on_stim_clicked)
        layout.addWidget(self.stim_btn)
        
        self.pulse_spin = QtWidgets.QSpinBox()
        self.pulse_spin.setRange(1, 100)
        self.pulse_spin.setValue(10)
        self.pulse_spin.setSuffix(" ms")
        layout.addWidget(self.pulse_spin)
    
    def _on_stim_clicked(self) -> None:
        """Deliver a stimulus pulse."""
        duration = self.pulse_spin.value()
        # Run in background to not block UI
        QtCore.QTimer.singleShot(0, lambda: 
            self._device.pulse_digital_output(0, duration)
        )
```

---

## 📊 Custom Analysis Pipelines

For specialized analysis beyond the built-in spike detection, you can create custom analysis workers that process the data stream.

### Pattern: Custom Analysis Worker

```python
# analysis/my_analysis_worker.py
"""Custom analysis for burst detection."""
from __future__ import annotations

import threading
import queue
import numpy as np
from typing import Optional

from shared.models import Chunk


class BurstDetectionWorker:
    """Background worker for detecting burst patterns.
    
    Consumes Chunks from a queue and emits burst events when
    multiple spikes occur within a short time window.
    """
    
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue):
        self._input = input_queue
        self._output = output_queue
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Burst detection parameters
        self.min_spikes = 3
        self.max_interval_ms = 20.0
        self._recent_spike_times: list[float] = []
    
    def start(self) -> None:
        """Start the analysis thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the analysis thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _run(self) -> None:
        """Main analysis loop."""
        while self._running:
            try:
                chunk = self._input.get(timeout=0.1)
                self._process_chunk(chunk)
            except queue.Empty:
                continue
    
    def _process_chunk(self, chunk: Chunk) -> None:
        """Analyze a chunk for burst patterns."""
        # Your burst detection logic here
        # When burst detected, emit to output queue:
        # self._output.put(BurstEvent(...))
        pass
```

---

## 🎓 Building Educational Lessons

SpikeHound is ideal for creating structured lab exercises. Here's a pattern for lesson modules:

### Pattern: Self-Contained Lesson Tab

```python
# gui/lessons/action_potential_lesson.py
"""Interactive lesson on action potentials."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets, QtGui


class ActionPotentialLesson(QtWidgets.QWidget):
    """Interactive lesson: Understanding Action Potentials.
    
    Guides students through:
    1. Observing resting membrane potential
    2. Triggering action potentials
    3. Measuring spike properties
    4. Understanding refractory periods
    """
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_step = 0
        self._steps = [
            self._step_introduction,
            self._step_observe_resting,
            self._step_trigger_spike,
            self._step_measure_properties,
            self._step_refractory_period,
            self._step_conclusion,
        ]
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        
        # Instructions panel
        self.instruction_text = QtWidgets.QTextEdit()
        self.instruction_text.setReadOnly(True)
        self.instruction_text.setMaximumHeight(150)
        layout.addWidget(self.instruction_text)
        
        # Visualization area (embedded scope or custom)
        self.viz_area = QtWidgets.QWidget()
        layout.addWidget(self.viz_area, stretch=1)
        
        # Navigation
        nav = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("← Previous")
        self.prev_btn.clicked.connect(self._prev_step)
        nav.addWidget(self.prev_btn)
        
        self.step_label = QtWidgets.QLabel("Step 1 of 6")
        nav.addWidget(self.step_label, alignment=QtCore.Qt.AlignCenter)
        
        self.next_btn = QtWidgets.QPushButton("Next →")
        self.next_btn.clicked.connect(self._next_step)
        nav.addWidget(self.next_btn)
        
        layout.addLayout(nav)
        
        # Initialize first step
        self._update_step()
    
    def _update_step(self) -> None:
        """Display current step content."""
        self.step_label.setText(f"Step {self._current_step + 1} of {len(self._steps)}")
        self.prev_btn.setEnabled(self._current_step > 0)
        self.next_btn.setEnabled(self._current_step < len(self._steps) - 1)
        
        # Run step setup
        self._steps[self._current_step]()
    
    def _prev_step(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step()
    
    def _next_step(self) -> None:
        if self._current_step < len(self._steps) - 1:
            self._current_step += 1
            self._update_step()
    
    def _step_introduction(self) -> None:
        self.instruction_text.setHtml("""
        <h3>Welcome to the Action Potential Lab!</h3>
        <p>In this lesson, you will learn to:</p>
        <ul>
            <li>Identify the resting membrane potential</li>
            <li>Observe action potential waveforms</li>
            <li>Measure spike amplitude and duration</li>
            <li>Understand the refractory period</li>
        </ul>
        <p>Click <b>Next</b> to begin.</p>
        """)
    
    def _step_observe_resting(self) -> None:
        self.instruction_text.setHtml("""
        <h3>Step 1: Observe Resting Potential</h3>
        <p>Look at the oscilloscope display. The flat baseline you see 
        represents the <b>resting membrane potential</b>.</p>
        <p><i>Question: What is the approximate voltage of the resting potential?</i></p>
        """)
    
    # ... additional step methods ...
```

---

## 🔧 Development Environment Setup

### Recommended Setup for AI-Assisted Development

1. **Clone and install**:
   ```bash
   git clone https://github.com/guslott/spikehound.git
   cd spikehound
   pip install numpy scipy PySide6 pyqtgraph miniaudio pyserial
   ```

2. **Configure your AI assistant** (Claude, Cursor, Copilot, etc.):
   - Open the project root as your workspace
   - Point the AI to relevant documentation per task
   - Use slash commands when available

3. **Test your changes**:
   ```bash
   python -m pytest test/ -v
   python main.py  # Launch and verify manually
   ```

### Integration Tests as Reference

See `test/test_integration.py` for patterns on:
- Setting up the pipeline programmatically
- Verifying data flow
- Testing filter and trigger propagation
- Checking health metrics

---

## 📚 Quick Reference: Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `MainWindow` | `gui/main_window.py` | Central UI orchestrator |
| `PipelineController` | `core/controller.py` | Pipeline lifecycle |
| `Dispatcher` | `core/dispatcher.py` | Data routing |
| `SignalConditioner` | `core/conditioning.py` | Filter processing |
| `AnalysisWorker` | `analysis/analysis_worker.py` | Spike detection |
| `BaseDevice` | `daq/base_device.py` | Hardware interface |

| Data Type | Module | Purpose |
|-----------|--------|---------|
| `Chunk` | `shared/models.py` | Raw data packets |
| `DetectionEvent` | `shared/models.py` | Detected spikes |
| `FilterSettings` | `core/conditioning.py` | Filter configuration |
| `TriggerConfig` | `core/models.py` | Trigger settings |

---

## 💡 Tips for Forking as a Platform

### Keep Your Customizations Separate

```
spikehound/
├── gui/
│   ├── [core files]
│   └── my_experiments/     # Your custom tabs
│       ├── __init__.py
│       ├── burst_lesson.py
│       └── muscle_fatigue.py
├── daq/
│   ├── [core files]
│   └── my_hardware/        # Your custom drivers
│       ├── __init__.py
│       └── arduino_dio.py
└── analysis/
    ├── [core files]
    └── my_analysis/        # Your custom analysis
        ├── __init__.py
        └── burst_detector.py
```

### Staying Up-to-Date

If you want to pull updates from upstream while keeping your customizations:

```bash
# Set up upstream remote
git remote add upstream https://github.com/guslott/spikehound.git

# Fetch and merge updates
git fetch upstream
git merge upstream/main

# Resolve any conflicts in your custom files
```

### No Contribution Required

**This is designed as a platform.** You are welcome to:
- Fork and customize without contributing back
- Build proprietary experiment modules on top
- Use for commercial educational products
- Adapt for your specific research needs

See `LICENSE` for terms (MIT License).

---

## 🤝 Contributing Back (Optional)

If you do want to contribute improvements to the core:

1. **Fork** the repository
2. **Create a branch** for your feature
3. **Follow existing patterns** (documentation, tests, style)
4. **Submit a Pull Request** with clear description

### Contribution Priorities

| Priority | Type |
|----------|------|
| High | Bug fixes, documentation improvements |
| Medium | New device drivers with broad appeal |
| Lower | Highly specialized experiment modules |

---

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/guslott/spikehound/issues)
- **Discussions**: [GitHub Discussions](https://github.com/guslott/spikehound/discussions)
- **AI Assistance**: Point your AI to the relevant `*_readme.md` files

---

*SpikeHound is developed by Gus K. Lott III with support from Manlius Pebble Hill School and the Cornell Neuroethology Lab.*
