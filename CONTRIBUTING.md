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

The plugin tab system is the primary extension point for custom experiments. Tabs are loaded from `gui/tabs/` and can receive real-time payloads from the dispatcher while also pushing config changes back through the runtime.

### Architecture: How Tabs Connect to Data

```
┌────────────────────────────────────────────────────────────────────┐
│                            MainWindow                              │
│  ┌──────────────────────────┐   ┌───────────────────────────────┐ │
│  │     TabPluginManager     │──▶│         AnalysisDock          │ │
│  │  (discovers gui/tabs/*)  │   │ QTabWidget: Scope/Settings/...│ │
│  └──────────────────────────┘   │ + YOUR PLUGIN TAB             │ │
│                                 └──────────────┬────────────────┘ │
│                                                │                  │
│                                                ▼                  │
│                              SpikeHoundRuntime / PipelineController│
│                                                │                  │
│                                                ▼                  │
│                         Dispatcher tick payloads (`samples`, etc.) │
└────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step: Create an Experiment Tab

#### 1. Create Your Tab File

```python
# gui/tabs/my_experiment_tab.py
"""Custom experiment control tab."""
from __future__ import annotations

from typing import Optional
from PySide6 import QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg

from gui.dispatcher_adapter import connect_dispatcher_signals
from gui.tab_plugin_manager import BaseTab
from shared.models import TriggerConfig

class MyExperimentTab(BaseTab):
    """Tab for [your experiment description].
    
    This tab receives real-time data from dispatcher tick payloads and provides
    custom visualization and control for your specific experiment.
    """

    TAB_TITLE = "My Experiment"

    def __init__(self, runtime, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(runtime, parent)
        self._dispatcher_signals = None
        self._dispatcher_unsubscribe = None
        self._data_buffer: list[np.ndarray] = []
        self._sample_rate: float = 10_000.0
        self._setup_ui()
        self._bind_dispatcher_if_available()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.param_spin = QtWidgets.QDoubleSpinBox()
        self.param_spin.setRange(-10.0, 10.0)
        self.param_spin.setValue(0.0)
        self.param_spin.setPrefix("Threshold: ")
        self.param_spin.valueChanged.connect(self._on_threshold_changed)
        layout.addWidget(self.param_spin)

        self.plot_widget = pg.PlotWidget()
        self.curve = self.plot_widget.plot(pen="y")
        layout.addWidget(self.plot_widget)

    def _bind_dispatcher_if_available(self) -> None:
        dispatcher = self.runtime.dispatcher
        if dispatcher is None:
            return
        self._dispatcher_signals, self._dispatcher_unsubscribe = connect_dispatcher_signals(dispatcher)
        self._dispatcher_signals.tick.connect(self._on_tick)

    @QtCore.Slot(dict)
    def _on_tick(self, payload: dict) -> None:
        samples = payload.get("samples")
        status = payload.get("status", {})
        if status.get("sample_rate", 0.0) > 0:
            self._sample_rate = float(status["sample_rate"])
        if samples is None or samples.size == 0:
            return

        self._data_buffer.append(samples[0].astype(np.float32))
        max_samples = int(2.0 * self._sample_rate)
        while sum(len(d) for d in self._data_buffer) > max_samples:
            self._data_buffer.pop(0)

        data = np.concatenate(self._data_buffer)
        times = np.arange(data.size, dtype=np.float32) / max(self._sample_rate, 1.0)
        self.curve.setData(times, data)

    def _on_threshold_changed(self, value: float) -> None:
        trigger_cfg = TriggerConfig(
            channel_index=0,       # Replace with selected channel id
            threshold=float(value),
            hysteresis=0.0,
            pretrigger_frac=0.01,
            window_sec=1.0,
            mode="repeated",       # "stream", "single", "repeated"
        )
        self.runtime.configure_acquisition(trigger_cfg=trigger_cfg)

    def closeEvent(self, event):
        if self._dispatcher_unsubscribe is not None:
            self._dispatcher_unsubscribe()
            self._dispatcher_unsubscribe = None
        super().closeEvent(event)
```

#### 2. Register via Auto-Discovery

No manual `MainWindow` edits are needed when you use plugin tabs.

Place your file in `gui/tabs/`, subclass `BaseTab`, and set `TAB_TITLE`. Startup code in `gui/main_window.py` already handles discovery:

```python
self._tab_manager = TabPluginManager(self.runtime)
plugin_tabs = self._tab_manager.discover_and_instantiate()
for tab in plugin_tabs:
    title = getattr(tab, "TAB_TITLE", "Plugin Tab")
    self._analysis_dock.add_plugin_tab(tab, title)
```

#### 3. Connect to Data Flow

To receive real-time data in your tab, subscribe to dispatcher tick payloads. The payload includes:
- `samples`: `np.ndarray` of shape `(channels, samples)`
- `times`: `np.ndarray` time axis for the window
- `channel_ids` / `channel_names`
- `status`: includes `sample_rate` and `window_sec`

```python
from gui.dispatcher_adapter import connect_dispatcher_signals

if self.runtime.dispatcher is not None:
    signals, unsubscribe = connect_dispatcher_signals(self.runtime.dispatcher)
    signals.tick.connect(self._on_tick)
```

For trigger updates, use canonical modes: `"stream"`, `"single"`, and `"repeated"` (`"continuous"` is accepted only as a legacy alias and normalized internally).

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

import threading
import time
from typing import Callable, Optional, Sequence

import numpy as np
from .base_device import BaseDevice, DeviceInfo, ChannelInfo, Capabilities, ActualConfig

class MyDAQWithDIO(BaseDevice):
    """DAQ device with digital input/output capabilities.
    
    Extends the standard BaseDevice pattern to include:
    - Digital output for stimulus control
    - Digital input for external triggers
    """
    
    @classmethod
    def device_class_name(cls) -> str:
        return "My DAQ"

    @classmethod
    def list_available_devices(cls) -> list[DeviceInfo]:
        return [DeviceInfo(id="dev0", name="My DAQ USB", vendor="ACME")]

    def get_capabilities(self, device_id: str) -> Capabilities:
        return Capabilities(max_channels_in=4, sample_rates=[10_000, 20_000], dtype="float32")

    def list_available_channels(self, device_id: str) -> list[ChannelInfo]:
        return [ChannelInfo(id=i, name=f"AI{i}", units="V", range=(-10.0, 10.0)) for i in range(4)]

    def __init__(self, queue_maxsize: int = 64) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        self._worker: Optional[threading.Thread] = None
        self._dio_state: int = 0
        self._trigger_callback: Optional[Callable[[float], None]] = None
    
    # --- Standard DAQ hooks ---
    
    def _open_impl(self, device_id: str) -> None:
        # Open device handle
        self._handle = my_sdk.open(device_id)
    
    def _configure_impl(
        self,
        sample_rate: int,
        channels: Sequence[int],
        chunk_size: int,
        **opts,
    ) -> ActualConfig:
        # Configure analog input
        my_sdk.configure_ai(self._handle, sample_rate, channels, chunk_size)
        # Configure digital lines
        my_sdk.configure_dio(self._handle, lines=[0, 1, 2, 3])
        configured = [c for c in self._available_channels if c.id in channels]
        return ActualConfig(
            sample_rate=int(sample_rate),
            channels=configured,
            chunk_size=int(chunk_size),
            latency_s=None,
            dtype="float32",
        )
    
    def _start_impl(self) -> None:
        self._worker = threading.Thread(target=self._producer_loop, daemon=True)
        self._worker.start()
    
    def _stop_impl(self) -> None:
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
    
    def _close_impl(self) -> None:
        my_sdk.close(self._handle)

    def _producer_loop(self) -> None:
        while not self.stop_event.is_set():
            # SDK returns (frames, channels) float32
            block = my_sdk.read_ai(self._handle, self.config.chunk_size).astype(np.float32)
            self.emit_array(block, device_time=None)
    
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
   pip install -r requirements.txt
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
| `SpikeHoundRuntime` | `core/runtime.py` | Runtime orchestration and configuration API |
| `Dispatcher` | `core/dispatcher.py` | Data routing |
| `SignalConditioner` | `core/conditioning.py` | Filter processing |
| `AnalysisWorker` | `analysis/analysis_worker.py` | Spike detection |
| `TriggerController` | `gui/trigger_controller.py` | Trigger state and capture logic |
| `AnalysisDock` | `gui/analysis_dock.py` | Workspace tabs (Scope, Settings, plugins) |
| `TabPluginManager` | `gui/tab_plugin_manager.py` | Plugin tab discovery (`gui/tabs/`) |
| `BaseDevice` | `daq/base_device.py` | Hardware interface |

| Data Type | Module | Purpose |
|-----------|--------|---------|
| `Chunk` | `shared/models.py` | Raw data packets |
| `DetectionEvent` | `shared/models.py` | Detected spikes |
| `FilterSettings` | `core/conditioning.py` | Filter configuration |
| `TriggerConfig` | `shared/models.py` | Trigger settings |

---

## 💡 Tips for Forking as a Platform

### Keep Your Customizations Separate

```
spikehound/
├── gui/
│   ├── [core files]
│   └── tabs/               # Auto-discovered BaseTab plugins
│       ├── __init__.py
│       ├── burst_lesson_tab.py
│       └── muscle_fatigue_tab.py
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
