# GUI Module

The Qt-based graphical user interface for SpikeHound. Built with **PySide6** and **PyQtGraph**.

* Modular widget architecture
* Signal-based communication pattern
* Immutable dataclass configuration
* Clear separation: Widgets (view) ↔ Managers (logic) ↔ Controllers (state)

> This README covers widget patterns for adding new UI features.

---

## Quick Start

### Adding a new control to an existing widget

```python
# In an existing widget's _build_ui() method:
self.my_button = QtWidgets.QPushButton("My Action")
layout.addWidget(self.my_button)

# In _connect_signals():
self.my_button.clicked.connect(self._on_my_action)

# Handler:
def _on_my_action(self) -> None:
    # emit signal or call controller
    self.mySignal.emit(value)
```

### Adding a new settings panel

```python
from PySide6 import QtCore, QtWidgets

class MyControlWidget(QtWidgets.QWidget):
    # 1. Define signals for external communication
    configChanged = QtCore.Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self) -> None:
        """Build the widget layout."""
        layout = QtWidgets.QVBoxLayout(self)
        self.my_spin = QtWidgets.QDoubleSpinBox()
        layout.addWidget(self.my_spin)
    
    def _connect_signals(self) -> None:
        """Wire internal widgets to handler methods."""
        self.my_spin.valueChanged.connect(self._on_config_changed)
    
    def _on_config_changed(self) -> None:
        """Collect state and emit configuration."""
        self.configChanged.emit(self.my_spin.value())
```

---

## Architecture

### Widget Hierarchy

```
MainWindow (QMainWindow)
├── Central placeholder widget             # Layout anchor
├── Scope + control panels                 # Oscilloscope + controls
└── AnalysisDock (QDockWidget)
    └── Internal QTabWidget
        ├── Scope tab                      # Primary scope view
        ├── Settings tab                   # App settings & health
        ├── Analysis tabs                  # Opened per channel
        └── Plugin tabs                    # Loaded by TabPluginManager
```

### Managers (Business Logic)

| Manager | Responsibility | Used By |
|---------|---------------|---------|
| `ChannelManager` | Channel state (add/remove, config) | MainWindow |
| `PlotManager` | Trace rendering, curve management | MainWindow |
| `DeviceManager` | Qt adapter for DeviceRegistry | MainWindow |
| `AudioListenManager` | Audio monitoring control | MainWindow |
| `ScopeConfigManager` | Save/load scope configurations | MainWindow |
| `TriggerController` | Trigger detection state machine | MainWindow |

### Controllers (Pure State)

| Controller | Responsibility | Location |
|-----------|---------------|----------|
| `TriggerController` | Trigger config, history, capture | `gui/trigger_controller.py` |
| `PipelineController` | DAQ pipeline lifecycle | `core/controller.py` |

---

## Data Types

### ChannelConfig (gui/types.py)

Per-channel display and filter configuration:

```python
@dataclass
class ChannelConfig:
    color: QtGui.QColor              # Trace color
    display_enabled: bool = True     # Show on scope
    vertical_span_v: float = 1.0     # Y-axis scale (±1V = 2V span)
    screen_offset: float = 0.5       # Y position (0=bottom, 1=top)
    notch_enabled: bool = False      # 60Hz filter
    notch_freq_hz: float = 60.0
    highpass_enabled: bool = False   # High-pass filter
    highpass_hz: float = 10.0
    lowpass_enabled: bool = False    # Low-pass filter
    lowpass_hz: float = 1000.0
    listen_enabled: bool = False     # Audio monitoring
    analyze_enabled: bool = False    # Spike analysis
    channel_name: str = ""
```

### TriggerConfig (shared/models.py)

Trigger parameters (immutable, shared with core):

```python
@dataclass(frozen=True)
class TriggerConfig:
    channel_index: int
    threshold: float
    hysteresis: float
    pretrigger_frac: float
    window_sec: float
    mode: str  # "stream", "single", "repeated"
```

---

## Widget Patterns

### Standard Widget Structure

Every widget follows this pattern:

```python
class MyWidget(QtWidgets.QWidget):
    # 1. Signals at class level (for external communication)
    valueChanged = QtCore.Signal(float)
    actionRequested = QtCore.Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 2. Instance variables (internal state)
        self._current_value = 0.0
        
        # 3. Build UI and connect signals
        self._setup_ui()       # or _build_ui()
        self._connect_signals()
    
    def _setup_ui(self) -> None:
        """Create and arrange widgets."""
        layout = QtWidgets.QVBoxLayout(self)
        # ... create widgets ...
    
    def _connect_signals(self) -> None:
        """Connect widget signals to handlers."""
        self.some_button.clicked.connect(self._on_button_clicked)
    
    # 4. Event handlers: _on_* prefix
    def _on_button_clicked(self) -> None:
        """Handle button click."""
        self.actionRequested.emit()
    
    # 5. Public methods for external control
    def set_value(self, value: float) -> None:
        """Set value programmatically (blocks signals to avoid loops)."""
        self.spin_box.blockSignals(True)
        self.spin_box.setValue(value)
        self.spin_box.blockSignals(False)
```

### Avoiding Signal Loops

When setting widget values programmatically, block signals:

```python
def set_threshold(self, value: float) -> None:
    """Set threshold without triggering valueChanged."""
    self.threshold_spin.blockSignals(True)
    self.threshold_spin.setValue(value)
    self.threshold_spin.blockSignals(False)
    
    # If you need to propagate the change afterward:
    self._on_config_changed()  # Manually call handler
```

### Widget ↔ Controller Pattern

```python
class MyControlWidget(QtWidgets.QWidget):
    configChanged = QtCore.Signal(object)  # Emits MyConfig dataclass
    
    def __init__(self, controller: MyController, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._setup_ui()
        self._connect_signals()
    
    def _connect_signals(self) -> None:
        # UI → Controller
        self.my_button.clicked.connect(self._on_config_changed)
        
        # Controller → UI (if controller has signals)
        self._controller.stateChanged.connect(self._on_controller_changed)
    
    def _on_config_changed(self) -> None:
        """Gather UI state and update controller."""
        config = MyConfig(value=self.spin.value())
        self._controller.configure(config)
        self.configChanged.emit(config)  # Notify MainWindow
```

---

## Naming Conventions

| Pattern | Convention | Examples |
|---------|------------|----------|
| **Signals** | `camelCase` | `configChanged`, `valueChanged`, `actionRequested` |
| **Private methods** | `_underscore_prefix` | `_setup_ui`, `_connect_signals` |
| **Event handlers** | `_on_*` | `_on_button_clicked`, `_on_config_changed` |
| **Update methods** | `_update_*` | `_update_display`, `_update_axis_label` |
| **Build methods** | `_build_ui` or `_setup_ui` | (either is acceptable) |
| **Setters** | `set_*` | `set_value`, `set_enabled` |
| **Getters** | `get_*` or property | `get_selected_index`, `@property` |

---

## Adding a New Widget

### Step 1: Create the widget file

```python
# gui/my_new_widget.py
"""MyNewWidget - Description of what it does.

Provides controls for [feature], including:
- Item 1
- Item 2
"""
from __future__ import annotations

from typing import Optional
from PySide6 import QtCore, QtWidgets

class MyNewWidget(QtWidgets.QWidget):
    """Short description."""
    
    # Signals
    configChanged = QtCore.Signal(object)
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self) -> None:
        """Build the UI layout."""
        self.group_box = QtWidgets.QGroupBox("My Feature")
        layout = QtWidgets.QVBoxLayout(self.group_box)
        
        # Add widgets...
        self.my_spin = QtWidgets.QDoubleSpinBox()
        layout.addWidget(self.my_spin)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.group_box)
    
    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.my_spin.valueChanged.connect(self._on_config_changed)
    
    def _on_config_changed(self) -> None:
        """Handle configuration changes."""
        self.configChanged.emit(self.my_spin.value())
```

### Step 2: Add to MainWindow

```python
# In gui/main_window.py

# 1. Import
from .my_new_widget import MyNewWidget

# 2. Create in _init_ui() or appropriate location
self.my_widget = MyNewWidget()
self.right_panel_layout.addWidget(self.my_widget)

# 3. Connect signals
self.my_widget.configChanged.connect(self._on_my_config_changed)

# 4. Add handler
def _on_my_config_changed(self, value: float) -> None:
    """Handle my widget config changes."""
    # Forward to controller/runtime as needed
    self.runtime.configure_something(value)
```

---

## Adding a New Tab

### Step 1: Create the tab widget

```python
# gui/my_new_tab.py
from __future__ import annotations

from typing import Optional
from PySide6 import QtCore, QtWidgets

class MyNewTab(QtWidgets.QWidget):
    """Tab for [feature]."""
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        
        # Add your tab content
        self.label = QtWidgets.QLabel("My New Tab Content")
        layout.addWidget(self.label)
        
        layout.addStretch()  # Push content to top
    
    def refresh(self) -> None:
        """Called when tab becomes visible (optional)."""
        pass
```

### Step 2: Register in MainWindow

```python
# Save your file under gui/tabs/, subclassing BaseTab:
from gui.tab_plugin_manager import BaseTab

class MyNewTab(BaseTab):
    TAB_TITLE = "My Tab"
    ...

# TabPluginManager auto-discovers gui/tabs/*.py and
# AnalysisDock adds discovered tabs at startup.
```

---

## PyQtGraph Integration

### Adding a new plot

```python
import pyqtgraph as pg

# Create plot widget
self.plot = pg.PlotWidget()
self.plot.setBackground('w')
self.plot.showGrid(x=True, y=True, alpha=0.3)

# Create data curve
self.curve = self.plot.plot([], [], pen=pg.mkPen('b', width=2))

# Update data
def update_plot(self, x_data, y_data):
    self.curve.setData(x_data, y_data)
```

### Adding interactive lines

```python
# Horizontal line (e.g., threshold)
self.h_line = pg.InfiniteLine(
    pos=0.0,
    angle=0,
    movable=True,
    pen=pg.mkPen('r', width=2)
)
self.plot.addItem(self.h_line)
self.h_line.sigPositionChanged.connect(self._on_line_moved)

# Vertical line (e.g., trigger marker)
self.v_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False)
self.plot.addItem(self.v_line)
```

---

## PyQtGraph Performance Optimization

For real-time high-sample-rate visualization (100kHz+), these optimizations provide significant speedup:

### OpenGL Acceleration

OpenGL hardware rendering provides 5-20× speedup over software QPainter. Enable in `main.py` before any widgets are created:

```python
import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True, enableExperimental=True, antialias=True)
```

Requires `PyOpenGL` package. With OpenGL enabled, anti-aliasing has minimal performance cost.

### setData() Optimizations

Always use these parameters for high-frequency updates:

```python
# skipFiniteCheck: avoid scanning 100k+ points for NaN/Inf each frame
# connect='all': skip connection analysis, we know data is contiguous
self.curve.setData(x_data, y_data, skipFiniteCheck=True, connect='all')
```

### Pen Width

Thick pens (>2px) force slower rendering paths. Use `width=1` or `width=2` for maximum performance:

```python
# Fast
self.curve = self.plot.plot(pen=pg.mkPen('b', width=1))

# Slower (but more visible)
self.curve = self.plot.plot(pen=pg.mkPen('b', width=4))
```

### Downsampling Strategy

For 100kHz data with 1-second window (100k points), always downsample before display:

```python
# TraceRenderer.py uses automatic downsampling via setDownsampling()
self._curve.setDownsampling(ds=True, auto=True, method="peak")

# For manual downsampling when auto isn't available:
if samples.size > 4000:
    y_down, x_down = self._resample_peak(samples, times, target=2000)
```

Target ~2000 points (2× typical screen width) for smooth lines without wasted computation.

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `main_window.py` | 2066 | Central orchestrator, signal routing |
| `analysis_tab.py` | 2558 | Spike analysis and metrics display |
| `plot_manager.py` | 575 | Trace rendering, curve management |
| `device_control_widget.py` | 468 | Device selection, channel management |
| `channel_manager.py` | 453 | Channel state management |
| `trigger_controller.py` | 439 | Trigger detection state machine |
| `settings_tab.py` | 428 | App settings, health metrics |
| `scope_config_manager.py` | 416 | Save/load scope configurations |
| `channel_controls_widget.py` | 354 | Per-channel settings panels |
| `trigger_control_widget.py` | 272 | Trigger UI controls |
| `scope_widget.py` | 164 | Oscilloscope display wrapper |
| `types.py` | 26 | ChannelConfig dataclass |

---

## Checklist for New Widgets

- [ ] Create widget class with `_setup_ui()` and `_connect_signals()`
- [ ] Define signals for external communication at class level
- [ ] Use `_on_*` prefix for event handlers
- [ ] Block signals when setting values programmatically
- [ ] Add type hints to all methods
- [ ] Add docstring to class and complex methods
- [ ] Import and instantiate in MainWindow
- [ ] Connect widget signals to MainWindow handlers
- [ ] Test with device connected and disconnected

---

## FAQ

**Q: Why do signals use `object` type instead of specific dataclass?**

PySide6 Signal doesn't directly support frozen dataclasses. Use `Signal(object)` and document the expected type in comments.

**Q: Where should business logic go?**

In Managers (e.g., `ChannelManager`, `PlotManager`), not in widgets. Widgets should only handle UI state and emit signals.

**Q: How do I access the pipeline/runtime?**

Via `MainWindow.runtime` or `MainWindow._controller`. Widgets should emit signals that MainWindow handles, rather than accessing these directly.

**Q: How do I update the UI from a background thread?**

Use `QtCore.QMetaObject.invokeMethod()` or emit a signal. Never modify Qt widgets directly from non-GUI threads.
