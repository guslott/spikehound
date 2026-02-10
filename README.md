# SpikeHound 2.0

**Real-Time Neurophysiology Software for Education and Research**

SpikeHound 2.0 is a free, open-source desktop application for real-time bioelectric data acquisition, visualization, and analysis. Designed to bridge the gap between education and research, it scales from high school biology demonstrations to graduate-level neurophysiology data collection.

---

## 🎯 Who Is This For?

| Audience | Use Case |
|----------|----------|
| **High School Biology** | Demonstrate action potentials with simplified, "plug-and-play" hardware |
| **Undergraduate Labs** | Replace expensive proprietary systems in neurophysiology courses |
| **Graduate Research** | Thesis-level data collection with precise timing and raw data export |
| **Research Labs** | Rapid prototyping of new experimental paradigms and custom hardware drivers |
| **Citizen Scientists** | Explore electrophysiology with low-cost open hardware |

### Educational Applications
- **BioNB 491** (Cornell): Principles of Neurophysiology lab exercises
- **AP Biology**: Neuroscience and action potential demonstrations
- **Science Fairs**: Student-driven neurophysiology projects
- **Outreach**: Public demonstrations at science museums and events

---

## 🧪 Project Mission

SpikeHound **democratizes neurophysiology** by removing financial and technical barriers:

| Barrier | SpikeHound Solution |
|---------|---------------------|
| Expensive hardware | Works with low-cost hardware or free sound cards |
| MATLAB licenses ($$$) | No licenses required—100% free |
| Windows-only software | Runs on Windows/macOS via installers; Linux is supported from source |
| Complex setup | Single Python script, simple installation |
| Proprietary formats | Standard open formats (WAV, CSV, NumPy) |
| Black-box software | Fully inspection-ready Python codebase |

---

## 🚀 Key Features

### Real-Time Oscilloscope
- **Multi-channel visualization** with independent color, scaling, and offset
- **Live signal conditioning**: Notch (50/60Hz), high-pass, and low-pass filters
- **Audio monitoring**: "Hear the neuron"—route any channel to speakers
- **Triggered capture**: Single-shot or repeated threshold triggering

### Spike Detection & Analysis
- **Threshold detection** with configurable polarity and refractory period
- **Live metrics**: Energy density, peak frequency, inter-spike interval
- **Spike-triggered averaging**: Correlate events across channels
- **Event capture**: Extract waveforms around detected spikes

### Research-Grade Performance
- **Lossless Acquisition**: Ring-buffer architecture allows separation of visualization and recording, ensuring no data is lost even if the UI lags.
- **Precise Timing**: Uses host monotonic clock for reliable timestamping relative to hardware events.
- **Raw Data Integrity**: Direct float32 streaming from hardware to disk (WAV/CSV) without hidden processing.
- **Universal Hardware Abstraction**: Treat a \$5 sound card and a \$5000 DAQ the same way.

### Hardware Support
| Device | Status | Notes |
|--------|--------|-------|
| **Sound Card** | ✅ Supported | Any USB/built-in audio input (up to 96kHz+) |
| **Neuron SpikerBox** | ✅ Supported | Supports Muscle/Neuron/Pro models (Backyard Brains) |
| **Simulation** | ✅ Supported | Neural simulator for testing analysis pipelines |
| **WAV Files** | ✅ Supported | Replay recorded data for offline analysis |
| **NI-DAQmx** | 🔄 Planned | National Instruments boards (via generic Python drivers) |
| **LabJack** | 🔄 Planned | T-series DAQ devices (via generic Python drivers) |

---

## 📦 Installation

### Option 1: Download Binary Installer (Recommended)

Pre-built installers are available for **Windows** and **macOS**—no programming experience required!

1. Visit the [latest release page](https://github.com/guslott/spikehound/releases/latest)
2. Scroll to the **Assets** section at the bottom
3. Download the appropriate installer:
   - **Windows:** `SpikeHound-Setup.exe`
   - **macOS:** `SpikeHound.dmg`
4. Run the installer and follow the prompts

---

### Option 2: Run from Source (Advanced, required on Linux)

For developers or those who want to modify the code:

#### Requirements
- **Python 3.12+**
- Source checkout and terminal access (Windows, macOS, Linux)

#### Quick Start (Linux source-level support)
```bash
# Clone the repository
git clone https://github.com/guslott/spikehound.git
cd spikehound

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

> **Dependencies:** numpy, scipy, PySide6, pyqtgraph, miniaudio, pyserial (see `requirements.txt` for version constraints)

#### Building a Standalone App
```bash
pip install pyinstaller
pyinstaller SpikeHound.spec
# Output: dist/SpikeHound.app (macOS) or dist/SpikeHound.exe (Windows)
```

---

## 🤖 AI-Aided Development for Non-Programmers

> **SpikeHound is designed to be extended by anyone**, even without programming experience. The codebase is structured specifically for AI coding assistants to understand and modify.

### How It Works

1. **You describe what you want** in plain English
2. **AI reads the documentation** and understands the patterns
3. **AI implements the feature** following established conventions
4. **You test and verify** using the provided checklists

### What You Can Add (Without Being a Programmer)

| Feature Type | Difficulty | Example Request |
|-------------|------------|-----------------|
| **New metric** | Easy | "Add a spike width measurement" |
| **New filter** | Easy | "Add a bandpass filter option" |
| **New device** | Medium | "Add support for LabJack T7" |
| **New tab** | Medium | "Add a spectrogram view" |
| **New analysis** | Medium | "Add burst detection" |

### Getting Started with AI Development

1. **Open an AI coding assistant** (Claude, Cursor, GitHub Copilot, etc.)
2. **Describe your feature request** clearly
3. **Point the AI to the relevant workflow**:
   - `/add-daq-driver` - For new hardware support
   - `/add-analysis-metric` - For new measurements
   - `/add-gui-tab` - For new interface tabs
   - `/add-filter-type` - For new signal filters
4. **Review the AI's implementation** and test

### Documentation for AI Assistants

The codebase includes extensive documentation specifically designed for AI:

| Document | Location | Purpose |
|----------|----------|---------|
| DAQ Driver Guide | `daq/daq_readme.md` | Adding hardware support |
| GUI Widget Guide | `gui/gui_readme.md` | Adding UI components |
| Analysis Guide | `analysis/analysis_readme.md` | Adding metrics/detection |
| Dispatcher Guide | `doc/dispatcher_readme.md` | Understanding data flow |

### Example: Adding a New Metric

**Your request to AI:**
> "I want to add a 'spike width at half maximum' metric that measures how wide each spike is in milliseconds."

**What happens:**
1. AI reads `analysis/analysis_readme.md`
2. AI adds a function to `analysis/metrics.py`
3. AI integrates it into the analysis worker
4. AI optionally adds display to the GUI
5. You test with real or simulated data

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Oscilloscope│  │  Analysis   │  │  Settings   │     │
│  │   (Scope)   │  │    Tab      │  │    Tab      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Core Pipeline                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Dispatcher │→ │ Conditioning│→ │  Analysis   │     │
│  │  (Router)   │  │  (Filters)  │  │  (Metrics)  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 Hardware Abstraction                     │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐ │
│  │ SoundCard │ │ SpikerBox │ │ Simulator │ │WAV File │ │
│  └───────────┘ └───────────┘ └───────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

Each layer has its own documentation and follows consistent patterns that AI can understand and replicate.

---

## 📁 Project Structure

```
spikehound/
├── main.py                 # Application entry point
├── README.md               # This file
│
├── gui/                    # User interface components
│   ├── gui_readme.md       # 📖 Widget development guide
│   ├── main_window.py      # Central orchestrator
│   ├── scope_widget.py     # Oscilloscope display
│   └── analysis_tab.py     # Spike analysis view
│
├── daq/                    # Hardware drivers
│   ├── daq_readme.md       # 📖 Driver development guide
│   ├── base_device.py      # Abstract device interface
│   ├── soundcard_source.py # Audio input driver
│   └── backyard_brains.py  # Serial hardware driver
│
├── analysis/               # Spike detection & metrics
│   ├── analysis_readme.md  # 📖 Metric development guide
│   ├── metrics.py          # Metric functions
│   └── analysis_worker.py  # Background processor
│
├── core/                   # Pipeline orchestration
│   ├── dispatcher.py       # Data routing
│   ├── conditioning.py     # Signal filters
│   └── controller.py       # Pipeline lifecycle
│
├── shared/                 # Common data types
│   ├── models.py           # DetectionEvent, Chunk, etc.
│   └── types.py            # AnalysisEvent
│
├── .agent/workflows/       # AI development workflows
│   ├── add-daq-driver.md
│   ├── add-analysis-metric.md
│   ├── add-gui-tab.md
│   └── add-filter-type.md
│
└── test/                   # Automated tests (200+)
```

---

## 🧑‍🔬 For Educators

### Classroom Setup
1. Install Python 3.12+ on lab computers
2. Clone SpikeHound and install dependencies
3. Connect acquisition hardware (or use simulation mode)
4. Launch with `python main.py`

### Classroom Safety Note
- SpikeHound is for education and research demonstrations, not medical diagnosis or treatment.
- Follow your institution's lab safety and consent policies for any human-participant activities.

### Lab Exercise Ideas
- **Action Potential Recording**: Record from earthworm giant fibers
- **EMG Analysis**: Measure muscle activity with surface electrodes
- **ECG Demonstration**: Record heartbeat with simple electrodes
- **Neural Conduction Velocity**: Measure spike propagation speed
- **Filter Effects**: Demonstrate signal conditioning in real-time

### No Hardware? No Problem!
Use the built-in **Simulated Source** to demonstrate concepts without any physical equipment. The simulator generates realistic:
- Poisson spike trains
- Action potential waveforms
- Background noise
- 60Hz mains hum

---

## 📚 History & Acknowledgments

**SpikeHound** (formerly **g-PRIME**) was created by **Dr. Gus K. Lott III** as a doctoral student at **Cornell University** in **Dr. Ronald R. Hoy's** laboratory. It became a cornerstone of the *BioNB 491: Principles of Neurophysiology* course.

**SpikeHound 1.0** was developed under funding from **HHMI** while **Gus Lott** worked as an instrumentation engineer at the **Janelia Farm Research Campus** and supported frontier neuroscience research.

**SpikeHound 2.0** is a complete rewrite, developed with support from **Manlius Pebble Hill School (MPHS)** and continued mentorship from Cornell.

### Contributors
- **Author:** Gus K. Lott III, PhD
- **Student Contributor:** Taylor Mangoba (MPHS)
- **Special Thanks:** Dr. Ronald Hoy, Audrey Yeager

---

## 📄 License

Copyright © 2025 Gus K. Lott III

This software is open-source under the 0BSD License. See `LICENSE` for details.

---

## 🔗 Links

- **Repository:** [github.com/guslott/spikehound](https://github.com/guslott/spikehound)
- **Issues:** [Report bugs or request features](https://github.com/guslott/spikehound/issues)
