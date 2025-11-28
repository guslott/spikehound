# SpikeHound 2.0
**Scientific Data Acquisition for Physiology and Neuroscience**

SpikeHound 2.0 is a modern, open-source desktop application designed for real-time bioelectric data acquisition, visualization, and analysis. A complete rewrite of the legacy **g-PRIME** software, SpikeHound 2.0 leverages Python and hardware acceleration to turn standard laptops into research-grade oscilloscopes for neurophysiology education and research.

## 🧪 Project Mission
SpikeHound democratizes access to neuroscience by removing financial and technical barriers. It provides a free, cross-platform "Lab on a Chip" solution that replaces expensive proprietary hardware and software with:
* **Hardware Agnosticism:** Works with standard sound cards, low-cost hobbyist kits (e.g., Backyard Brains), and professional DAQ boards.
* **Cross-Platform Support:** Runs natively on Windows, macOS, and Linux.
* **Zero Cost:** Open-source (MIT/BSD-style license) with no MATLAB license required.

## 🚀 Key Features

### 1. Real-Time Oscilloscope
* **Multi-Channel Visualization:** Stream data from multiple sources simultaneously with independent scaling, offset, and color controls.
* **Live Conditioning:** Apply software-based filters in real-time:
    * **Notch Filter:** Remove 50/60Hz mains hum.
    * **High-Pass:** Eliminate DC offset and drift (AC coupling).
    * **Low-Pass:** Reduce high-frequency noise.
* **Audio Monitoring:** "Hear the neuron"—route any channel to your system speakers for real-time audio feedback of spike events.

### 2. Advanced Analysis
* **Threshold Detection:** Configure primary and secondary voltage thresholds to detect and capture spike events live.
* **Live Metrics:** Visualize event statistics in real-time, including:
    * Energy Density
    * Peak Frequency
    * Inter-Spike Interval (ISI) & Rate
* **Spike-Triggered Averaging (STA):** Correlate events across channels to visualize synaptic potentials or conduction delays in real-time.

### 3. Flexible Hardware Support
SpikeHound uses a plugin-based architecture to support various input sources:
* **Sound Card:** Uses the system's default audio input.
* **Backyard Brains:** Direct serial support for SpikerBox devices.
* **Simulation:** Built-in neural simulator (Poisson spike trains, noise, and mains hum) for testing and education without hardware.
* *(Planned)* **NI-DAQmx:** Support for National Instruments research boards.

---

## 📦 Installation & Usage

### Requirements
* **Python 3.12+**

### Dependencies
SpikeHound relies on the scientific Python stack and Qt for its GUI:
```bash
pip install numpy scipy PySide6 pyqtgraph sounddevice pyserial