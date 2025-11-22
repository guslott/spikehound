# spikehound 2.0
Scientific data acquisition for physiology and neuroscience

SpikeHound 2.0 is an open-source desktop application for live, multi-channel bioelectric data acquisition, visualization, and first-pass analysis. Built with Python, PySide6, pyqtgraph, and a pluggable DAQ layer, it gives students and researchers a "see and hear the spikes" experience without hiding the mechanics of electrophysiology.

## Core capabilities for multi-channel electrophysiology
- Unified DAQ abstraction with drivers for sound cards, Backyard Brains SpikerBox devices, and a simulated multi-unit nerve prep -- so classrooms can move between inexpensive kits and lab interfaces without changing workflows.
- Real-time oscilloscope traces across many channels with per-channel color, scaling, and visibility controls for comparing nerve, muscle, EEG/ECG, or extracellular signals.
- Interactive conditioning controls (notch, high-pass, low-pass) to teach noise management and highlight events in the right frequency band.
- Optional "listen" mode to route channels to audio output, reinforcing spike timing and burst structure by ear.
- A live analysis pipeline that performs threshold-based spike/event detection, keeps rolling buffers of events, and powers an analysis tab for overlays, measurement cursors, and feature exploration.
- Modular architecture (see `daq/` and `core/`) that supports new hardware drivers and reuse of the data pipeline in custom teaching or research setups.

## A tool that grows with learners
- Middle school and outreach: pair a Backyard Brains SpikerBox or the built-in simulated source with simple sensors (cockroach nerve, plant potentials, EMG) to project spikes in real time; turn on audio monitoring so students can hear firing patterns.
- High school physiology: run multiple channels to compare muscle groups or reflex pathways, apply notch/high-pass filters to clean mains hum, and use threshold detection to mark conduction delays or response latency.
- Undergraduate neuroscience labs: stream extracellular recordings or EEG/ECG through the unified DAQ layer, annotate events, and use the analysis tab to align and overlay spikes from several electrodes while adjusting thresholds per channel.
- Graduate courses and research labs: prototype multi-channel acquisition pipelines, swap DAQ backends without changing the UI, and extend the analysis code for rapid spike-detection experiments or driver development for new amplifiers.

## Why it fits neuroscience education
SpikeHound keeps the signal chain visible: students see raw voltage traces, the impact of each filter, and the timing of detected events. The same interface works with low-cost teaching hardware and higher-channel lab rigs, lowering the barrier to authentic electrophysiology practice from middle school through advanced research training.

## Headless/runtime usage
The `core/` package exposes `SpikeHoundRuntime`, a GUI-agnostic orchestrator that owns DAQ attachment, dispatcher/queues, analysis workers, and health metrics. See `core/runtime_readme.md` for how to embed the runtime in a CLI recorder or alternate frontend.
