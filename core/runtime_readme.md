# SpikeHoundRuntime

`SpikeHoundRuntime` is the headless orchestration layer for SpikeHound. It owns DAQ attachment, the dispatcher and fan-out queues, analysis workers, audio monitoring, and health metrics so UIs (or CLI tools) can remain thin.

## Responsibilities
- Hold the active DAQ source and pipeline controller, attach/detach devices, and start/stop acquisition.
- Own the dispatcher fan-out queues (`visualization_queue`, `audio_queue`, `logging_queue`) plus dispatcher metrics and ring-buffer status.
- Manage analysis workers per channel/sample rate and expose their output queues.
- Persist and apply app-level settings (e.g., listen output device) via `AppSettingsStore`.
- Track health snapshot data (dispatcher stats, queue utilization, chunk/plot rates, sample rate).

## Public API (current)
- `attach_source(driver, sample_rate, channels)`: attach a configured DAQ driver to the pipeline.
- `configure_acquisition(channels=None, filter_settings=None, trigger_cfg=None)`: push filter/trigger/channel updates into the pipeline.
- `start_acquisition()` / `stop_acquisition()`: start/stop streaming through the dispatcher.
- `shutdown()`: stop everything and release resources.
- `open_analysis_stream(channel_name, sample_rate) -> (queue, worker)`: start an `AnalysisWorker` for a channel and return its output queue.
- `set_listen_output_device(device_key)`: persist the audio output preference.
- `health_snapshot()`: return dispatcher stats, queue depths, and current rates.
- `update_metrics(chunk_rate=None, plot_refresh_hz=None, sample_rate=None)`: let UI push live rate info for health snapshots.

## Using the runtime headlessly
```python
from core.runtime import SpikeHoundRuntime
from core import PipelineController

runtime = SpikeHoundRuntime(pipeline=PipelineController())

# Attach an already-open driver (from daq.registry) and configure
driver = ...  # BaseDevice
channels = driver.list_available_channels(driver.config.device_id)
runtime.attach_source(driver, sample_rate=20_000, channels=channels)
runtime.configure_acquisition(channels=[ch.id for ch in channels])
runtime.start_acquisition()

# Subscribe to analysis for a channel
queue, worker = runtime.open_analysis_stream("Channel 0", 20_000)
batch = queue.get(timeout=1)

print(runtime.health_snapshot())
runtime.stop_acquisition()
runtime.shutdown()
```

This makes it practical to build CLI recorders, headless demos, or notebook-based analyses that reuse the same pipeline without the GUI.
