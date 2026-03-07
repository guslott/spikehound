# SpikeHoundRuntime

`SpikeHoundRuntime` is the headless orchestration layer for SpikeHound. It owns DAQ attachment, the dispatcher and fan-out queues, analysis workers, audio monitoring, and health metrics so UIs (or CLI tools) can remain thin.

## Responsibilities
- Hold the active DAQ source and pipeline controller, attach/detach devices, and start/stop acquisition.
- Own the dispatcher fan-out queues (`visualization_queue`, `audio_queue`, `logging_queue`) plus dispatcher metrics and ring-buffer status.
- Manage analysis workers per channel/sample rate and expose their output queues.
- Persist and apply app-level settings (e.g., listen output device) via `AppSettingsStore`.
- Track health snapshot data (dispatcher stats, queue utilization, chunk/plot rates, sample rate).

## Public API (current)
- `connect_device(device_key, sample_rate, chunk_size=...)`: connect via the runtime-owned device manager.
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

# Configure the underlying pipeline explicitly
controller = runtime.controller
assert controller is not None
controller.switch_source(...)
controller.set_active_channels([0])
runtime.start_acquisition()

# Subscribe to analysis for a channel
queue, worker = runtime.open_analysis_stream("Channel 0", 20_000)
batch = queue.get(timeout=1)

print(runtime.health_snapshot())
runtime.stop_acquisition()
runtime.shutdown()
```

This makes it practical to build CLI recorders, headless demos, or notebook-based analyses that reuse the same pipeline without the GUI.
