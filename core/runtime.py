from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional, Sequence, TYPE_CHECKING, Tuple

from daq.base_device import BaseDevice

from .conditioning import FilterSettings, SignalConditioner
from .device_registry import DeviceRegistry
from shared.app_settings import AppSettingsStore
from shared.models import TriggerConfig
from shared.types import AnalysisEvent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .controller import PipelineController
    from analysis.analysis_worker import AnalysisWorker
else:  # pragma: no cover - runtime fallback
    PipelineController = Any
    AnalysisWorker = Any


class SpikeHoundRuntime:
    """
    Skeleton runtime orchestrator for DAQ, conditioning, analysis, audio, and metrics.

    This will evolve into the headless controller that replaces direct MainWindow
    ownership of pipeline pieces.
    
    Both device_registry and device_manager can be injected for GUI integration.
    When running headless, omit device_manager to avoid Qt dependencies.
    """

    def __init__(
        self,
        *,
        app_settings_store: Optional[AppSettingsStore] = None,
        logger: Optional[logging.Logger] = None,
        pipeline: Optional["PipelineController"] = None,
        device_registry: Optional["DeviceRegistry"] = None,
        device_manager: Optional[Any] = None,
    ) -> None:
        self.app_settings_store = app_settings_store
        self.logger = logger or logging.getLogger(__name__)
        self.daq_source: Optional[BaseDevice] = None
        self.conditioner = SignalConditioner()
        self._queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._pipeline: Optional["PipelineController"] = pipeline
        self._analysis_workers: dict[Tuple[str, float], "AnalysisWorker"] = {}
        # Core registry (pure Python) - use provided or create default
        self._device_registry = device_registry if device_registry is not None else DeviceRegistry()
        # Qt adapter is injected from GUI layer (or None for headless)
        self.device_manager = device_manager
        self.dispatcher = getattr(pipeline, "dispatcher", None) if pipeline is not None else None
        self.visualization_queue: Optional[queue.Queue] = getattr(pipeline, "visualization_queue", None)
        self.audio_queue: Optional[queue.Queue] = getattr(pipeline, "audio_queue", None)
        self.logging_queue: Optional[queue.Queue] = getattr(pipeline, "logging_queue", None)
        self.chunk_rate: float = 0.0
        self.plot_refresh_hz: float = 0.0
        self.sample_rate: float = 0.0
        self._acquisition_start_time: Optional[float] = None  # For uptime tracking
        
        # Initialize AudioManager and wire it to pipeline
        from .audio_manager import AudioManager
        audio_manager = AudioManager(runtime=self)
        if pipeline is not None:
            pipeline._audio_manager = audio_manager
            audio_manager.start()  # Start audio routing thread

    def set_pipeline(self, controller: Optional["PipelineController"]) -> None:
        """Update the underlying pipeline controller used for delegation."""
        self._pipeline = controller
        if controller is not None:
            self.dispatcher = controller.dispatcher
            self.visualization_queue = controller.visualization_queue
            self.audio_queue = controller.audio_queue
            self.logging_queue = controller.logging_queue
    
    @property
    def controller(self) -> Optional["PipelineController"]:
        """Get the underlying pipeline controller."""
        return self._pipeline



    def open_device(self, driver: BaseDevice, sample_rate: float, channels: Sequence[object]) -> None:
        """Open and prepare the requested DAQ backend/device."""
        self.attach_source(driver, sample_rate, channels)

    def connect_device(self, device_key: str, sample_rate: float, chunk_size: int = 1024) -> None:
        """Connect a device via the DeviceManager and wire it into the pipeline."""
        # OPTIMIZATION: Calculate chunk size for ~20ms latency if not explicitly overridden
        # default was 1024, which is >100ms at 10kHz.
        # We target 20ms (0.02s) with a minimum of 32 samples for safety.
        if chunk_size == 1024:  # Only override if it's the default
            target_latency = 0.02  # 20ms
            chunk_size = max(32, int(sample_rate * target_latency))
            # Align to 2/4 for cleanliness? Not strictly necessary but 128/256 are nice.
            # actually strict 20ms is better for time alignment.
            
        driver = self.device_manager.connect_device(device_key, sample_rate, chunk_size=chunk_size)
        channels = self.device_manager.get_available_channels()
        self.attach_source(driver, sample_rate, channels)
        # Emit deviceConnected AFTER dispatcher is created so GUI can bind to it
        self._device_registry.emit_device_connected(device_key)

    def configure_acquisition(
        self,
        *,
        sample_rate: Optional[int] = None,
        channels: Optional[list[int]] = None,
        chunk_size: Optional[int] = None,
        filter_settings: Optional[FilterSettings] = None,
        trigger_cfg: Optional[TriggerConfig] = None,
    ) -> None:
        """Configure the acquisition pipeline with updated settings.
        
        Updates filter settings, trigger configuration, and active channel list
        on the underlying pipeline controller. All parameters are optional;
        only provided values are applied.
        
        Args:
            sample_rate: Target sample rate in Hz (currently unused, set at device level).
            channels: List of channel IDs to enable. Empty list clears all channels.
            chunk_size: Samples per chunk (currently unused, set at device level).
            filter_settings: Signal conditioning parameters (high-pass, low-pass, notch).
            trigger_cfg: Trigger configuration (threshold, mode, pretrigger, window).
        """
        controller = self._pipeline
        if controller is None:
            return
        if filter_settings is not None:
            try:
                controller.update_filter_settings(filter_settings)
            except Exception as exc:
                self.logger.warning("Failed to update filter settings: %s", exc)
        if trigger_cfg is not None:
            try:
                controller.update_trigger_config(trigger_cfg)
            except Exception as exc:
                self.logger.warning("Failed to update trigger config: %s", exc)
        if channels is not None:
            if channels:
                try:
                    controller.set_active_channels(channels)
                except Exception as exc:
                    self.logger.warning("Failed to set active channels: %s", exc)
            else:
                try:
                    controller.clear_active_channels()
                except Exception as exc:
                    self.logger.warning("Failed to clear active channels: %s", exc)

    def start_acquisition(self) -> None:
        """Start streaming data from the DAQ source into the pipeline.
        
        Starts the dispatcher thread which pulls data from the source and
        fans it out to visualization, audio, logging, and analysis queues.
        Also begins uptime tracking for health metrics.
        """
        controller = self._pipeline
        if controller is None:
            return
        try:
            controller.start()
            self._acquisition_start_time = time.monotonic()  # Track uptime
        except Exception as exc:
            self.logger.warning("Failed to start acquisition: %s", exc)
            return

    def stop_acquisition(self) -> None:
        """Stop streaming and wait for pipeline threads to exit.
        
        Stops the dispatcher thread and blocks until it exits. Resets uptime
        tracking. Note: Does NOT disconnect the device, allowing acquisition
        to be resumed later without reconnecting.
        """
        controller = self._pipeline
        if controller is None:
            return
        self._acquisition_start_time = None  # Reset uptime
        try:
            controller.stop(join=True)
        except Exception as exc:
            self.logger.warning("Failed to stop acquisition: %s", exc)
        # FIX: Do not detach device here. MainWindow calls this when channels are empty,
        # but we want to keep the device open so we can resume later when channels are added.
        # try:
        #     controller.detach_device()
        # except Exception:
        #     pass
        # self.daq_source = None

    def attach_source(self, driver: BaseDevice, sample_rate: float, channels) -> None:
        """Attach a connected driver to the pipeline controller."""
        controller = self._pipeline
        if controller is None or driver is None:
            return
        self.daq_source = driver
        try:
            controller.attach_source(driver, float(sample_rate), channels)
        except Exception as exc:
            self.logger.warning("Failed to attach source: %s", exc)
            return
        self.dispatcher = getattr(controller, "dispatcher", None)
        self.visualization_queue = getattr(controller, "visualization_queue", None)
        self.audio_queue = getattr(controller, "audio_queue", None)
        self.logging_queue = getattr(controller, "logging_queue", None)

    def health_snapshot(self) -> dict[str, object]:
        """Return current queue depths, rates, source stats, and dispatcher health."""
        controller = self._pipeline
        stats = controller.dispatcher_stats() if controller is not None else {}
        queues: dict[str, dict] = {}

        def _queue_status(q: Optional[queue.Queue]) -> dict:
            if q is None:
                return {"size": 0, "max": 0, "utilization": 0.0}
            size = q.qsize()
            maxsize = q.maxsize
            util = (size / maxsize) if maxsize > 0 else 0.0
            return {"size": size, "max": maxsize, "utilization": util}

        queues["visualization"] = _queue_status(self.visualization_queue)
        queues["audio"] = _queue_status(self.audio_queue)
        queues["logging"] = _queue_status(self.logging_queue)

        dispatcher = self.dispatcher
        if dispatcher is not None:
            try:
                queues["viz_buffer"] = dispatcher.buffer_status()
            except Exception:
                queues["viz_buffer"] = {}

        # Source device stats
        source_stats: dict[str, object] = {}
        source = controller.source if controller is not None else None
        if source is not None:
            try:
                src_info = source.stats()
                source_stats = {
                    "xruns": src_info.get("xruns", 0),
                    "drops": src_info.get("drops", 0),
                    "queue_size": src_info.get("queue_size", 0),
                    "queue_max": src_info.get("queue_maxsize", 0),
                }
            except Exception:
                pass

        # Uptime calculation
        uptime: Optional[float] = None
        if self._acquisition_start_time is not None:
            uptime = time.monotonic() - self._acquisition_start_time

        return {
            "chunk_rate": float(self.chunk_rate),
            "plot_refresh_hz": float(self.plot_refresh_hz),
            "sample_rate": float(self.sample_rate),
            "uptime": uptime,
            "source": source_stats,
            "dispatcher": stats,
            "queues": queues,
        }

    def update_metrics(
        self,
        *,
        chunk_rate: Optional[float] = None,
        plot_refresh_hz: Optional[float] = None,
        sample_rate: Optional[float] = None,
    ) -> None:
        """Update runtime-facing metrics so health_snapshot can report them."""
        if chunk_rate is not None:
            self.chunk_rate = float(chunk_rate)
        if plot_refresh_hz is not None:
            self.plot_refresh_hz = float(plot_refresh_hz)
        if sample_rate is not None:
            self.sample_rate = float(sample_rate)

    def set_listen_output_device(self, device_key: Optional[str]) -> None:
        """Select audio output target for listen/mirror mode (persist only for now)."""
        if self.app_settings_store is not None:
            try:
                self.app_settings_store.update(listen_output_key=device_key)
            except Exception as exc:
                self.logger.warning("Failed to set listen output device: %s", exc)
                return

    def set_list_all_audio_devices(self, enabled: bool) -> None:
        """Propagate audio device listing preference to the device manager."""
        try:
            self.device_manager.set_list_all_audio_devices(enabled, refresh=True)
        except Exception as exc:
            self.logger.warning("Failed to set list all audio devices: %s", exc)

    def set_audio_monitoring(self, channel_id: Optional[int]) -> None:
        """Enable or disable audio monitoring (listen) for a channel."""
        controller = self._pipeline
        if controller is not None:
            controller.set_audio_monitoring(channel_id)

    def set_audio_gain(self, gain: float) -> None:
        """Set output volume gain (0.0 - 1.0)."""
        controller = self._pipeline
        if controller is not None:
            controller.set_audio_gain(gain)

    def set_audio_output_device(self, device_id: Optional[int]) -> None:
        """Set the active audio output device index."""
        controller = self._pipeline
        if controller is not None:
            controller.set_audio_device(device_id)

    def open_analysis_stream(self, channel_name: str, sample_rate: float) -> tuple[Optional[queue.Queue], Optional["AnalysisWorker"]]:
        """
        Create and start an AnalysisWorker for the given channel and return its output queue.
        """
        if self._pipeline is None:
            return None, None
        try:
            from analysis.analysis_worker import AnalysisWorker  # local import to avoid cycles
        except Exception:
            return None, None
        key = (str(channel_name), float(sample_rate))
        worker = self._analysis_workers.get(key)
        
        # Check if the cached worker is still alive - if the analysis tab was closed,
        # the worker was stopped and is no longer running. We need to create a new one.
        if worker is not None and not worker.is_alive():
            # Worker was stopped (e.g., tab was closed) - remove from cache
            del self._analysis_workers[key]
            worker = None
        
        if worker is None:
            try:
                worker = AnalysisWorker(self._pipeline, channel_name, sample_rate)
                worker.start()
                self._analysis_workers[key] = worker
            except Exception:
                return None, None
        return getattr(worker, "output_queue", None), worker

    def collect_trigger_window(
        self,
        event: AnalysisEvent,
        target_channel_id: int,
        window_ms: float,
    ) -> tuple[Any, int, int]:
        """Delegate trigger window collection to the pipeline controller."""
        import numpy as np
        controller = self._pipeline
        if controller is None:
            return np.empty(0, dtype=np.float32), 0, 0
        return controller.collect_trigger_window(
            event,
            target_channel_id=target_channel_id,
            window_ms=window_ms,
        )

