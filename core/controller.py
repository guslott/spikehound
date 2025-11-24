from __future__ import annotations

import queue
import threading
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type

import numpy as np
from PySide6 import QtCore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from daq.base_source import ActualConfig, BaseSource, ChannelInfo
    from daq.registry import DeviceDescriptor
else:  # pragma: no cover - runtime fallback
    ActualConfig = Any
    BaseSource = Any
    ChannelInfo = Any
    DeviceDescriptor = Any

from .conditioning import FilterSettings
from .dispatcher import Dispatcher
from .audio_manager import AudioManager
from shared.models import Chunk, EndOfStream, TriggerConfig
from shared.ring_buffer import SharedRingBuffer
from analysis.settings import AnalysisSettingsStore
from shared.app_settings import AppSettings, AppSettingsStore
from shared.event_buffer import AnalysisEvents, EventRingBuffer
from shared.types import Event as SharedEvent


def _registry():
    from daq import registry as reg  # local import to avoid circular dependencies

    return reg


logger = logging.getLogger(__name__)


class DeviceManager(QtCore.QObject):
    """Tracks available DAQ drivers and manages a single active connection."""

    devicesChanged = QtCore.Signal(list)
    deviceConnected = QtCore.Signal(str)
    deviceDisconnected = QtCore.Signal()
    availableChannelsChanged = QtCore.Signal(list)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._descriptors: Dict[str, DeviceDescriptor] = {}
        self._device_entries: Dict[str, Dict[str, object]] = {}
        self._active_key: Optional[str] = None
        self._driver: Optional[BaseSource] = None
        self._channels: List[ChannelInfo] = []
        self._list_all_audio_devices = False
        self.refresh_devices()

    def set_list_all_audio_devices(self, enabled: bool, *, refresh: bool = False) -> None:
        self._list_all_audio_devices = bool(enabled)
        if refresh:
            self.refresh_devices()

    def refresh_devices(self) -> None:
        reg = _registry()
        reg.scan_devices(force=True)
        descriptors = reg.list_devices()
        try:
            from daq.soundcard_source import SoundCardSource
            SoundCardSource.set_list_all_devices(self._list_all_audio_devices)
        except Exception:
            pass
        self._descriptors = {d.key: d for d in descriptors}
        self._device_entries = {}
        payload: List[Dict[str, object]] = []
        for descriptor in descriptors:
            try:
                available = descriptor.cls.list_available_devices()
            except Exception:
                continue

            if not available:
                continue

            for device in sorted(available, key=lambda info: info.name.lower()):
                entry_key = f"{descriptor.key}::{device.id}"
                self._device_entries[entry_key] = {
                    "descriptor_key": descriptor.key,
                    "device_id": device.id,
                }
                capabilities = descriptor.capabilities
                try:
                    drv = reg.create_device(descriptor.key)
                    capabilities = drv.get_capabilities(device.id)
                except Exception:
                    pass
                entry: Dict[str, object] = {
                    "key": entry_key,
                    "name": f"{descriptor.name} - {device.name}",
                    "module": descriptor.module,
                    "capabilities": capabilities,
                    "driver_key": descriptor.key,
                    "driver_name": descriptor.name,
                    "device_id": device.id,
                    "device_name": device.name,
                }
                vendor = getattr(device, "vendor", None)
                if vendor:
                    entry["device_vendor"] = vendor
                details = getattr(device, "details", None)
                if details:
                    entry["device_details"] = details
                payload.append(entry)

        def _priority(item: Dict[str, object]) -> tuple:
            driver = str(item.get("driver_name", "")).lower()
            module = str(item.get("module", "")).lower()
            name = str(item.get("device_name", item.get("name", ""))).lower()
            # Prioritize: 1) Sound card (default), 2) Simulated, 3) Other DAQ
            if "sound card" in driver or "soundcard" in module:
                rank = 0
            elif "simulated" in driver or "simulated" in module:
                rank = 1
            else:
                rank = 2
            return (rank, driver, name)

        payload.sort(key=_priority)
        self.devicesChanged.emit(payload)

    def get_device_list(self) -> List[DeviceDescriptor]:
        return list(self._descriptors.values())

    def connect_device(self, device_key: str, sample_rate: float, *, chunk_size: int = 1024, **driver_kwargs) -> BaseSource:
        self.disconnect_device()

        entry = self._device_entries.get(device_key)
        if entry is None:
            raise KeyError(f"Unknown device key: {device_key!r}")

        descriptor_key = entry.get("descriptor_key")
        device_id = entry.get("device_id")
        if descriptor_key is None or not isinstance(descriptor_key, str):
            raise KeyError(f"Invalid descriptor for key: {device_key!r}")

        descriptor = self._descriptors.get(descriptor_key)
        if descriptor is None:
            raise KeyError(f"Unknown descriptor key: {descriptor_key!r}")

        if device_id is None or device_id == "":
            error_message = entry.get("error") or "No hardware devices detected for this driver."
            raise RuntimeError(error_message)

        reg = _registry()
        driver = reg.create_device(descriptor_key, **driver_kwargs)
        available = descriptor.cls.list_available_devices()
        target = next((dev for dev in available if str(dev.id) == str(device_id)), None)
        if target is None:
            raise RuntimeError(f"Selected device is no longer available: {device_id!r}")

        driver.open(target.id)
        channels = driver.list_available_channels(target.id)

        configure_kwargs = {
            "sample_rate": int(sample_rate),
            "channels": [ch.id for ch in channels] if channels else None,
            "chunk_size": chunk_size,
        }
        configure_kwargs = {k: v for k, v in configure_kwargs.items() if v is not None}

        try:
            driver.configure(**configure_kwargs)
        except Exception:
            driver.close()
            raise

        self._driver = driver
        self._active_key = device_key
        self._channels = list(channels)
        self.availableChannelsChanged.emit(list(self._channels))
        # NOTE: deviceConnected signal is now emitted from runtime.connect_device()
        # after attach_source() completes to ensure dispatcher is ready
        return driver

    def disconnect_device(self) -> None:
        if self._driver is None:
            return
        try:
            if getattr(self._driver, "running", False):
                try:
                    self._driver.stop()
                except Exception:
                    pass
            try:
                self._driver.close()
            except Exception:
                pass
        finally:
            self._driver = None
            self._active_key = None
            self._channels = []
            self.availableChannelsChanged.emit([])
            self.deviceDisconnected.emit()

    def get_available_channels(self) -> List[ChannelInfo]:
        return list(self._channels)

    def active_key(self) -> Optional[str]:
        return self._active_key

    def current_driver(self) -> Optional[BaseSource]:
        return self._driver


class PipelineController:
    """Owns the acquisition pipeline lifecycle (source + dispatcher)."""

    def __init__(
        self,
        *,
        filter_settings: Optional[FilterSettings] = None,
        visualization_queue_size: int = 256,
        audio_queue_size: int = 256,
        logging_queue_size: int = 512,
        dispatcher_poll_timeout: float = 0.05,
    ) -> None:
        self._filter_settings = filter_settings or FilterSettings()
        self.visualization_queue: "queue.Queue" = queue.Queue(maxsize=visualization_queue_size)
        self.audio_queue: "queue.Queue" = queue.Queue(maxsize=audio_queue_size)
        self.logging_queue: "queue.Queue" = queue.Queue(maxsize=logging_queue_size)
        self._analysis_settings = AnalysisSettingsStore()
        self._app_settings_store = AppSettingsStore()
        self._event_buffer = EventRingBuffer(capacity=1000)
        self._analysis_events = AnalysisEvents(self._event_buffer)

        self._dispatcher_timeout = dispatcher_poll_timeout
        self._dispatcher: Optional[Dispatcher] = None
        self._source: Optional[BaseSource] = None
        self._actual_config: Optional[ActualConfig] = None
        self._device_id: Optional[str] = None
        self._configure_kwargs: Dict[str, Any] = {}
        self._running = False
        self._lock = threading.RLock()
        self._window_sec: float = 1.0
        self._streaming: bool = False
        self._active_channel_ids: List[int] = []
        self._channel_infos: List[ChannelInfo] = []
        self._list_all_audio_devices: bool = False
        
        # Audio management (will be initialized by Runtime)
        self._audio_manager: Optional[AudioManager] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def source(self) -> Optional[BaseSource]:
        return self._source

    @property
    def dispatcher(self) -> Optional[Dispatcher]:
        return self._dispatcher

    @property
    def filter_settings(self) -> FilterSettings:
        return self._filter_settings

    @property
    def analysis_settings_store(self) -> AnalysisSettingsStore:
        return self._analysis_settings

    @property
    def event_buffer(self) -> EventRingBuffer:
        return self._event_buffer

    @property
    def analysis_events(self) -> AnalysisEvents:
        return self._analysis_events

    @property
    def app_settings_store(self) -> AppSettingsStore:
        return self._app_settings_store

    @property
    def app_settings(self) -> AppSettings:
        return self._app_settings_store.get()

    def update_app_settings(self, **kwargs) -> None:
        self._app_settings_store.update_app_settings(**kwargs)

    # Audio control API (delegates to AudioManager)
    
    def set_audio_monitoring(self, channel_id: Optional[int]) -> None:
        """Enable or disable audio monitoring for a channel.
        
        Args:
            channel_id: Channel ID to monitor, or None to stop
        """
        if self._audio_manager is not None:
            self._audio_manager.set_listen_channel(channel_id)
    
    def set_audio_device(self, device_id: Optional[int]) -> None:
        """Set audio output device.
        
        Args:
            device_id: Device ID for output, or None for default
        """
        if self._audio_manager is not None:
            self._audio_manager.set_output_device(device_id)
    
    def set_audio_gain(self, gain: float) -> None:
        """Set audio output gain.
        
        Args:
            gain: Gain value (0.0 to 1.0)
        """
        if self._audio_manager is not None:
            self._audio_manager.set_gain(gain)

    def set_list_all_audio_devices(self, enabled: bool) -> None:
        self._list_all_audio_devices = bool(enabled)

    @property
    def sample_rate(self) -> Optional[float]:
        return None if self._actual_config is None else float(self._actual_config.sample_rate)

    def active_channels(self) -> Sequence[ChannelInfo]:
        return [] if self._actual_config is None else list(self._actual_config.channels)

    def switch_source(
        self,
        source_cls: Type[BaseSource],
        *,
        device_id: Optional[str] = None,
        configure_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ActualConfig:
        """Swap in a new source; keeps queues and filter settings intact."""

        configure_kwargs = dict(configure_kwargs or {})

        with self._lock:
            self.stop(join=True)
            self._destroy_dispatcher()
            self._close_source()

            source = source_cls()
            devices = source_cls.list_available_devices()
            if not devices:
                raise RuntimeError(f"No devices available for {source_cls.__name__}")

            selected = None
            if device_id is None:
                selected = devices[0]
            else:
                for dev in devices:
                    if str(dev.id) == str(device_id):
                        selected = dev
                        break
                if selected is None:
                    raise ValueError(f"Device id {device_id!r} not found for {source_cls.__name__}")

            source.open(selected.id)
            available_channels = source.list_available_channels(selected.id)
            if "channels" not in configure_kwargs or not configure_kwargs["channels"]:
                # Default to “all” channels, preserving hardware order. Callers
                # can still override this by providing their own `channels`
                # sequence in `configure_kwargs`.
                configure_kwargs["channels"] = [ch.id for ch in available_channels]

            try:
                actual = source.configure(**configure_kwargs)
            except Exception:
                source.close()
                raise

            self._source = source
            self._actual_config = actual
            self._device_id = selected.id
            self._configure_kwargs = configure_kwargs
            self._channel_infos = list(actual.channels)
            self._streaming = False
            self._dispatcher = Dispatcher(
                raw_queue=source.data_queue,
                visualization_queue=self.visualization_queue,
                audio_queue=self.audio_queue,
                logging_queue=self.logging_queue,
                filter_settings=self._filter_settings,
                poll_timeout=self._dispatcher_timeout,
            )
            try:
                self._dispatcher.set_source_buffer(source.get_buffer(), sample_rate=actual.sample_rate)
            except Exception:
                pass
            self._running = False
            self._reset_output_queues()
            return actual

    def start(self) -> None:
        with self._lock:
            if self._source is None or self._dispatcher is None:
                raise RuntimeError("No source configured. Call switch_source() first.")
            if self._running:
                return

            self._reset_output_queues()
            self._dispatcher.start()
            self._source.start()
            self._running = True

    def stop(self, *, join: bool = True) -> None:
        """Stop the pipeline; currently always blocks until threads exit."""
        with self._lock:
            if self._source is None:
                return

            if self._source.running:
                try:
                    self._source.stop()
                except Exception:
                    pass

            if self._dispatcher is not None:
                self._push_end_of_stream()
                self._dispatcher.stop()

            self._running = False

            if join:
                self._flush_queue(self._source.data_queue)

    def shutdown(self) -> None:
        """Stop everything and close the active source."""
        with self._lock:
            # Stop audio manager first
            if self._audio_manager is not None:
                self._audio_manager.stop()
            
            self.stop(join=True)
            self._destroy_dispatcher()
            self._close_source()
            self._actual_config = None
            self._device_id = None
            self._configure_kwargs = {}

    def update_filter_settings(self, settings: FilterSettings) -> None:
        with self._lock:
            if settings == self._filter_settings:
                return
            self._filter_settings = settings
            if self._dispatcher is not None:
                self._dispatcher.update_filter_settings(settings)

    def update_analysis_settings(self, *, event_window_ms: Optional[float] = None) -> None:
        updates: Dict[str, float] = {}
        if event_window_ms is not None:
            updates["event_window_ms"] = float(event_window_ms)
        if not updates:
            return
        with self._lock:
            self._analysis_settings.update(**updates)

    def pull_analysis_events(self, last_event_id: Optional[int] = None):
        return self._analysis_events.pull_events(last_event_id)

    # Trigger configuration placeholder ---------------------------------

    def update_trigger_config(self, config: Dict[str, Any]) -> None:
        """Receive trigger configuration updates from the GUI and forward them."""
        with self._lock:
            if self._dispatcher is None:
                return

            channel_index = int(config.get("channel_index", -1))
            trigger_conf = TriggerConfig(
                channel_index=channel_index,
                threshold=float(config.get("threshold", 0.0)),
                hysteresis=float(config.get("hysteresis", 0.0)),
                pretrigger_frac=float(config.get("pretrigger_frac", 0.0)),
                window_sec=float(config.get("window_sec", 0.0)),
                mode=str(config.get("mode", "continuous")),
            )

            self._window_sec = trigger_conf.window_sec or self._window_sec
            sample_rate = self.sample_rate or 0.0
            self._dispatcher.set_trigger_config(trigger_conf, sample_rate)
            self._dispatcher.set_window_duration(trigger_conf.window_sec)

    def attach_source(self, driver: BaseSource, sample_rate: float, channels: Sequence[ChannelInfo]) -> None:
        with self._lock:
            self._stop_streaming_locked()
            self._destroy_dispatcher()
            self._close_source()

            self._source = driver
            self._actual_config = getattr(driver, "config", None)
            self._device_id = None
            self._configure_kwargs = {}
            self._channel_infos = list(channels)
            self._active_channel_ids = []
            self._streaming = False

            self._dispatcher = Dispatcher(
                raw_queue=driver.data_queue,
                visualization_queue=self.visualization_queue,
                audio_queue=self.audio_queue,
                logging_queue=self.logging_queue,
                filter_settings=self._filter_settings,
                poll_timeout=self._dispatcher_timeout,
            )
            effective_sr = float(sample_rate) if sample_rate is not None else 0.0
            if effective_sr <= 0:
                try:
                    cfg = getattr(driver, "config", None)
                    effective_sr = float(getattr(cfg, "sample_rate", 0.0))
                except Exception:
                    effective_sr = 0.0
            if effective_sr <= 0:
                logger.error("Invalid sample rate for dispatcher wiring (driver=%s)", driver)
                raise ValueError("Sample rate must be positive to attach source")
            if getattr(driver, "ring_buffer", None) is None:
                try:
                    channel_ids_cfg = [info.id for info in channels] if channels else None
                    driver.configure(sample_rate=int(round(effective_sr)), channels=channel_ids_cfg)
                except Exception as exc:
                    logger.error("Failed to configure driver before wiring dispatcher: %s", exc)
                    raise
            try:
                self._dispatcher.set_source_buffer(driver.get_buffer(), sample_rate=effective_sr)
                logger.info("Attached source buffer to dispatcher. SR=%s", effective_sr)
            except Exception as exc:
                logger.error("Failed to link source buffer: %s", exc)
                raise
            channel_ids = [info.id for info in channels]
            channel_names = [info.name for info in channels]
            self._dispatcher.set_channel_layout(channel_ids, channel_names)
            self._dispatcher.set_window_duration(self._window_sec)
            self._dispatcher.clear_active_channels()
            logger.info("Source attached successfully (sr=%s, channels=%s)", effective_sr, channel_ids)

    def detach_device(self) -> None:
        with self._lock:
            self._stop_streaming_locked()
            self._destroy_dispatcher()
            self._close_source()
            self._channel_infos = []
            self._active_channel_ids = []

    def set_active_channels(self, channel_ids: Sequence[int]) -> None:
        with self._lock:
            self._active_channel_ids = list(channel_ids)
            if self._dispatcher is not None:
                info_map = {info.id: info for info in self._channel_infos}
                ordered_infos = [info_map[cid] for cid in self._active_channel_ids if cid in info_map]
                names = [info.name for info in ordered_infos]
                self._dispatcher.set_channel_layout(self._active_channel_ids, names)
                self._dispatcher.set_active_channels(self._active_channel_ids)
            if self._source is not None:
                try:
                    self._source.set_active_channels(self._active_channel_ids)
                except Exception:
                    pass
                # Rewire dispatcher to the source buffer in case channel changes resized it.
                if self._dispatcher is not None and hasattr(self._source, 'ring_buffer') and self._source.ring_buffer is not None:
                    try:
                        sr = self._actual_config.sample_rate if self._actual_config is not None else None
                        self._dispatcher.set_source_buffer(self._source.get_buffer(), sample_rate=sr)
                    except Exception as exc:
                        logger.debug("Could not re-link source buffer on channel change: %s", exc)

            if self._active_channel_ids and self._dispatcher is not None and self._source is not None:
                if not self._streaming:
                    self._start_streaming_locked()
            else:
                if self._streaming:
                    self._stop_streaming_locked()

    def clear_active_channels(self) -> None:
        self.set_active_channels([])

    def start_recording(self, path: str, rollover: bool) -> None:
        """Placeholder for future recording start logic."""
        _ = (path, rollover)

    def stop_recording(self) -> None:
        """Placeholder for future recording stop logic."""
        pass

    def update_window_span(self, window_sec: float) -> None:
        with self._lock:
            if self._dispatcher is None:
                return
            self._dispatcher.set_window_duration(window_sec)

    def dispatcher_stats(self) -> Dict[str, object]:
        with self._lock:
            if self._dispatcher is None:
                return {}
            return self._dispatcher.snapshot()

    def dispatcher_signals(self):
        with self._lock:
            return None if self._dispatcher is None else self._dispatcher.signals

    def viz_buffer(self) -> Optional[SharedRingBuffer]:
        with self._lock:
            dispatcher = self._dispatcher
        if dispatcher is None:
            return None
        return dispatcher.viz_buffer

    def queue_depths(self) -> Dict[str, dict]:
        """Return per-queue health metrics for visualization, analysis, audio, and logging."""
        with self._lock:
            depths: Dict[str, dict] = {}

        def _queue_status(q: "queue.Queue") -> dict:
            size = q.qsize()
            maxsize = q.maxsize
            utilization = (size / maxsize) if maxsize > 0 else 0.0
            return {
                "size": size,
                    "max": maxsize,
                    "utilization": utilization,
                }

        depths["visualization"] = _queue_status(self.visualization_queue)
        if self._dispatcher is not None:
            depths["viz_buffer"] = self._dispatcher.buffer_status()

        depths["audio"] = _queue_status(self.audio_queue)
        depths["logging"] = _queue_status(self.logging_queue)
        return depths

    def collect_trigger_window(
        self,
        event: SharedEvent,
        target_channel_id: int,
        window_ms: float,
    ) -> tuple[np.ndarray, int, int]:
        """Return samples centered on an event plus missing-prefix/suffix counts."""
        dispatcher = self._dispatcher
        if dispatcher is None:
            return np.empty(0, dtype=np.float32), 0, 0
        if event is None:
            return np.empty(0, dtype=np.float32), 0, 0
        sr = float(getattr(event, "sampleRateHz", 0.0) or 0.0)
        if sr <= 0:
            return np.empty(0, dtype=np.float32), 0, 0
        try:
            channel_id = int(target_channel_id)
        except (TypeError, ValueError):
            return np.empty(0, dtype=np.float32), 0, 0
        window_samples = max(1, int(round(window_ms * sr / 1000.0)))
        if window_samples % 2 == 0:
            window_samples += 1
        center_index = int(getattr(event, "crossingIndex", -1))
        if center_index < 0:
            return np.empty(0, dtype=np.float32), 0, window_samples
        half = window_samples // 2
        start_index = max(0, center_index - half)
        data, missing_prefix, missing_suffix = dispatcher.collect_window(
            start_index,
            window_samples,
            channel_id,
            return_info=True,
        )
        return data, missing_prefix, missing_suffix

    def register_analysis_queue(self, data_queue: "queue.Queue[Chunk | EndOfStream]") -> Optional[int]:
        with self._lock:
            if self._dispatcher is None:
                return None
            return self._dispatcher.register_analysis_queue(data_queue)

    def unregister_analysis_queue(self, token: Optional[int]) -> None:
        if token is None:
            return
        with self._lock:
            if self._dispatcher is None:
                return
            self._dispatcher.unregister_analysis_queue(token)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _push_end_of_stream(self) -> None:
        if self._source is None:
            return
        raw_queue = self._source.data_queue
        try:
            raw_queue.put_nowait(EndOfStream)
        except queue.Full:
            try:
                raw_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                raw_queue.put_nowait(EndOfStream)
            except queue.Full:
                pass

    def _flush_queue(self, q: "queue.Queue") -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return

    def _reset_output_queues(self) -> None:
        """Clear downstream queues so the next start() begins fresh."""
        for q in (self.visualization_queue, self.audio_queue, self.logging_queue):
            self._flush_queue(q)
        self._event_buffer.clear()

    def _destroy_dispatcher(self) -> None:
        if self._dispatcher is None:
            return
        try:
            self._dispatcher.stop()
            self._dispatcher.emit_empty_tick()
        except Exception:
            pass
        self._dispatcher = None

    def _close_source(self) -> None:
        if self._source is None:
            return
        try:
            if self._source.running:
                self._source.stop()
            self._source.close()
        except Exception:
            pass
        self._source = None
        self._streaming = False
        self._active_channel_ids = []

    def _start_streaming_locked(self) -> None:
        if self._dispatcher is None or self._source is None:
            return
        try:
            info_map = {info.id: info for info in self._channel_infos}
            ordered_infos = [info_map[cid] for cid in self._active_channel_ids if cid in info_map]
            names = [info.name for info in ordered_infos]
            self._dispatcher.set_channel_layout(self._active_channel_ids, names)
            self._dispatcher.set_active_channels(self._active_channel_ids)
            self._dispatcher.start()
        except Exception:
            pass
        try:
            self._source.start()
        except Exception:
            pass
        self._streaming = True

    def _stop_streaming_locked(self) -> None:
        if not self._streaming:
            return
        if self._source is not None:
            try:
                if self._source.running:
                    self._source.stop()
            except Exception:
                pass
        if self._dispatcher is not None:
            try:
                self._dispatcher.clear_active_channels()
                self._dispatcher.reset_buffers()
                self._dispatcher.emit_empty_tick()
                self._dispatcher.stop()
            except Exception:
                pass
        self._streaming = False
