from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type

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
from .models import EndOfStream, TriggerConfig


def _registry():
    from daq import registry as reg  # local import to avoid circular dependencies

    return reg


class DeviceManager(QtCore.QObject):
    """Tracks available DAQ drivers and manages a single active connection."""

    devicesChanged = QtCore.Signal(list)
    deviceConnected = QtCore.Signal(str)
    deviceDisconnected = QtCore.Signal()
    availableChannelsChanged = QtCore.Signal(list)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._descriptors: Dict[str, DeviceDescriptor] = {}
        self._active_key: Optional[str] = None
        self._driver: Optional[BaseSource] = None
        self._channels: List[ChannelInfo] = []
        self.refresh_devices()

    def refresh_devices(self) -> None:
        reg = _registry()
        reg.scan_devices(force=True)
        self._descriptors = {d.key: d for d in reg.list_devices()}
        payload = [
            {
                "key": d.key,
                "name": d.name,
                "module": d.module,
                "capabilities": d.capabilities,
            }
            for d in self._descriptors.values()
        ]
        self.devicesChanged.emit(payload)

    def get_device_list(self) -> List[DeviceDescriptor]:
        return list(self._descriptors.values())

    def connect_device(self, device_key: str, sample_rate: float, *, chunk_size: int = 1024, **driver_kwargs) -> BaseSource:
        self.disconnect_device()

        descriptor = self._descriptors.get(device_key)
        if descriptor is None:
            raise KeyError(f"Unknown device key: {device_key!r}")

        reg = _registry()
        driver = reg.create_device(device_key, **driver_kwargs)
        available = descriptor.cls.list_available_devices()
        if not available:
            raise RuntimeError(f"No hardware targets available for {descriptor.name}")

        target = available[0]
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
        self.deviceConnected.emit(device_key)
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
        analysis_queue_size: int = 256,
        audio_queue_size: int = 256,
        logging_queue_size: int = 512,
        dispatcher_poll_timeout: float = 0.05,
    ) -> None:
        self._filter_settings = filter_settings or FilterSettings()
        self.visualization_queue: "queue.Queue" = queue.Queue(maxsize=visualization_queue_size)
        self.analysis_queue: "queue.Queue" = queue.Queue(maxsize=analysis_queue_size)
        self.audio_queue: "queue.Queue" = queue.Queue(maxsize=audio_queue_size)
        self.logging_queue: "queue.Queue" = queue.Queue(maxsize=logging_queue_size)

        self._dispatcher_timeout = dispatcher_poll_timeout
        self._dispatcher: Optional[Dispatcher] = None
        self._source: Optional[BaseSource] = None
        self._actual_config: Optional[ActualConfig] = None
        self._device_id: Optional[str] = None
        self._configure_kwargs: Dict[str, Any] = {}
        self._running = False
        self._lock = threading.RLock()
        self._window_sec: float = 0.2
        self._streaming: bool = False
        self._active_channel_ids: List[int] = []
        self._channel_infos: List[ChannelInfo] = []

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
                analysis_queue=self.analysis_queue,
                audio_queue=self.audio_queue,
                logging_queue=self.logging_queue,
                filter_settings=self._filter_settings,
                poll_timeout=self._dispatcher_timeout,
            )
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
            self.stop(join=True)
            self._destroy_dispatcher()
            self._close_source()
            self._actual_config = None
            self._device_id = None
            self._configure_kwargs = {}

    def update_filter_settings(self, settings: FilterSettings) -> None:
        with self._lock:
            self._filter_settings = settings
            if self._dispatcher is not None:
                self._dispatcher.update_filter_settings(settings)

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
                analysis_queue=self.analysis_queue,
                audio_queue=self.audio_queue,
                logging_queue=self.logging_queue,
                filter_settings=self._filter_settings,
                poll_timeout=self._dispatcher_timeout,
            )
            channel_ids = [info.id for info in channels]
            channel_names = [info.name for info in channels]
            self._dispatcher.set_channel_layout(channel_ids, channel_names)
            self._dispatcher.set_window_duration(self._window_sec)
            self._dispatcher.clear_active_channels()

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

    def queue_depths(self) -> Dict[str, tuple[int, int]]:
        """Return (size, maxsize) for every queue the controller manages."""
        with self._lock:
            depths: Dict[str, tuple[int, int]] = {}
            if self._source is not None:
                depths["raw"] = (self._source.data_queue.qsize(), self._source.data_queue.maxsize)
            depths["visualization"] = (self.visualization_queue.qsize(), self.visualization_queue.maxsize)
            depths["analysis"] = (self.analysis_queue.qsize(), self.analysis_queue.maxsize)
            depths["audio"] = (self.audio_queue.qsize(), self.audio_queue.maxsize)
            depths["logging"] = (self.logging_queue.qsize(), self.logging_queue.maxsize)
            return depths

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
        for q in (self.visualization_queue, self.analysis_queue, self.audio_queue, self.logging_queue):
            self._flush_queue(q)

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
