from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type

if TYPE_CHECKING:  # pragma: no cover - typing only
    from daq.base_source import ActualConfig, BaseSource, ChannelInfo
else:  # pragma: no cover - runtime fallback
    ActualConfig = Any
    BaseSource = Any
    ChannelInfo = Any

from .conditioning import FilterSettings
from .dispatcher import Dispatcher
from .models import EndOfStream


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

    def dispatcher_stats(self) -> Dict[str, object]:
        with self._lock:
            if self._dispatcher is None:
                return {}
            return self._dispatcher.snapshot()

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
        except Exception:
            pass
        # Downstream queues remain in place; they are owned by the controller.
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
        self._source = None
