from __future__ import annotations

import queue
from types import SimpleNamespace

from core.runtime import SpikeHoundRuntime
from shared.models import ChannelInfo


class _FakePipeline:
    def __init__(self) -> None:
        self.dispatcher = None
        self.visualization_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.logging_queue = queue.Queue()
        self.analysis_settings_store = None
        self.event_buffer = None
        self.attached: tuple[object, float, list[ChannelInfo]] | None = None
        self.sample_rate: float | None = None

    def attach_audio_manager(self, audio_manager) -> None:
        self._audio_manager = audio_manager

    def attach_source(self, driver, sample_rate: float, channels) -> None:
        channel_list = list(channels)
        self.attached = (driver, float(sample_rate), channel_list)
        self.sample_rate = float(sample_rate)


class _FakeDeviceManager:
    def __init__(self, driver, available_channels: list[ChannelInfo]) -> None:
        self._driver = driver
        self._available_channels = list(available_channels)

    def connect_device(self, device_key: str, sample_rate: float, *, chunk_size: int = 1024, **driver_kwargs):
        return self._driver

    def get_available_channels(self) -> list[ChannelInfo]:
        return list(self._available_channels)


def test_runtime_connect_device_uses_actual_driver_config(monkeypatch) -> None:
    monkeypatch.setattr("core.audio_manager.AudioManager.start", lambda self: None)

    configured_channels = [ChannelInfo(id=11, name="Configured", units="V")]
    available_channels = [
        ChannelInfo(id=0, name="Available 0", units="V"),
        ChannelInfo(id=1, name="Available 1", units="V"),
    ]
    driver = SimpleNamespace(
        config=SimpleNamespace(sample_rate=8_000, channels=configured_channels),
    )
    pipeline = _FakePipeline()
    device_manager = _FakeDeviceManager(driver, available_channels)
    runtime = SpikeHoundRuntime(pipeline=pipeline, device_manager=device_manager)

    runtime.connect_device("fake-device", 44_100)

    assert pipeline.attached is not None
    attached_driver, attached_rate, attached_channels = pipeline.attached
    assert attached_driver is driver
    assert attached_rate == 8_000.0
    assert attached_channels == configured_channels
    assert runtime.sample_rate == 8_000.0
