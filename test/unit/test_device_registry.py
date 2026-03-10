from __future__ import annotations

from types import SimpleNamespace

from core.device_registry import DeviceRegistry
from daq.base_device import Capabilities, ChannelInfo, DeviceInfo


class _VirtualFileDriver:
    opened_device_id: str | None = None
    configured_sample_rate: int | None = None

    @classmethod
    def list_available_devices(cls):
        return [DeviceInfo(id="file", name="File Playback")]

    def open(self, device_id: str) -> None:
        type(self).opened_device_id = device_id

    def list_available_channels(self, device_id: str):
        return [ChannelInfo(id=0, name="Channel 1", units="V")]

    def get_capabilities(self, device_id: str):
        return Capabilities(max_channels_in=1, sample_rates=[20_000], dtype="float32")

    def configure(self, *, sample_rate: int, channels, chunk_size: int):
        type(self).configured_sample_rate = sample_rate
        return SimpleNamespace(sample_rate=sample_rate, channels=channels, chunk_size=chunk_size)

    def close(self) -> None:
        pass


def test_connect_device_uses_device_id_override_for_virtual_sources(monkeypatch):
    descriptor_key = "daq.file_source.FileSource"
    descriptor = SimpleNamespace(
        key=descriptor_key,
        name="File Source",
        cls=_VirtualFileDriver,
        module="daq.file_source",
        capabilities={},
    )

    fake_registry = SimpleNamespace(
        scan_devices=lambda force=False: None,
        list_devices=lambda: [descriptor],
        create_device=lambda key, **kwargs: _VirtualFileDriver(),
    )

    monkeypatch.setattr("core.device_registry._registry", lambda: fake_registry)

    registry = DeviceRegistry()
    registry.refresh_devices()

    device_key = f"{descriptor_key}::file"
    driver = registry.connect_device(
        device_key,
        sample_rate=0.0,
        device_id_override="/tmp/example.wav",
    )

    assert isinstance(driver, _VirtualFileDriver)
    assert _VirtualFileDriver.opened_device_id == "/tmp/example.wav"
    assert _VirtualFileDriver.configured_sample_rate == 20_000
    assert registry.active_key() == device_key
