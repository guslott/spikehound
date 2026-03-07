from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import daq.backyard_brains as byb


def _encode_frames(values: list[int]) -> bytes:
    buf = bytearray()
    for value in values:
        buf.append(((value >> 7) & 0x7F) | 0x80)
        buf.append(value & 0x7F)
    return bytes(buf)


@dataclass
class _FakePort:
    device: str
    name: str
    vid: int
    pid: int
    description: str = "Backyard Brains"
    manufacturer: str = "Backyard Brains"
    product: str = "Backyard Brains"
    interface: str | None = None
    serial_number: str | None = None
    hwid: str = "USB"


class _FakeSerialHandle:
    def __init__(self, script: dict[tuple[int, str] | str, bytes], baudrate: int) -> None:
        self._script = script
        self._buf = bytearray()
        self.baudrate = int(baudrate)
        self.is_open = True
        self.write_log: list[str] = []

    @property
    def in_waiting(self) -> int:
        return len(self._buf)

    def read(self, n: int) -> bytes:
        chunk = bytes(self._buf[:n])
        del self._buf[:n]
        return chunk

    def write(self, payload: bytes) -> int:
        cmd = payload.decode("ascii")
        self.write_log.append(cmd)
        response = self._script.get((self.baudrate, cmd))
        if response is None:
            response = self._script.get(cmd)
        if response:
            self._buf.extend(response)
        return len(payload)

    def reset_input_buffer(self) -> None:
        self._buf.clear()

    def close(self) -> None:
        self.is_open = False


class _FakeSerialModule:
    def __init__(self, ports: list[_FakePort], script: dict[tuple[int, str] | str, bytes]) -> None:
        self._ports = ports
        self._script = script
        self.instances: list[_FakeSerialHandle] = []
        self.tools = SimpleNamespace(
            list_ports=SimpleNamespace(comports=lambda: list(self._ports))
        )

    def Serial(self, *, port: str, baudrate: int, timeout: float, write_timeout: float):  # noqa: N802
        handle = _FakeSerialHandle(self._script, baudrate)
        self.instances.append(handle)
        return handle


def _make_hid_packet(payload: bytes) -> bytes:
    packet = bytearray(64)
    packet[0] = 0x01
    packet[1] = min(len(payload), 62)
    packet[2 : 2 + min(len(payload), 62)] = payload[:62]
    return bytes(packet)


class _FakeHIDHandle:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self._responses = responses
        self.read_queue: list[bytes] = []
        self.writes: list[bytes] = []
        self.path: bytes | None = None
        self.nonblocking: int | None = None

    def open_path(self, path: bytes) -> None:
        self.path = bytes(path)

    def set_nonblocking(self, value: int) -> None:
        self.nonblocking = int(value)

    def send_feature_report(self, packet: bytes) -> int:
        packet = bytes(packet)
        self.writes.append(packet)
        cmd = packet[2:].split(b"\x00", 1)[0].decode("ascii")
        response = self._responses.get(cmd)
        if response:
            self.read_queue.append(_make_hid_packet(response))
        return len(packet)

    def read(self, size: int, timeout_ms: int = 0) -> list[int]:
        if self.read_queue:
            return list(self.read_queue.pop(0))
        return []

    def close(self) -> None:
        return


class _FakeHIDModule:
    def __init__(self, entries: list[dict[str, object]], responses: dict[str, bytes]) -> None:
        self._entries = entries
        self._responses = responses
        self.handle = _FakeHIDHandle(responses)

    def enumerate(self, vendor_id: int, product_id: int = 0) -> list[dict[str, object]]:
        return list(self._entries)

    def device(self):
        return self.handle


def test_serial_probe_resolves_profile_and_truthful_capabilities(monkeypatch):
    port = _FakePort(
        device="/dev/tty.neuron",
        name="tty.neuron",
        vid=0x2E73,
        pid=0x0007,
        description="Neuron SpikerBox Pro",
        product="Neuron SpikerBox Pro",
    )
    script = {
        (222222, "?:;"): b"FWV:1.02;HWT:NEURONSB;HWV:2.00;",
        (222222, "b:;"): b"HWT:NSBPCDC;",
        (222222, "board:;"): b"BRD:2;",
    }
    fake_serial = _FakeSerialModule([port], script)
    monkeypatch.setattr(byb, "serial", fake_serial)
    monkeypatch.setattr(byb, "hid", None)

    devices = byb.BackyardBrainsSource.list_available_devices()
    assert devices[0].details["transport"] == "serial"
    assert devices[0].details["hid_support_available"] is False

    src = byb.BackyardBrainsSource()
    src.open(port.device)
    caps = src.get_capabilities(port.device)

    assert src._profile is not None
    assert src._profile.key == "neuron_pro"
    assert src._fallback_identification is False
    assert caps.sample_rates == [5000, 10000]
    assert src.stats()["board_type"] == 2
    src.close()


def test_shared_ftdi_id_is_disambiguated_by_probe_and_baud(monkeypatch):
    port = _FakePort(
        device="/dev/tty.ftdi",
        name="tty.ftdi",
        vid=0x0403,
        pid=0x6015,
        description="FT231X USB UART",
        product="FT231X USB UART",
    )
    script = {
        (500000, "b:;"): b"HWT:HHIBOX;",
        (500000, "?:;"): b"FWV:3.10;",
    }
    fake_serial = _FakeSerialModule([port], script)
    monkeypatch.setattr(byb, "serial", fake_serial)
    monkeypatch.setattr(byb, "hid", None)

    src = byb.BackyardBrainsSource()
    src.open(port.device)

    assert src._profile is not None
    assert src._profile.key == "hhi_new"
    assert src.stats()["baudrate"] == 500000
    src.close()


def test_hid_probe_uses_feature_report_wrapping_and_resolves_profile(monkeypatch):
    hid_entry = {
        "path": b"hid-neuron",
        "vendor_id": 0x2E73,
        "product_id": 0x0002,
        "product_string": "Neuron SpikerBox Pro",
        "manufacturer_string": "Backyard Brains",
    }
    responses = {
        "h:;": b"",
        "?:;": b"FWV:1.03;HWT:NEURONSB;HWV:2.10;",
        "b:;": b"HWT:NEURONSB;",
        "board:;": b"BRD:1;",
    }
    fake_hid = _FakeHIDModule([hid_entry], responses)
    monkeypatch.setattr(byb, "hid", fake_hid)
    monkeypatch.setattr(byb, "serial", None)

    devices = byb.BackyardBrainsSource.list_available_devices()
    assert devices and devices[0].details["transport"] == "hid"

    src = byb.BackyardBrainsSource()
    src.open(devices[0].id)

    assert src._profile is not None
    assert src._profile.key == "neuron_pro"
    assert any(packet[0] == 0x3F and packet[1] == 0x3E for packet in fake_hid.handle.writes)
    assert any(packet[2:5] == b"?:;" for packet in fake_hid.handle.writes)
    src.close()


def test_configure_rejects_invalid_rate_channel_combo(monkeypatch):
    port = _FakePort(
        device="/dev/tty.neuron",
        name="tty.neuron",
        vid=0x2E73,
        pid=0x0007,
        description="Neuron SpikerBox Pro",
        product="Neuron SpikerBox Pro",
    )
    script = {
        (222222, "?:;"): b"FWV:1.02;HWT:NEURONSB;HWV:2.00;",
        (222222, "b:;"): b"HWT:NSBPCDC;",
    }
    fake_serial = _FakeSerialModule([port], script)
    monkeypatch.setattr(byb, "serial", fake_serial)
    monkeypatch.setattr(byb, "hid", None)

    src = byb.BackyardBrainsSource()
    src.open(port.device)

    with pytest.raises(ValueError, match="10000 Hz"):
        src.configure(sample_rate=5000, channels=[0, 1], chunk_size=32)

    cfg = src.configure(sample_rate=5000, channels=[0, 2], chunk_size=32)
    assert cfg.sample_rate == 5000
    assert src.stats()["stream_channels"] == 3
    assert any("c:3;" in handle.write_log for handle in fake_serial.instances)
    src.close()


def test_decoder_infers_ambiguous_stream_width():
    decoder = byb._BYBDecoder(bits=10, candidate_widths=(2, 4))
    raw = bytearray()
    for _ in range(16):
        raw.extend(_encode_frames([512, 512, 512, 512]))

    messages, decoded = decoder.feed(bytes(raw))

    assert messages == []
    assert decoded is not None
    assert decoder.stream_width == 4
    assert decoded.shape == (16, 4)


def test_decoder_resyncs_after_midstream_corruption():
    decoder = byb._BYBDecoder(bits=10, candidate_widths=(2,))
    prefix = _encode_frames([512, 512] * 6)
    suffix = _encode_frames([512, 512] * 6)
    corrupted = bytearray(prefix + suffix)
    corrupted[len(prefix)] = 0x00  # break expected high-byte marker at the boundary

    messages, first = decoder.feed(bytes(corrupted))
    assert messages == []
    assert first is not None
    assert first.shape[1] == 2

    second_messages, second = decoder.feed(b"")
    third_messages, third = decoder.feed(b"")

    assert second_messages == []
    assert second is None
    assert third_messages == []
    assert third is not None
    assert third.shape[1] == 2


def test_decoder_preserves_wrapped_messages_split_across_reads():
    decoder = byb._BYBDecoder(bits=10, candidate_widths=(1,))
    message = byb._MSG_START + b"EVNT:4;" + byb._MSG_END

    first_messages, first_decoded = decoder.feed(message[:7])
    second_messages, second_decoded = decoder.feed(message[7:])

    assert first_messages == []
    assert first_decoded is None
    assert second_messages == ["EVNT:4;"]
    assert second_decoded is None


def test_hid_enumeration_is_optional(monkeypatch):
    port = _FakePort(
        device="/dev/tty.plant",
        name="tty.plant",
        vid=0x2341,
        pid=0x8036,
        description="Plant SpikerBox",
        product="Plant SpikerBox",
    )
    fake_serial = _FakeSerialModule([port], {})
    monkeypatch.setattr(byb, "serial", fake_serial)
    monkeypatch.setattr(byb, "hid", None)

    devices = byb.BackyardBrainsSource.list_available_devices()
    assert len(devices) == 1
    assert devices[0].details["hid_support_available"] is False
