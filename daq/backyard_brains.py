from __future__ import annotations

import base64
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    import hid  # `hidapi` package exposes the module as `hid`
except ImportError:
    hid = None

from daq.base_device import ActualConfig, BaseDevice, Capabilities, ChannelInfo, DeviceInfo

_LOGGER = logging.getLogger(__name__)

_BYB_VID = 0x2E73
_FTDI_VID = 0x0403
_ARDUINO_VID = 0x2341

_BOOTLOADER_PIDS = {
    (_BYB_VID, 0x0005),  # Human bootloader
    (_BYB_VID, 0x000A),  # Neuron bootloader
    (_BYB_VID, 0x000B),  # Spike Station bootloader
}

_MSG_START = bytes([0xFF, 0xFF, 0x01, 0x01, 0x80, 0xFF])
_MSG_END = bytes([0xFF, 0xFF, 0x01, 0x01, 0x81, 0xFF])

_DEFAULT_PROBE_COMMANDS = ("?:;", "b:;")

_PROBE_RX_TIMEOUT_S = 0.12
_SYNC_TIMEOUT_S_DEFAULT = 0.5
_MAX_READ_SIZE = 4096
_HID_PACKET_SIZE = 64
_HID_PAYLOAD_SIZE = 62


@dataclass(frozen=True)
class _BYBProfile:
    key: str
    label: str
    hardware_aliases: tuple[str, ...]
    transports: tuple[str, ...]
    bits: int
    max_channels: int
    candidate_stream_channels: tuple[int, ...]
    sample_rate_map: Mapping[int, int]
    baud_rates: tuple[int, ...]
    requires_start: bool = False
    supports_channel_cfg: bool = False
    supports_board_query: bool = False
    supports_human_controls: bool = False
    supports_spike_station_queries: bool = False
    infer_stream_width: bool = False
    description: str = ""

    @property
    def sample_rates(self) -> list[int]:
        return sorted({int(v) for v in self.sample_rate_map.values()})

    @property
    def default_stream_channels(self) -> int:
        if self.candidate_stream_channels:
            return self.candidate_stream_channels[0]
        return 1

    def actual_rate_for_stream_channels(self, stream_channels: int) -> int:
        if stream_channels in self.sample_rate_map:
            return int(self.sample_rate_map[stream_channels])
        if len(self.sample_rate_map) == 1:
            return int(next(iter(self.sample_rate_map.values())))
        raise ValueError(
            f"profile {self.key} does not support {stream_channels} stream channels"
        )

    def supports_hardware_type(self, hardware_type: Optional[str]) -> bool:
        if not hardware_type:
            return False
        token = hardware_type.strip().upper()
        return token in self.hardware_aliases


@dataclass
class _BYBProtocolState:
    hardware_type: Optional[str] = None
    hardware_version: Optional[str] = None
    firmware_version: Optional[str] = None
    board_type: Optional[int] = None
    last_event: Optional[int] = None
    joy_state: Optional[int] = None
    p300_enabled: Optional[bool] = None
    sound_enabled: Optional[bool] = None
    high_gain_channels: set[int] = field(default_factory=set)
    high_hpf_channels: set[int] = field(default_factory=set)
    presets: Dict[int, str] = field(default_factory=dict)
    filters: Dict[int, Dict[str, Optional[float]]] = field(default_factory=dict)
    last_messages: List[str] = field(default_factory=list)

    def note_message(self, message: str) -> None:
        self.last_messages.append(message)
        if len(self.last_messages) > 32:
            del self.last_messages[:-32]


_PROFILES: Dict[str, _BYBProfile] = {
    "muscle_pro": _BYBProfile(
        key="muscle_pro",
        label="Muscle SpikerBox Pro",
        hardware_aliases=("MUSCLESB", "MSBPCDC"),
        transports=("serial", "hid"),
        bits=10,
        max_channels=4,
        candidate_stream_channels=(2, 3, 4),
        sample_rate_map={2: 10000, 3: 5000, 4: 5000},
        baud_rates=(222222,),
        requires_start=True,
        supports_board_query=True,
        infer_stream_width=True,
        description="Pre-2023 Pro family with expansion port and 2-4 channel modes.",
    ),
    "neuron_pro": _BYBProfile(
        key="neuron_pro",
        label="Neuron SpikerBox Pro",
        hardware_aliases=("NEURONSB", "NSBPCDC"),
        transports=("serial", "hid"),
        bits=10,
        max_channels=4,
        candidate_stream_channels=(2, 3, 4),
        sample_rate_map={2: 10000, 3: 5000, 4: 5000},
        baud_rates=(222222,),
        requires_start=True,
        supports_board_query=True,
        infer_stream_width=True,
        description="Pre-2023 Pro family with expansion port and 2-4 channel modes.",
    ),
    "human_spikerbox": _BYBProfile(
        key="human_spikerbox",
        label="Human SpikerBox",
        hardware_aliases=("HUMANSB",),
        transports=("serial",),
        bits=14,
        max_channels=4,
        candidate_stream_channels=(2, 3, 4),
        sample_rate_map={2: 5000, 3: 5000, 4: 5000},
        baud_rates=(222222, 500000),
        supports_board_query=True,
        supports_human_controls=True,
        infer_stream_width=True,
        description="Composite CDC device with 2-4 input channels and expansion events.",
    ),
    "neuron_pro_mfi": _BYBProfile(
        key="neuron_pro_mfi",
        label="Neuron SpikerBox Pro (Serial + MFi)",
        hardware_aliases=("NRNSBPRO",),
        transports=("serial",),
        bits=14,
        max_channels=4,
        candidate_stream_channels=(4,),
        sample_rate_map={4: 10000},
        baud_rates=(222222, 500000),
        requires_start=True,
        supports_board_query=False,
        description="Composite Neuron Pro variant that streams a fixed 4-channel 14-bit packet layout.",
    ),
    "plant": _BYBProfile(
        key="plant",
        label="Plant SpikerBox",
        hardware_aliases=("PLANTSS",),
        transports=("serial",),
        bits=10,
        max_channels=1,
        candidate_stream_channels=(1,),
        sample_rate_map={1: 10000},
        baud_rates=(222222, 230400),
        description="Single-channel CDC device.",
    ),
    "spikershield": _BYBProfile(
        key="spikershield",
        label="Muscle SpikerShield",
        hardware_aliases=("MUSCLESS", "HEARTSS"),
        transports=("serial",),
        bits=10,
        max_channels=6,
        candidate_stream_channels=(1, 2, 3, 4, 5, 6),
        sample_rate_map={1: 10000, 2: 5000, 3: 3333, 4: 2500, 5: 2000, 6: 1666},
        baud_rates=(222222, 230400),
        supports_channel_cfg=True,
        description="Arduino-based multi-channel shield family with `c:` stream-width control.",
    ),
    "heart_brain": _BYBProfile(
        key="heart_brain",
        label="Heart & Brain SpikerBox",
        hardware_aliases=("HBLEOSB",),
        transports=("serial",),
        bits=10,
        max_channels=1,
        candidate_stream_channels=(1,),
        sample_rate_map={1: 10000},
        baud_rates=(222222,),
        description="Single-channel FTDI device.",
    ),
    "hhi_new": _BYBProfile(
        key="hhi_new",
        label="Human Human Interface",
        hardware_aliases=("HHIBOX",),
        transports=("serial",),
        bits=10,
        max_channels=1,
        candidate_stream_channels=(1,),
        sample_rate_map={1: 10000},
        baud_rates=(500000,),
        description="Second-generation Human-Human-Interface with 500000 baud.",
    ),
    "spike_station": _BYBProfile(
        key="spike_station",
        label="Spike Station",
        hardware_aliases=("UNIBOX",),
        transports=("serial",),
        bits=14,
        max_channels=2,
        candidate_stream_channels=(2,),
        sample_rate_map={2: 42661},
        baud_rates=(222222, 500000),
        supports_board_query=True,
        supports_spike_station_queries=True,
        description="Two-channel 14-bit Spike Station / UniBox family.",
    ),
    "generic_serial": _BYBProfile(
        key="generic_serial",
        label="Backyard Brains (generic serial)",
        hardware_aliases=(),
        transports=("serial",),
        bits=10,
        max_channels=1,
        candidate_stream_channels=(1,),
        sample_rate_map={1: 10000},
        baud_rates=(222222, 230400, 500000),
        description="Conservative fallback when only USB IDs are available.",
    ),
}


@dataclass(frozen=True)
class _ProfileHint:
    profile_key: str
    reason: str


def _make_serial_meta(port_info) -> dict[str, Any]:
    return {
        "transport": "serial",
        "device": getattr(port_info, "device", None),
        "name": getattr(port_info, "name", None),
        "description": getattr(port_info, "description", None),
        "manufacturer": getattr(port_info, "manufacturer", None),
        "product": getattr(port_info, "product", None),
        "interface": getattr(port_info, "interface", None),
        "vid": getattr(port_info, "vid", None),
        "pid": getattr(port_info, "pid", None),
        "serial_number": getattr(port_info, "serial_number", None),
        "hwid": getattr(port_info, "hwid", None),
    }


def _make_hid_meta(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "transport": "hid",
        "path": entry.get("path"),
        "device": entry.get("path"),
        "description": entry.get("product_string") or entry.get("product") or "BYB HID device",
        "manufacturer": entry.get("manufacturer_string") or entry.get("manufacturer"),
        "product": entry.get("product_string") or entry.get("product"),
        "interface": entry.get("interface_number"),
        "vid": entry.get("vendor_id"),
        "pid": entry.get("product_id"),
        "serial_number": entry.get("serial_number"),
    }


def _candidate_profile_hints(meta: Mapping[str, Any]) -> list[_ProfileHint]:
    transport = str(meta.get("transport", "serial"))
    vid = meta.get("vid")
    pid = meta.get("pid")
    product_blob = " ".join(
        str(meta.get(key, "") or "") for key in ("product", "description", "interface")
    ).lower()

    hints: list[_ProfileHint] = []

    if transport == "hid":
        if (vid, pid) == (_BYB_VID, 0x0001):
            return [_ProfileHint("muscle_pro", "usb_id")]
        if (vid, pid) == (_BYB_VID, 0x0002):
            return [_ProfileHint("neuron_pro", "usb_id")]
        return []

    if (vid, pid) == (_BYB_VID, 0x0006):
        return [_ProfileHint("muscle_pro", "usb_id")]
    if (vid, pid) == (_BYB_VID, 0x0007):
        return [_ProfileHint("neuron_pro", "usb_id")]
    if (vid, pid) == (_BYB_VID, 0x0004):
        return [_ProfileHint("human_spikerbox", "usb_id")]
    if (vid, pid) == (_BYB_VID, 0x0009):
        return [_ProfileHint("neuron_pro_mfi", "usb_id")]
    if (vid, pid) == (_BYB_VID, 0x000D):
        return [_ProfileHint("spike_station", "usb_id")]
    if (vid, pid) == (_ARDUINO_VID, 0x8036):
        return [_ProfileHint("plant", "usb_id")]
    if (vid, pid) == (_ARDUINO_VID, 0x0043):
        return [_ProfileHint("spikershield", "usb_id")]
    if (vid, pid) == (_FTDI_VID, 0x6015):
        if "human human interface" in product_blob or "hhi 1v1" in product_blob:
            return [_ProfileHint("hhi_new", "usb_string")]
        if "heart" in product_blob or "brain" in product_blob:
            return [_ProfileHint("heart_brain", "usb_string")]
        return [
            _ProfileHint("heart_brain", "shared_usb_id"),
            _ProfileHint("hhi_new", "shared_usb_id"),
            _ProfileHint("generic_serial", "shared_usb_id"),
        ]

    if vid in {_BYB_VID, _ARDUINO_VID, _FTDI_VID}:
        return [_ProfileHint("generic_serial", "vendor_only")]
    return []


def _profiles_for_hints(hints: Iterable[_ProfileHint], transport: str) -> list[_BYBProfile]:
    seen: set[str] = set()
    resolved: list[_BYBProfile] = []
    for hint in hints:
        profile = _PROFILES.get(hint.profile_key)
        if profile is None or hint.profile_key in seen:
            continue
        if transport not in profile.transports:
            continue
        resolved.append(profile)
        seen.add(hint.profile_key)
    return resolved


def _resolve_profile(
    meta: Mapping[str, Any],
    probe_state: _BYBProtocolState,
    candidate_profiles: Sequence[_BYBProfile],
) -> tuple[_BYBProfile, bool]:
    transport = str(meta.get("transport", "serial"))
    hardware_type = (probe_state.hardware_type or "").strip().upper()

    if hardware_type:
        matches = [p for p in candidate_profiles if p.supports_hardware_type(hardware_type)]
        if not matches:
            matches = [
                profile
                for profile in _PROFILES.values()
                if transport in profile.transports and profile.supports_hardware_type(hardware_type)
            ]
        if len(matches) == 1:
            return matches[0], False
        if matches:
            return matches[0], False

    if candidate_profiles:
        return candidate_profiles[0], True

    fallback_key = "generic_serial" if transport == "serial" else "muscle_pro"
    return _PROFILES[fallback_key], True


def _profile_summary(profile: _BYBProfile) -> dict[str, Any]:
    return {
        "key": profile.key,
        "label": profile.label,
        "bits": profile.bits,
        "max_channels": profile.max_channels,
        "sample_rates": profile.sample_rates,
        "baud_rates": list(profile.baud_rates),
    }


def _encode_hid_id(path: Any) -> str:
    if isinstance(path, str):
        raw = path.encode("utf-8")
    else:
        raw = bytes(path)
    token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"hid:{token}"


def _decode_hid_id(device_id: str) -> bytes:
    token = device_id.split(":", 1)[1]
    pad = "=" * (-len(token) % 4)
    return base64.urlsafe_b64decode(token + pad)


def _iter_complete_messages(text_buffer: str) -> tuple[list[str], str]:
    messages: list[str] = []
    last = text_buffer.rfind(";")
    if last < 0:
        return messages, text_buffer[-512:]

    complete = text_buffer[: last + 1]
    remainder = text_buffer[last + 1 :]
    for token in complete.split(";"):
        token = token.strip()
        if ":" not in token:
            continue
        messages.append(token + ";")
    return messages, remainder[-512:]


def _parse_bool_scalar(value: str) -> Optional[bool]:
    value = value.strip()
    if value == "0":
        return False
    if value == "1":
        return True
    return None


def _parse_channel_value(value: str) -> tuple[Optional[int], str]:
    head, _, tail = value.partition("_")
    try:
        channel = int(head)
    except ValueError:
        return None, tail
    return channel, tail


def _parse_gamepad_state(value: str) -> Optional[int]:
    chunks = re.findall(r"0b([01]{8})", value)
    if len(chunks) != 2:
        return None
    high_nibble = int(chunks[0][-4:], 2)
    low_nibble = int(chunks[1][-4:], 2)
    return (high_nibble << 4) | low_nibble


def _update_protocol_state(state: _BYBProtocolState, message: str) -> None:
    token = message.strip()
    if not token.endswith(";"):
        token += ";"
    state.note_message(token)

    body = token[:-1]
    key, _, raw_value = body.partition(":")
    key = key.strip()
    value = raw_value.strip()
    key_lower = key.lower()

    if key == "HWT":
        state.hardware_type = value.upper()
        return
    if key == "HWV":
        state.hardware_version = value
        return
    if key == "FWV":
        state.firmware_version = value
        return
    if key == "BRD":
        try:
            state.board_type = int(value)
        except ValueError:
            pass
        return
    if key == "EVNT":
        try:
            state.last_event = int(value)
        except ValueError:
            pass
        return
    if key == "JOY":
        parsed = _parse_gamepad_state(value)
        if parsed is not None:
            state.joy_state = parsed
        return
    if key_lower == "p300":
        state.p300_enabled = _parse_bool_scalar(value)
        return
    if key_lower == "sound":
        state.sound_enabled = _parse_bool_scalar(value)
        return
    if key_lower == "preset":
        channel, preset = _parse_channel_value(value)
        if channel is not None:
            state.presets[channel] = preset
        return
    if key_lower in {"hpfilter", "lpfilter", "notch"}:
        channel, raw_filter_value = _parse_channel_value(value)
        if channel is None:
            return
        filt = state.filters.setdefault(channel, {"hpfilter": None, "lpfilter": None, "notch": None})
        try:
            numeric = float(raw_filter_value)
        except ValueError:
            numeric = None
        if numeric is not None and numeric < 0:
            numeric = None
        filt[key_lower] = numeric


def _extract_wrapped_message_payloads(buffer: bytearray) -> list[str]:
    messages: list[str] = []

    while True:
        start_idx = buffer.find(_MSG_START)
        if start_idx < 0:
            if len(buffer) > len(_MSG_START):
                del buffer[:-len(_MSG_START)]
            break

        if start_idx > 0:
            del buffer[:start_idx]
            start_idx = 0

        end_idx = buffer.find(_MSG_END, len(_MSG_START))
        if end_idx < 0:
            break

        payload = bytes(buffer[len(_MSG_START) : end_idx])
        del buffer[: end_idx + len(_MSG_END)]
        text = payload.decode("ascii", errors="ignore")
        wrapped_messages, _ = _iter_complete_messages(text)
        messages.extend(wrapped_messages)

    return messages


class _BYBTransport:
    kind: str = "serial"

    def reset_input_buffer(self) -> None:
        return

    def write_command(self, cmd: str) -> None:
        raise NotImplementedError

    def read(self, timeout_ms: int = 0) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        raise NotImplementedError

    @property
    def baudrate(self) -> Optional[int]:
        return None


class _SerialTransport(_BYBTransport):
    kind = "serial"

    def __init__(
        self,
        port: str,
        *,
        baudrate: int,
        timeout: float = 0.05,
        write_timeout: float = 0.05,
        serial_factory: Any = None,
    ) -> None:
        if serial is None and serial_factory is None:
            raise RuntimeError("pyserial is not installed.")
        self._factory = serial_factory or serial.Serial
        self._port = port
        self._baudrate = int(baudrate)
        self._timeout = float(timeout)
        self._write_timeout = float(write_timeout)
        self._ser: Any = None

    def open(self) -> None:
        self._ser = self._factory(
            port=self._port,
            baudrate=self._baudrate,
            timeout=self._timeout,
            write_timeout=self._write_timeout,
        )

    @property
    def serial_handle(self) -> Any:
        return self._ser

    @property
    def is_open(self) -> bool:
        return bool(self._ser is not None and getattr(self._ser, "is_open", True))

    @property
    def baudrate(self) -> Optional[int]:
        return self._baudrate

    def reset_input_buffer(self) -> None:
        if self._ser is not None and hasattr(self._ser, "reset_input_buffer"):
            self._ser.reset_input_buffer()

    def write_command(self, cmd: str) -> None:
        if self._ser is None:
            raise RuntimeError("serial transport is not open")
        self._ser.write(cmd.encode("ascii"))

    def read(self, timeout_ms: int = 0) -> bytes:
        if self._ser is None:
            return b""

        available = int(getattr(self._ser, "in_waiting", 0) or 0)
        if available > 0:
            return bytes(self._ser.read(min(available, _MAX_READ_SIZE)))

        if timeout_ms > 0:
            chunk = self._ser.read(1)
            if not chunk:
                return b""
            available = int(getattr(self._ser, "in_waiting", 0) or 0)
            if available > 0:
                chunk += self._ser.read(min(available, _MAX_READ_SIZE - len(chunk)))
            return bytes(chunk)
        return b""

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None


class _CompatSerialTransport(_BYBTransport):
    kind = "serial"

    def __init__(self, serial_like: Any) -> None:
        self._ser = serial_like

    @property
    def is_open(self) -> bool:
        return self._ser is not None and getattr(self._ser, "is_open", True)

    def write_command(self, cmd: str) -> None:
        if self._ser is None:
            raise RuntimeError("serial transport is not open")
        payload = cmd.encode("ascii")
        if hasattr(self._ser, "write"):
            self._ser.write(payload)

    def read(self, timeout_ms: int = 0) -> bytes:
        if self._ser is None:
            return b""
        available = int(getattr(self._ser, "in_waiting", 0) or 0)
        if available > 0:
            return bytes(self._ser.read(min(available, _MAX_READ_SIZE)))
        if timeout_ms > 0:
            time.sleep(timeout_ms / 1000.0)
            available = int(getattr(self._ser, "in_waiting", 0) or 0)
            if available > 0:
                return bytes(self._ser.read(min(available, _MAX_READ_SIZE)))
        return b""

    def close(self) -> None:
        return


class _HIDTransport(_BYBTransport):
    kind = "hid"

    def __init__(self, path: bytes, hid_module: Any = None) -> None:
        if hid is None and hid_module is None:
            raise RuntimeError("hidapi is not installed.")
        self._hid = hid_module or hid
        self._path = bytes(path)
        self._dev: Any = None

    def open(self) -> None:
        if hasattr(self._hid, "device"):
            self._dev = self._hid.device()
            self._dev.open_path(self._path)
        elif hasattr(self._hid, "Device"):
            self._dev = self._hid.Device(path=self._path)
        else:
            raise RuntimeError("Unsupported hidapi binding; expected `device()` or `Device()`")
        if hasattr(self._dev, "set_nonblocking"):
            self._dev.set_nonblocking(1)

    @property
    def is_open(self) -> bool:
        return self._dev is not None

    def write_command(self, cmd: str) -> None:
        if self._dev is None:
            raise RuntimeError("hid transport is not open")
        payload = cmd.encode("ascii")
        packet = bytearray(_HID_PACKET_SIZE)
        packet[0] = 0x3F
        packet[1] = 0x3E
        packet[2 : 2 + min(len(payload), _HID_PAYLOAD_SIZE)] = payload[:_HID_PAYLOAD_SIZE]
        if hasattr(self._dev, "send_feature_report"):
            self._dev.send_feature_report(bytes(packet))
        elif hasattr(self._dev, "write"):
            self._dev.write(bytes(packet))
        else:
            raise RuntimeError("HID binding does not support write path")

    def read(self, timeout_ms: int = 0) -> bytes:
        if self._dev is None:
            return b""

        if hasattr(self._dev, "read"):
            try:
                raw = self._dev.read(_HID_PACKET_SIZE, timeout_ms)
            except TypeError:
                raw = self._dev.read(_HID_PACKET_SIZE)
        else:
            return b""

        if not raw:
            return b""
        if isinstance(raw, list):
            packet = bytes(raw)
        else:
            packet = bytes(raw)

        if len(packet) >= _HID_PACKET_SIZE + 1 and packet[0] == 0:
            packet = packet[1:]
        if len(packet) < 2:
            return b""

        payload_length = min(packet[1], _HID_PAYLOAD_SIZE)
        return packet[2 : 2 + payload_length]

    def close(self) -> None:
        if self._dev is not None:
            try:
                if hasattr(self._dev, "close"):
                    self._dev.close()
            finally:
                self._dev = None


class _BYBDecoder:
    def __init__(self, *, bits: int, candidate_widths: Sequence[int]) -> None:
        self.bits = int(bits)
        self._raw = bytearray()
        self._msg_buf = bytearray()
        self._candidate_widths = tuple(sorted({int(v) for v in candidate_widths if int(v) > 0}))
        self._stream_width: Optional[int] = None
        self._frame_mode: str = "per_sample"

    @property
    def stream_width(self) -> Optional[int]:
        return self._stream_width

    def feed(self, data: bytes) -> tuple[list[str], Optional[np.ndarray]]:
        if data:
            self._raw.extend(data)
            self._msg_buf.extend(data)

        messages = _extract_wrapped_message_payloads(self._msg_buf)
        if self._stream_width is None:
            self._infer_stream_width()

        decoded = self._decode_available_frames()
        return messages, decoded

    def _infer_stream_width(self) -> None:
        if not self._candidate_widths:
            return

        best: tuple[int, int, int, str] | None = None  # (score_bytes, width, offset, mode)
        for width in self._candidate_widths:
            frame_size = width * 2
            if len(self._raw) < frame_size * 2:
                continue
            max_offset = min(frame_size, len(self._raw))
            for offset in range(max_offset):
                for mode in ("frame_start", "per_sample"):
                    score = self._score_alignment(width, offset, mode)
                    if score <= 0:
                        continue
                    if best is None or score > best[0]:
                        best = (score, width, offset, mode)
                    elif best is not None and score == best[0]:
                        _, best_width, _, best_mode = best
                        if width > best_width or (width == best_width and mode == "frame_start" and best_mode != "frame_start"):
                            best = (score, width, offset, mode)

        if best is None:
            self._drop_leading_garbage()
            return

        score, width, offset, mode = best
        frame_size = width * 2
        if score < frame_size * 2:
            return

        if offset > 0:
            del self._raw[:offset]
        self._stream_width = width
        self._frame_mode = mode

    def _score_alignment(self, width: int, offset: int, mode: str) -> int:
        frame_size = width * 2
        usable = len(self._raw) - offset
        usable -= usable % frame_size
        if usable <= 0:
            return 0

        chunk = memoryview(self._raw)[offset : offset + usable]
        score = 0
        for pos, byte in enumerate(chunk):
            if self._byte_flag_expected(pos, frame_size, mode):
                if byte & 0x80:
                    score += 1
                else:
                    break
            else:
                if byte & 0x80:
                    break
                score += 1
        return score

    @staticmethod
    def _byte_flag_expected(pos: int, frame_size: int, mode: str) -> bool:
        if mode == "frame_start":
            return (pos % frame_size) == 0
        return (pos % 2) == 0

    def _drop_leading_garbage(self) -> None:
        idx = next((i for i, b in enumerate(self._raw) if b & 0x80), -1)
        if idx < 0:
            self._raw.clear()
        elif idx > 0:
            del self._raw[:idx]

    def _decode_available_frames(self) -> Optional[np.ndarray]:
        if self._stream_width is None:
            return None

        frame_size = self._stream_width * 2
        total_complete = len(self._raw) // frame_size
        if total_complete <= 0:
            return None

        valid_bytes = 0
        for frame_idx in range(total_complete):
            start = frame_idx * frame_size
            end = start + frame_size
            if self._frame_bytes_valid(self._raw[start:end]):
                valid_bytes = end
                continue

            if frame_idx == 0:
                self._resync_after_invalid()
                return None

            break

        if valid_bytes <= 0:
            return None

        payload = bytes(self._raw[:valid_bytes])
        del self._raw[:valid_bytes]

        frame_count = len(payload) // frame_size
        raw_bytes = np.frombuffer(payload, dtype=np.uint8)
        frame_bytes = raw_bytes.reshape(frame_count, self._stream_width, 2)
        highs = frame_bytes[:, :, 0].astype(np.int32)
        lows = frame_bytes[:, :, 1].astype(np.int32)
        raw_vals = ((highs & 0x7F) << 7) | (lows & 0x7F)

        center_val = float(1 << (self.bits - 1))
        decoded = (raw_vals - center_val) / center_val
        return np.asarray(decoded, dtype=np.float32)

    def _frame_bytes_valid(self, frame_bytes: memoryview) -> bool:
        frame_size = len(frame_bytes)
        for pos, byte in enumerate(frame_bytes):
            if self._byte_flag_expected(pos, frame_size, self._frame_mode):
                if (byte & 0x80) == 0:
                    return False
            else:
                if byte & 0x80:
                    return False
        return True

    def _resync_after_invalid(self) -> None:
        if self._stream_width is None:
            return
        frame_size = self._stream_width * 2

        for offset in range(1, min(len(self._raw), frame_size * 2 + 1)):
            for mode in (self._frame_mode, "frame_start", "per_sample"):
                score = self._score_alignment(self._stream_width, offset, mode)
                if score >= frame_size * 2:
                    del self._raw[:offset]
                    self._frame_mode = mode
                    return

        self._stream_width = None
        self._frame_mode = "per_sample"
        self._drop_leading_garbage()


class BackyardBrainsSource(BaseDevice):
    """
    Backyard Brains USB backend with probe-based profile resolution.

    The public contract remains the standard BaseDevice lifecycle:
    open -> configure -> start/stop -> close.
    """

    _SYNC_TIMEOUT_S: float = _SYNC_TIMEOUT_S_DEFAULT

    @classmethod
    def device_class_name(cls) -> str:
        return "Backyard Brains"

    def __init__(self, queue_maxsize: int = 64) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        self._ser: Any = None  # legacy hook used by existing tests
        self._transport: Optional[_BYBTransport] = None
        self._transport_kind: str = "serial"
        self._producer_thread: Optional[threading.Thread] = None
        self._profile: Optional[_BYBProfile] = None
        self._profile_candidates: list[_BYBProfile] = []
        self._profile_hint_reason: Optional[str] = None
        self._fallback_identification: bool = True
        self._baudrate: Optional[int] = None
        self._bits = 10
        self._requires_start = False
        self._supports_channel_cfg = False
        self._stream_channel_count = 1
        self._decoder_candidate_widths: tuple[int, ...] = (1,)
        self._resolved_meta: dict[str, Any] = {}
        self._protocol_state = _BYBProtocolState()
        self._max_channels = 1
        self._runtime_width_inferred = False

    @classmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        devices: list[DeviceInfo] = []
        devices.extend(cls._list_serial_devices())
        devices.extend(cls._list_hid_devices())
        return devices

    @classmethod
    def _list_serial_devices(cls) -> List[DeviceInfo]:
        if serial is None:
            return []

        devices: list[DeviceInfo] = []
        try:
            ports = serial.tools.list_ports.comports()
        except Exception as exc:
            _LOGGER.error("Failed to enumerate serial ports: %s", exc)
            return []

        for port in ports:
            meta = _make_serial_meta(port)
            if not cls._is_supported_serial_candidate(meta):
                continue
            hints = _candidate_profile_hints(meta)
            profiles = _profiles_for_hints(hints, "serial")
            profile_hint = profiles[0] if profiles else _PROFILES["generic_serial"]
            display_name = profile_hint.label
            port_label = getattr(port, "name", None) or getattr(port, "device", None)
            if port_label:
                display_name = f"{display_name} ({port_label})"
            details = {
                "transport": "serial",
                "vid": meta.get("vid"),
                "pid": meta.get("pid"),
                "probe_status": "not_probed",
                "profile_hint": profile_hint.label,
                "profile_hint_reason": hints[0].reason if hints else "vendor_only",
                "hid_support_available": hid is not None,
            }
            if hints and len(profiles) > 1:
                details["fallback_identification"] = True
            devices.append(
                DeviceInfo(
                    id=str(meta.get("device")),
                    name=display_name,
                    vendor=meta.get("manufacturer") or "Backyard Brains",
                    details=details,
                )
            )
        return devices

    @classmethod
    def _list_hid_devices(cls) -> List[DeviceInfo]:
        if hid is None:
            return []

        devices: list[DeviceInfo] = []
        try:
            entries = hid.enumerate(_BYB_VID, 0)
        except Exception as exc:
            _LOGGER.error("Failed to enumerate HID devices: %s", exc)
            return []

        for entry in entries:
            meta = _make_hid_meta(entry)
            if not cls._is_supported_hid_candidate(meta):
                continue
            hints = _candidate_profile_hints(meta)
            profiles = _profiles_for_hints(hints, "hid")
            if not profiles:
                continue
            profile = profiles[0]
            details = {
                "transport": "hid",
                "vid": meta.get("vid"),
                "pid": meta.get("pid"),
                "probe_status": "not_probed",
                "profile_hint": profile.label,
                "profile_hint_reason": hints[0].reason if hints else "usb_id",
                "hid_support_available": True,
            }
            devices.append(
                DeviceInfo(
                    id=_encode_hid_id(meta["path"]),
                    name=profile.label,
                    vendor=meta.get("manufacturer") or "Backyard Brains",
                    details=details,
                )
            )
        return devices

    @staticmethod
    def _is_supported_serial_candidate(meta: Mapping[str, Any]) -> bool:
        vid = meta.get("vid")
        pid = meta.get("pid")
        if (vid, pid) in _BOOTLOADER_PIDS:
            return False
        if vid == _BYB_VID:
            return True
        if (vid, pid) in {
            (_ARDUINO_VID, 0x8036),
            (_ARDUINO_VID, 0x0043),
            (_FTDI_VID, 0x6015),
        }:
            return True
        description = str(meta.get("description") or "").lower()
        product = str(meta.get("product") or "").lower()
        return "spiker" in description or "spiker" in product

    @staticmethod
    def _is_supported_hid_candidate(meta: Mapping[str, Any]) -> bool:
        return (meta.get("vid"), meta.get("pid")) in {
            (_BYB_VID, 0x0001),
            (_BYB_VID, 0x0002),
        }

    def _lookup_port(self, device_id: str):
        if serial is None:
            return None
        try:
            for port in serial.tools.list_ports.comports():
                if getattr(port, "device", None) == device_id:
                    return port
        except Exception as exc:
            _LOGGER.debug("Failed to lookup serial port %s: %s", device_id, exc)
        return None

    def _lookup_hid_entry(self, device_id: str) -> Optional[dict[str, Any]]:
        if hid is None or not device_id.startswith("hid:"):
            return None
        wanted = _decode_hid_id(device_id)
        try:
            entries = hid.enumerate(_BYB_VID, 0)
        except Exception as exc:
            _LOGGER.debug("Failed to enumerate HID for %s: %s", device_id, exc)
            return None
        for entry in entries:
            path = entry.get("path")
            if isinstance(path, str):
                path_bytes = path.encode("utf-8")
            else:
                path_bytes = bytes(path)
            if path_bytes == wanted:
                return dict(entry)
        return None

    def get_capabilities(self, device_id: str) -> Capabilities:
        if self._profile is not None:
            profile = self._profile
            max_channels = profile.max_channels
            sample_rates = profile.sample_rates
            notes = (
                f"{profile.label}. "
                f"Transport={self._transport_kind}. "
                f"{'Fallback identification.' if self._fallback_identification else 'Probed profile.'}"
            )
        else:
            meta = self._meta_for_device(device_id)
            hints = _candidate_profile_hints(meta)
            profiles = _profiles_for_hints(hints, meta.get("transport", "serial"))
            if not profiles:
                profiles = [_PROFILES["generic_serial"]]
            max_channels = max(profile.max_channels for profile in profiles)
            sample_rates = sorted({rate for profile in profiles for rate in profile.sample_rates})
            notes = (
                "Backyard Brains USB device. "
                "Capabilities are descriptor-based until the device is opened and probed."
            )
        return Capabilities(
            max_channels_in=max_channels,
            sample_rates=sample_rates,
            dtype="float32",
            notes=notes,
        )

    def list_available_channels(self, device_id: str) -> List[ChannelInfo]:
        profile = self._profile
        if profile is None:
            meta = self._meta_for_device(device_id)
            profiles = _profiles_for_hints(_candidate_profile_hints(meta), meta.get("transport", "serial"))
            if profiles:
                profile = profiles[0]
            else:
                profile = _PROFILES["generic_serial"]

        visible_channels = self._visible_channel_count(profile)
        channels: list[ChannelInfo] = []
        for idx in range(visible_channels):
            label = f"Channel {idx + 1}"
            if profile.key in {"muscle_pro", "neuron_pro", "human_spikerbox", "neuron_pro_mfi"} and idx >= 2:
                label = f"Expansion {idx - 1}"
            channels.append(ChannelInfo(id=idx, name=label, units="V", range=(-5.0, 5.0)))
        return channels

    def _resolved_stream_width_candidates(
        self,
        profile: Optional[_BYBProfile] = None,
    ) -> tuple[int, ...]:
        profile = profile or self._profile
        if profile is None:
            return (1,)

        board_type = self._protocol_state.board_type
        if profile.key in {"muscle_pro", "neuron_pro"}:
            if board_type in {None, 0, 4, 5}:
                return (2,)
            if board_type == 1:
                return (3, 4)
        elif profile.key == "human_spikerbox":
            if board_type in {None, 0, 4, 5}:
                return (2,)
            if board_type == 1:
                return (3, 4)
        candidates = tuple(int(width) for width in profile.candidate_stream_channels if int(width) > 0)
        return candidates or (profile.default_stream_channels,)

    def _visible_channel_count(self, profile: _BYBProfile) -> int:
        return max(self._resolved_stream_width_candidates(profile))

    def _meta_for_device(self, device_id: str) -> dict[str, Any]:
        if device_id.startswith("hid:"):
            entry = self._lookup_hid_entry(device_id)
            if entry is not None:
                return _make_hid_meta(entry)
            return {"transport": "hid", "device": device_id}
        port = self._lookup_port(device_id)
        if port is not None:
            return _make_serial_meta(port)
        return {"transport": "serial", "device": device_id}

    def _open_impl(self, device_id: str) -> None:
        meta = self._meta_for_device(device_id)
        transport_kind = str(meta.get("transport", "serial"))
        hints = _candidate_profile_hints(meta)
        candidates = _profiles_for_hints(hints, transport_kind)

        if transport_kind == "hid":
            if hid is None:
                raise RuntimeError("hidapi is not installed; HID Backyard Brains devices are unavailable.")
            entry = self._lookup_hid_entry(device_id)
            if entry is None:
                raise RuntimeError(f"Unable to resolve HID device id {device_id!r}")
            meta = _make_hid_meta(entry)
            transport = _HIDTransport(bytes(meta["path"]))
            transport.open()
            self._ser = None
        else:
            if serial is None:
                raise RuntimeError("pyserial is not installed.")
            transport = self._open_serial_transport(str(meta.get("device")), candidates)
            self._ser = getattr(transport, "serial_handle", None)

        self._transport = transport
        self._transport_kind = transport.kind
        self._resolved_meta = dict(meta)
        self._profile_candidates = list(candidates)
        self._profile_hint_reason = hints[0].reason if hints else None

        probe_state = self._probe_transport(transport, candidates)
        profile, fallback_identification = _resolve_profile(meta, probe_state, candidates)
        self._profile = profile
        self._fallback_identification = fallback_identification
        self._protocol_state = probe_state
        self._bits = profile.bits
        self._max_channels = profile.max_channels
        self._requires_start = profile.requires_start
        self._supports_channel_cfg = profile.supports_channel_cfg
        self._baudrate = transport.baudrate

        self._decoder_candidate_widths = self._resolved_stream_width_candidates(profile)
        self._stream_channel_count = self._decoder_candidate_widths[0]
        self._runtime_width_inferred = profile.infer_stream_width and len(self._decoder_candidate_widths) > 1

        _LOGGER.info(
            "Opened BYB device transport=%s profile=%s fallback=%s baud=%s hw=%s fw=%s",
            self._transport_kind,
            profile.label,
            self._fallback_identification,
            self._baudrate,
            self._protocol_state.hardware_type,
            self._protocol_state.firmware_version,
        )

        self._query_optional_state()

    def _open_serial_transport(
        self,
        device_id: str,
        candidate_profiles: Sequence[_BYBProfile],
    ) -> _SerialTransport:
        baud_candidates = self._preferred_baud_rates(candidate_profiles)
        if not baud_candidates:
            baud_candidates = [222222]

        # Keep the legacy Serial+MFi path conservative. The older adapter opened
        # this profile at 222222 baud without probe traffic and it worked on real
        # hardware; use the same behavior here instead of gating the open on
        # probe replies from commands the guide does not list for this profile.
        if len(candidate_profiles) == 1 and candidate_profiles[0].key == "neuron_pro_mfi":
            transport = _SerialTransport(device_id, baudrate=baud_candidates[0])
            transport.open()
            return transport

        last_exc: Optional[Exception] = None
        fallback_baud: Optional[int] = None

        for baudrate in baud_candidates:
            transport = _SerialTransport(device_id, baudrate=baudrate)
            try:
                transport.open()
            except Exception as exc:
                last_exc = exc
                continue

            if fallback_baud is None:
                fallback_baud = baudrate

            probe_state = self._probe_transport(transport, candidate_profiles)
            if probe_state.hardware_type or probe_state.firmware_version:
                return transport
            transport.close()

        if fallback_baud is not None:
            transport = _SerialTransport(device_id, baudrate=fallback_baud)
            transport.open()
            return transport
        raise RuntimeError(f"Failed to open serial port {device_id}: {last_exc}")

    @staticmethod
    def _preferred_baud_rates(candidate_profiles: Sequence[_BYBProfile]) -> list[int]:
        bauds: list[int] = []
        for profile in candidate_profiles:
            for baud in profile.baud_rates:
                if baud not in bauds:
                    bauds.append(int(baud))
        if not bauds:
            bauds = [222222, 230400, 500000]
        return bauds

    def _probe_transport(
        self,
        transport: _BYBTransport,
        candidate_profiles: Sequence[_BYBProfile],
    ) -> _BYBProtocolState:
        state = _BYBProtocolState()
        if not transport.is_open:
            return state

        if len(candidate_profiles) == 1 and candidate_profiles[0].key == "neuron_pro_mfi":
            return state

        try:
            transport.reset_input_buffer()
        except Exception:
            pass

        if any(profile.requires_start for profile in candidate_profiles):
            try:
                transport.write_command("h:;")
                time.sleep(0.01)
            except Exception:
                pass

        wrapped_buffer = bytearray()
        text_buffer = ""

        for command in _DEFAULT_PROBE_COMMANDS:
            try:
                transport.write_command(command)
            except Exception:
                continue

            deadline = time.monotonic() + _PROBE_RX_TIMEOUT_S
            while time.monotonic() < deadline:
                chunk = transport.read(timeout_ms=25)
                if not chunk:
                    time.sleep(0.002)
                    continue

                wrapped_buffer.extend(chunk)
                for message in _extract_wrapped_message_payloads(wrapped_buffer):
                    _update_protocol_state(state, message)

                text_buffer += chunk.decode("ascii", errors="ignore")
                messages, text_buffer = _iter_complete_messages(text_buffer)
                for message in messages:
                    _update_protocol_state(state, message)

                if state.hardware_type and (state.firmware_version or command == "b:;"):
                    break

        return state

    def _query_optional_state(self) -> None:
        if self._transport is None or self._profile is None:
            return

        extra_commands: list[str] = []
        if self._profile.supports_board_query:
            extra_commands.append("board:;")
        if self._profile.supports_human_controls:
            extra_commands.extend(["p300?:;", "sound?:;"])
        if self._profile.supports_spike_station_queries:
            extra_commands.extend(["preset?:0;", "filter?:0;"])

        if not extra_commands:
            return

        wrapped_buffer = bytearray()
        text_buffer = ""
        for command in extra_commands:
            try:
                self._transport.write_command(command)
            except Exception:
                continue
            deadline = time.monotonic() + _PROBE_RX_TIMEOUT_S
            while time.monotonic() < deadline:
                chunk = self._transport.read(timeout_ms=25)
                if not chunk:
                    time.sleep(0.002)
                    continue
                wrapped_buffer.extend(chunk)
                for message in _extract_wrapped_message_payloads(wrapped_buffer):
                    _update_protocol_state(self._protocol_state, message)
                text_buffer += chunk.decode("ascii", errors="ignore")
                messages, text_buffer = _iter_complete_messages(text_buffer)
                for message in messages:
                    _update_protocol_state(self._protocol_state, message)

    def _close_impl(self) -> None:
        if self._transport is not None:
            try:
                self._transport.close()
            finally:
                self._transport = None
        self._ser = None
        self._profile = None
        self._profile_candidates = []
        self._profile_hint_reason = None
        self._fallback_identification = True
        self._baudrate = None
        self._resolved_meta = {}
        self._protocol_state = _BYBProtocolState()
        self._stream_channel_count = 1
        self._decoder_candidate_widths = (1,)
        self._runtime_width_inferred = False

    def _configure_impl(
        self,
        sample_rate: int,
        channels: Sequence[int],
        chunk_size: int,
        **options: Any,
    ) -> ActualConfig:
        if self._transport is None or self._profile is None:
            raise RuntimeError("Device is not open.")

        all_channels = self.list_available_channels(self._device_id)
        selected_channels = [all_channels[i] for i in channels if i < len(all_channels)]
        if not selected_channels:
            selected_channels = [all_channels[0]]

        highest_selected_id = max(ch.id for ch in selected_channels)

        if self._profile.supports_channel_cfg:
            required_stream_channels = max(1, highest_selected_id + 1)
            supported_widths = [
                width for width in self._profile.candidate_stream_channels if width >= required_stream_channels
            ]
            if not supported_widths:
                raise ValueError(
                    f"{self._profile.label} cannot expose selected channel ids {channels}"
                )
            stream_channels = supported_widths[0]
            actual_rate = self._profile.actual_rate_for_stream_channels(stream_channels)
            requested = int(sample_rate) if sample_rate else actual_rate
            if requested != actual_rate:
                raise ValueError(
                    f"{self._profile.label} uses {actual_rate} Hz when streaming "
                    f"{stream_channels} channels; requested {requested} Hz"
                )
            if self._requires_start:
                self._send_command("h:;")
                time.sleep(0.01)
            self._send_command(f"c:{stream_channels};")
            time.sleep(0.01)
            self._stream_channel_count = stream_channels
            self._decoder_candidate_widths = (stream_channels,)
            self._runtime_width_inferred = False
        else:
            dynamic_width_profile = self._profile.key in {
                "muscle_pro",
                "neuron_pro",
                "human_spikerbox",
            }
            minimum_stream_width = max(1, highest_selected_id + 1)
            base_candidate_widths = self._resolved_stream_width_candidates(self._profile)
            if dynamic_width_profile:
                candidate_widths = base_candidate_widths
            else:
                candidate_widths = tuple(
                    width for width in base_candidate_widths if width >= minimum_stream_width
                )
            if not candidate_widths:
                raise ValueError(
                    f"{self._profile.label} cannot expose selected channel ids {channels}"
                )
            rate_candidates = {
                self._profile.actual_rate_for_stream_channels(width) for width in candidate_widths
            }
            requested = int(sample_rate) if sample_rate else 0
            if requested > 0:
                matching_widths = tuple(
                    width
                    for width in candidate_widths
                    if self._profile.actual_rate_for_stream_channels(width) == requested
                )
                if matching_widths:
                    candidate_widths = matching_widths
                    actual_rate = requested
                elif len(rate_candidates) == 1:
                    actual_rate = next(iter(rate_candidates))
                    _LOGGER.info(
                        "BYB sample rate request %s Hz overridden to %s Hz for %s",
                        requested,
                        actual_rate,
                        self._profile.label,
                    )
                else:
                    raise ValueError(
                        f"{self._profile.label} does not support {requested} Hz for the current hardware state"
                    )
            elif len(rate_candidates) == 1:
                actual_rate = next(iter(rate_candidates))
            else:
                actual_rate = self._profile.actual_rate_for_stream_channels(candidate_widths[0])

            self._stream_channel_count = candidate_widths[0]
            self._decoder_candidate_widths = candidate_widths
            self._runtime_width_inferred = len(candidate_widths) > 1

        if self._profile.supports_spike_station_queries:
            self._query_optional_state()

        _LOGGER.info(
            "Configured BYB profile=%s board=%s active_channels=%s decode_widths=%s actual_rate=%s",
            self._profile.key,
            self._protocol_state.board_type,
            [ch.id for ch in selected_channels],
            self._decoder_candidate_widths,
            actual_rate,
        )

        return ActualConfig(
            sample_rate=actual_rate,
            channels=selected_channels,
            chunk_size=chunk_size,
            dtype="float32",
        )

    def _start_impl(self) -> None:
        if self._transport is None:
            raise RuntimeError("Device transport is not open")

        if self._requires_start:
            self._send_command("start:;")
            time.sleep(0.02)

        self._producer_thread = threading.Thread(target=self._run_loop, daemon=True, name="BackyardBrainsDAQ")
        self._producer_thread.start()

    def _stop_impl(self) -> None:
        if self._producer_thread is not None and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1.0)
        self._producer_thread = None

        if self._requires_start and self._transport is not None and self._transport.is_open:
            try:
                self._send_command("h:;")
            except Exception as exc:
                _LOGGER.debug("Failed to send BYB halt command: %s", exc)

    def _send_command(self, cmd: str) -> None:
        if self._transport is None:
            raise RuntimeError("Device transport is not open")
        self._transport.write_command(cmd)

    def _active_transport(self) -> Optional[_BYBTransport]:
        if self._transport is not None:
            return self._transport
        if self._ser is not None:
            return _CompatSerialTransport(self._ser)
        return None

    def _run_loop(self) -> None:
        transport = self._active_transport()
        if transport is None or self.config is None:
            return

        if self._profile is not None and self._profile.key == "neuron_pro_mfi":
            self._run_loop_legacy_serial_mfi(transport)
            return

        candidate_widths = self._decoder_candidate_widths or (self._stream_channel_count,)
        decoder = _BYBDecoder(bits=self._bits, candidate_widths=candidate_widths)
        raw_chunk_size = self.config.chunk_size
        sample_buffer = np.zeros((raw_chunk_size, max(candidate_widths)), dtype=np.float32)
        sample_idx = 0
        last_emit_time = time.monotonic()
        sync_lost = False
        sync_lost_time = 0.0

        while not self.stop_event.is_set():
            try:
                data = transport.read(timeout_ms=25)
            except Exception as exc:
                _LOGGER.error("BYB transport read error: %s", exc)
                break

            if not data:
                time.sleep(0.001)
            messages, decoded = decoder.feed(data)

            for message in messages:
                _update_protocol_state(self._protocol_state, message)

            if decoder.stream_width is not None and decoder.stream_width != self._stream_channel_count:
                self._stream_channel_count = decoder.stream_width
                self._runtime_width_inferred = False

            if decoded is not None and decoded.size:
                if decoded.shape[1] > sample_buffer.shape[1]:
                    sample_buffer = np.zeros((raw_chunk_size, decoded.shape[1]), dtype=np.float32)
                source_idx = 0
                total_frames = decoded.shape[0]
                while source_idx < total_frames:
                    needed = raw_chunk_size - sample_idx
                    to_copy = min(needed, total_frames - source_idx)
                    sample_buffer[sample_idx : sample_idx + to_copy, : decoded.shape[1]] = decoded[
                        source_idx : source_idx + to_copy
                    ]
                    sample_idx += to_copy
                    source_idx += to_copy

                    if sample_idx >= raw_chunk_size:
                        self._emit_active(sample_buffer, sample_idx)
                        sample_idx = 0
                        last_emit_time = time.monotonic()
                        if sync_lost:
                            _LOGGER.info(
                                "BYB frame sync recovered after %.1f s.",
                                last_emit_time - sync_lost_time,
                            )
                            sync_lost = False

            elapsed = time.monotonic() - last_emit_time
            if elapsed > self._SYNC_TIMEOUT_S and not sync_lost:
                sync_lost = True
                sync_lost_time = time.monotonic()
                _LOGGER.warning(
                    "BYB frame sync lost: no valid frames for %.1f s. "
                    "profile=%s board=%s decode_widths=%s current_width=%s. "
                    "Resetting decoder; check device mode, USB cable, and electrode connection.",
                    elapsed,
                    None if self._profile is None else self._profile.key,
                    self._protocol_state.board_type,
                    self._decoder_candidate_widths,
                    self._stream_channel_count,
                )
                self.note_xrun()
                decoder = _BYBDecoder(bits=self._bits, candidate_widths=candidate_widths)
                sample_idx = 0

    def _run_loop_legacy_serial_mfi(self, transport: _BYBTransport) -> None:
        if self.config is None:
            return

        stream_channels = max(1, int(self._stream_channel_count or 4))
        bits = max(8, min(self._bits, 16))
        center_val = float(1 << (bits - 1))
        raw_chunk_size = self.config.chunk_size
        sample_buffer = np.zeros((raw_chunk_size, stream_channels), dtype=np.float32)
        sample_idx = 0
        raw_buffer = bytearray()
        last_emit_time = time.monotonic()
        sync_lost = False
        sync_lost_time = 0.0
        frame_size = stream_channels * 2

        while not self.stop_event.is_set():
            try:
                data = transport.read(timeout_ms=25)
            except Exception as exc:
                _LOGGER.error("BYB transport read error: %s", exc)
                break

            if not data:
                time.sleep(0.001)
            else:
                raw_buffer.extend(data)

            while True:
                try:
                    start_idx = raw_buffer.find(_MSG_START)
                    if start_idx < 0:
                        break
                    end_idx = raw_buffer.find(_MSG_END, start_idx + len(_MSG_START))
                    if end_idx < 0:
                        break
                    payload = bytes(raw_buffer[start_idx + len(_MSG_START) : end_idx])
                    del raw_buffer[start_idx : end_idx + len(_MSG_END)]
                    for message in _iter_complete_messages(payload.decode("ascii", errors="ignore"))[0]:
                        _update_protocol_state(self._protocol_state, message)
                except Exception:
                    break

            while len(raw_buffer) >= 2:
                start_idx = next((i for i, b in enumerate(raw_buffer) if b & 0x80), -1)
                if start_idx < 0:
                    raw_buffer.clear()
                    break
                if start_idx > 0:
                    del raw_buffer[:start_idx]
                if len(raw_buffer) < frame_size:
                    break

                for ch_idx in range(stream_channels):
                    hi = raw_buffer[2 * ch_idx]
                    lo = raw_buffer[2 * ch_idx + 1]
                    raw_val = ((hi & 0x7F) << 7) | (lo & 0x7F)
                    sample_buffer[sample_idx, ch_idx] = (raw_val - center_val) / center_val

                del raw_buffer[:frame_size]
                sample_idx += 1
                if sample_idx >= raw_chunk_size:
                    self._emit_active(sample_buffer, sample_idx)
                    sample_buffer.fill(0.0)
                    sample_idx = 0
                    last_emit_time = time.monotonic()
                    if sync_lost:
                        _LOGGER.info(
                            "BYB frame sync recovered after %.1f s.",
                            last_emit_time - sync_lost_time,
                        )
                        sync_lost = False

            elapsed = time.monotonic() - last_emit_time
            if elapsed > self._SYNC_TIMEOUT_S and not sync_lost:
                sync_lost = True
                sync_lost_time = time.monotonic()
                _LOGGER.warning(
                    "BYB frame sync lost: no valid frames for %.1f s. "
                    "profile=%s board=%s decode_widths=%s current_width=%s. "
                    "Resetting decoder; check device mode, USB cable, and electrode connection.",
                    elapsed,
                    None if self._profile is None else self._profile.key,
                    self._protocol_state.board_type,
                    self._decoder_candidate_widths,
                    self._stream_channel_count,
                )
                self.note_xrun()
                raw_buffer.clear()
                sample_idx = 0

    def _emit_active(self, sample_buffer: np.ndarray, frames: int) -> None:
        with self._channel_lock:
            active_ids = list(self._active_channel_ids)
            avail_map = {ch.id: idx for idx, ch in enumerate(self._available_channels)}

        if not active_ids or frames <= 0:
            return

        out = np.zeros((frames, len(active_ids)), dtype=np.float32)
        for out_idx, channel_id in enumerate(active_ids):
            src_idx = avail_map.get(channel_id)
            if src_idx is None or src_idx >= sample_buffer.shape[1]:
                continue
            out[:, out_idx] = sample_buffer[:frames, src_idx]

        self.emit_array(out, mono_time=time.monotonic())

    def stats(self) -> dict[str, Any]:
        stats = super().stats()
        stats.update(
            {
                "transport": self._transport_kind,
                "baudrate": self._baudrate,
                "profile": None if self._profile is None else self._profile.key,
                "profile_label": None if self._profile is None else self._profile.label,
                "fallback_identification": self._fallback_identification,
                "stream_channels": self._stream_channel_count,
                "bits": self._bits,
                "hardware_type": self._protocol_state.hardware_type,
                "hardware_version": self._protocol_state.hardware_version,
                "firmware_version": self._protocol_state.firmware_version,
                "board_type": self._protocol_state.board_type,
                "last_event": self._protocol_state.last_event,
                "joy_state": self._protocol_state.joy_state,
            }
        )
        return stats


__all__ = [
    "BackyardBrainsSource",
    "_BYBDecoder",
    "_BYBProtocolState",
    "_HIDTransport",
    "_ProfileHint",
    "_PROFILES",
    "_candidate_profile_hints",
    "_resolve_profile",
    "_update_protocol_state",
]
