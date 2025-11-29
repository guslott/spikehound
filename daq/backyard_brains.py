from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import List, Optional, Sequence

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

from daq.base_device import ActualConfig, BaseDevice, Capabilities, ChannelInfo, Chunk, DeviceInfo

_LOGGER = logging.getLogger(__name__)

# Known Backyard Brains VIDs from documentation
_BYB_VIDS = {
    0x2E73,  # Backyard Brains
    0x0403,  # FTDI (older devices and vendor specific)
    0x2341,  # Arduino (SpikerShields)
}

# Heuristics for common BYB devices keyed by (vid, pid)
_DEVICE_HINTS = {
    (0x2E73, 0x0001): {"label": "Muscle SpikerBox Pro (HID)", "channels": 4, "bits": 10, "start": True, "set_channels": False},
    (0x2E73, 0x0002): {"label": "Neuron SpikerBox Pro (HID)", "channels": 4, "bits": 10, "start": True, "set_channels": False},
    (0x2E73, 0x0004): {"label": "Human SpikerBox", "channels": 4, "bits": 14, "start": False, "set_channels": False},
    (0x2E73, 0x0006): {"label": "Muscle SpikerBox Pro (Serial)", "channels": 4, "bits": 10, "start": True, "set_channels": False, "sample_rate": 10000},
    (0x2E73, 0x0007): {"label": "Neuron SpikerBox Pro (Serial)", "channels": 4, "bits": 10, "start": True, "set_channels": False, "sample_rate": 10000},
    # Neuron SpikerBox Pro (Serial + MFi) - PID 0x0009
    # NOTE: Official docs say "2 channels", but the firmware streams 4 channels @ 10kHz (80kB/s).
    # This is due to MFi requirements for fixed packet sizes; the device always streams
    # the 2 main channels + 2 expansion port analog inputs (even if unused).
    # We must parse 4 channels to correctly decode the 10kHz frame rate.
    (0x2E73, 0x0009): {"label": "Neuron SpikerBox Pro (Serial + MFi)", "channels": 4, "bits": 14, "start": True, "set_channels": False, "sample_rate": 10000},
    (0x2E73, 0x000D): {"label": "Spike Station", "channels": 2, "bits": 14, "start": True, "set_channels": False, "sample_rate": 42661},
    (0x2341, 0x8036): {"label": "Plant SpikerBox", "channels": 1, "bits": 10, "start": False, "set_channels": False, "sample_rate": 10000},
    (0x2341, 0x0043): {"label": "SpikerShield (Arduino)", "channels": 6, "bits": 10, "start": False, "set_channels": True, "sample_rate": 10000},
    (0x0403, 0x6015): {"label": "FTDI-based SpikerBox", "channels": 1, "bits": 10, "start": False, "set_channels": False, "sample_rate": 10000},
}

# Escape sequences for custom messages
_MSG_START = bytes([0xFF, 0xFF, 0x01, 0x01, 0x80, 0xFF])
_MSG_END = bytes([0xFF, 0xFF, 0x01, 0x01, 0x81, 0xFF])


class BackyardBrainsSource(BaseDevice):
    """
    Driver for Backyard Brains SpikerBox devices (Neuron, Muscle, Plant, etc.)
    communicating via USB Serial (CDC).

    Implements the custom BYB binary protocol:
    - 14-bit (or 10-bit) samples packed into 2 bytes.
    - MSB (bit 7) is 1 for the first byte of a frame, 0 otherwise.
    - Embedded ASCII messages wrapped in escape sequences.
    """

    @classmethod
    def device_class_name(cls) -> str:
        return "Backyard Brains"

    def __init__(self, queue_maxsize: int = 64) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        if serial is None:
            # We allow instantiation so discovery can report the missing dependency error
            pass
        self._ser: Optional[serial.Serial] = None
        self._producer_thread: Optional[threading.Thread] = None
        self._buffer = bytearray()
        self._bits = 10
        self._requires_start = True
        self._supports_channel_cfg = False
        self._hint_sample_rate: Optional[int] = None
        self._hint_sample_rate: Optional[int] = None
        self._stream_channel_count = 1
        # Default capabilities fallback (updated on open)
        self._max_channels = 2

    @classmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        if serial is None:
            raise RuntimeError("pyserial is not installed; Backyard Brains devices require pyserial to enumerate over USB CDC.")
        
        candidates = []
        try:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                # Loose matching: check VID or "SpikerBox" in description
                is_byb = (p.vid in _BYB_VIDS) or ("SpikerBox" in str(p.description))
                if not is_byb:
                    continue

                hint = _DEVICE_HINTS.get((p.vid, p.pid))
                name = hint.get("label") if hint else None
                if not name:
                    name = p.description or "Backyard Brains device"
                name = f"{name} ({p.device})"
                candidates.append(
                    DeviceInfo(
                        id=p.device,
                        name=name,
                        vendor="Backyard Brains" if p.vid == 0x2E73 else "Generic/FTDI/Arduino",
                        details={"vid": p.vid, "pid": p.pid, "hwid": p.hwid},
                    )
                )
        except Exception as e:
            _LOGGER.error("Failed to scan serial ports: %s", e)
        
        return candidates

    def get_capabilities(self, device_id: str) -> Capabilities:
        # Resolve hints for this device_id if we haven't already (e.g. before open)
        hint_rate = self._hint_sample_rate
        supports_cfg = self._supports_channel_cfg
        
        if hint_rate is None:
            port_info = self._lookup_port(device_id)
            if port_info:
                hint = _DEVICE_HINTS.get((port_info.vid, port_info.pid))
                if hint:
                    hint_rate = int(hint["sample_rate"]) if "sample_rate" in hint else None
                    supports_cfg = bool(hint.get("set_channels", False))

        # Most BYB devices support up to 10kHz. Use hints if we have them.
        # If a device has a specific sample rate hint AND does not support configuration,
        # we treat it as a fixed-rate device.
        if hint_rate and not supports_cfg:
            # Fixed rate device - ONLY support the native rate
            sample_rates = [hint_rate]
        else:
            sample_rates = [1000, 2500, 3333, 5000, 10000]
            if hint_rate:
                sample_rates = sorted(set(sample_rates + [hint_rate]))
        
        notes = f"Serial/CDC SpikerBox. {self._bits}-bit resolution expected."
        return Capabilities(
            max_channels_in=self._max_channels,
            sample_rates=sample_rates,
            dtype="float32",
            notes=notes,
        )

    def list_available_channels(self, device_id: str) -> List[ChannelInfo]:
        # We'll expose the max likely channels. 
        # The user configures how many they actually want to read.
        return [
            ChannelInfo(id=i, name=f"Channel {i+1}", units="V", range=(-5.0, 5.0))
            for i in range(self._max_channels)
        ]

    def _lookup_port(self, device_id: str):
        if serial is None:
            return None
        try:
            for p in serial.tools.list_ports.comports():
                if p.device == device_id:
                    return p
        except Exception:
            return None
        return None

    def _apply_port_hints(self, port_info) -> None:
        """
        Configure driver expectations (channels, bit depth, start command)
        based on VID/PID hints from enumeration.
        """
        if port_info is None:
            return
        hint = _DEVICE_HINTS.get((port_info.vid, port_info.pid))
        if hint:
            self._max_channels = max(1, int(hint.get("channels", self._max_channels)))
            self._bits = int(hint.get("bits", self._bits))
            self._requires_start = bool(hint.get("start", self._requires_start))
            self._supports_channel_cfg = bool(hint.get("set_channels", False))
            self._hint_sample_rate = int(hint["sample_rate"]) if "sample_rate" in hint else None
            
            _LOGGER.info(f"BYB Hint: {hint.get('label')}, max_ch={self._max_channels}, supp_cfg={self._supports_channel_cfg}, rate={self._hint_sample_rate}")
        else:
            # Unknown device: keep conservative defaults
            self._max_channels = max(1, 2)
            self._bits = 10
            self._requires_start = True
            self._supports_channel_cfg = False
            self._hint_sample_rate = None
            _LOGGER.info(f"BYB Hint: None (Unknown device) VID={port_info.vid} PID={port_info.pid}")

    def _open_impl(self, device_id: str) -> None:
        if serial is None:
            raise RuntimeError("pyserial module is not installed.")
        
        # 222222 is a common baud rate for BYB, though CDC usually ignores it.
        # 500000 is used by newer Human Human Interface.
        try:
            port_info = self._lookup_port(device_id)
            self._apply_port_hints(port_info)
            if port_info:
                _LOGGER.info(
                    "Opening BYB device %s (vid=0x%04X pid=0x%04X, channels=%d, bits=%d, start_cmd=%s)",
                    device_id,
                    port_info.vid or 0,
                    port_info.pid or 0,
                    self._max_channels,
                    self._bits,
                    self._requires_start,
                )
            self._ser = serial.Serial(
                port=device_id,
                baudrate=222222,
                timeout=0.5,
                write_timeout=0.5
            )
            # Reset buffer
            self._ser.reset_input_buffer()
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {device_id}: {e}")

    def _close_impl(self) -> None:
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _configure_impl(
        self, sample_rate: int, channels: Sequence[int], chunk_size: int, **options
    ) -> ActualConfig:
        if not self._ser:
            raise RuntimeError("Device is not open.")

        # Bound channels to what the hardware can actually emit
        all_chans = self.list_available_channels(self._device_id)
        selected_chans = [all_chans[i] for i in channels if i < len(all_chans)]
        if not selected_chans:
            selected_chans = [all_chans[0]]
        num_channels = len(selected_chans)

        # Some devices allow channel-count configuration (SpikerShield family).
        # Others stream a fixed set; in those cases we simply decode that many.
        # Some devices allow channel-count configuration (SpikerShield family).
        # Others stream a fixed set; in those cases we simply decode that many.
        if self._supports_channel_cfg:
            # Stop first to be safe on devices that honor start/stop
            if self._requires_start:
                self._send_command("h:;")
                time.sleep(0.05)
            req_channels = num_channels
            self._send_command(f"c:{req_channels};")
            time.sleep(0.05)
            self._stream_channel_count = req_channels
        else:
            self._stream_channel_count = self._max_channels
            
        _LOGGER.info(f"BYB Config Logic: supports_cfg={self._supports_channel_cfg}, stream_ch={self._stream_channel_count}, max_ch={self._max_channels}")

        # If the device has a fixed sample rate hint, we generally prefer it,
        # BUT if the user requested a specific rate that is also valid (e.g. in capabilities),
        # we should respect it.
        # If sample_rate is 0 (unspecified), default to 1000 or the hint.
        req_rate = sample_rate if sample_rate and sample_rate > 0 else 1000
        
        native_rate = self._hint_sample_rate or 10000
        
        # We only support native rate now
        actual_rate = native_rate

        _LOGGER.info(f"BYB Config: req={req_rate}, native={native_rate}, actual={actual_rate}, stream_ch={self._stream_channel_count}")

        return ActualConfig(
            sample_rate=actual_rate,
            channels=selected_chans,
            chunk_size=chunk_size,
            dtype="float32",
        )

    def _start_impl(self) -> None:
        if not self._ser:
            raise RuntimeError("Serial port not open")

        if self._requires_start:
            self._send_command("start:;")
            time.sleep(0.02)

        self._producer_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._producer_thread.start()

    def _stop_impl(self) -> None:
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1.0)
        
        # Send halt command
        if self._requires_start and self._ser and self._ser.is_open:
            try:
                self._send_command("h:;")
            except Exception:
                pass

    def _send_command(self, cmd: str) -> None:
        if self._ser:
            self._ser.write(cmd.encode("ascii"))

    def _run_loop(self) -> None:
        if not self._ser or not self.config:
            return

        # Always decode the full stream channel count (hardware order).
        # Always decode the full stream channel count (hardware order).
        stream_channels = self._stream_channel_count
        stream_channels = max(1, stream_channels)
        bits = max(8, min(self._bits, 16))
        center_val = float(1 << (bits - 1))
        
        # We want to emit `chunk_size` samples.
        # We need to collect `chunk_size` raw samples.
        target_chunk_size = self.config.chunk_size
        raw_chunk_size = target_chunk_size
        
        # Buffers
        raw_buffer = bytearray()
        # Buffer for unpacked raw samples
        sample_buffer = np.zeros((raw_chunk_size, stream_channels), dtype=np.float32)
        sample_idx = 0
        
        # For un-escaping logic
        msg_start_seq = list(_MSG_START)
        msg_end_seq = list(_MSG_END)

        while not self.stop_event.is_set():
            try:
                # Read available bytes
                if self._ser.in_waiting:
                    data = self._ser.read(self._ser.in_waiting)
                    raw_buffer.extend(data)
                else:
                    # Sleep briefly to yield if no data
                    time.sleep(0.001)
                    continue
            except Exception as e:
                _LOGGER.error("Serial read error: %s", e)
                break

            # 1. Process Escape Sequences (Messages)
            # Simple scan: we look for start sequence.
            # If found, we look for end sequence.
            # If found, we extract message and remove from buffer.
            
            while True:
                try:
                    start_idx = raw_buffer.find(_MSG_START)
                    if start_idx >= 0:
                        end_idx = raw_buffer.find(_MSG_END, start_idx)
                        if end_idx != -1:
                            # Message found
                            msg_bytes = raw_buffer[start_idx + len(_MSG_START) : end_idx]
                            try:
                                msg_str = msg_bytes.decode("ascii", errors="ignore")
                                # TODO: emit event or log message
                                # e.g. "EVNT: 4;"
                                if msg_str:
                                    _LOGGER.info("SpikerBox Message: %s", msg_str)
                            except Exception:
                                pass
                            
                            # Remove message from buffer
                            del raw_buffer[start_idx : end_idx + len(_MSG_END)]
                            continue # Check for more messages
                except Exception:
                    pass
                break

            # 2. Process frames sample-wise.
            # We align to the next byte with MSB=1 (frame start), then read
            # expected_channels samples of two bytes each (high, low). If only the
            # first sample is flagged, we still read the following pairs; if each
            # channel is flagged, we tolerate MSB=1 on subsequent highs.
            frame_size = stream_channels * 2
            while len(raw_buffer) >= 2:
                # Align to a start marker
                start_idx = next((i for i, b in enumerate(raw_buffer) if b & 0x80), -1)
                if start_idx < 0:
                    raw_buffer.clear()
                    break
                if start_idx > 0:
                    del raw_buffer[:start_idx]
                    if len(raw_buffer) < 2:
                        break

                if len(raw_buffer) < frame_size:
                    break

                # Decode expected_channels pairs
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

    def _emit_active(self, sample_buffer: np.ndarray, frames: int) -> None:
        """
        Emit a chunk mapped to the current active channel order.
        """
        with self._channel_lock:
            active_ids = list(self._active_channel_ids)
            avail_map = {ch.id: idx for idx, ch in enumerate(self._available_channels)}

        if not active_ids or frames <= 0:
            return

        out = np.zeros((frames, len(active_ids)), dtype=np.float32)
        for out_idx, cid in enumerate(active_ids):
            src_idx = avail_map.get(cid)
            if src_idx is not None and src_idx < sample_buffer.shape[1]:
                out[:, out_idx] = sample_buffer[:frames, src_idx]

        # _LOGGER.info(f"Emit: frames={frames}, active={len(active_ids)}, shape={out.shape}")
        meta = {"active_channel_ids": active_ids}
        self.emit_array(out, mono_time=time.monotonic(), meta=meta)
