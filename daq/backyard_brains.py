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

from .base_source import ActualConfig, BaseSource, Capabilities, ChannelInfo, Chunk, DeviceInfo

_LOGGER = logging.getLogger(__name__)

# Known Backyard Brains VIDs from documentation
_BYB_VIDS = {
    0x2E73,  # Backyard Brains
    0x0403,  # FTDI (older devices)
    0x2341,  # Arduino (SpikerShields)
}

# Escape sequences for custom messages
_MSG_START = bytes([0xFF, 0xFF, 0x01, 0x01, 0x80, 0xFF])
_MSG_END = bytes([0xFF, 0xFF, 0x01, 0x01, 0x81, 0xFF])


class BackyardBrainsSource(BaseSource):
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
        # Default capabilities
        self._max_channels = 6  # Max supported by protocol (e.g. Muscle SpikerShield Pro)

    @classmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        if serial is None:
            return []
        
        candidates = []
        try:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                # Loose matching: check VID or "SpikerBox" in description
                is_byb = (p.vid in _BYB_VIDS) or ("SpikerBox" in str(p.description))
                if is_byb:
                    candidates.append(
                        DeviceInfo(
                            id=p.device,
                            name=f"{p.description} ({p.device})",
                            vendor="Backyard Brains" if p.vid == 0x2E73 else "Generic/FTDI",
                            details={"vid": p.vid, "pid": p.pid, "hwid": p.hwid},
                        )
                    )
        except Exception as e:
            _LOGGER.error("Failed to scan serial ports: %s", e)
        
        return candidates

    def get_capabilities(self, device_id: str) -> Capabilities:
        # Most BYB devices support up to 10kHz.
        # We don't know the exact model yet without opening it, so we report generic specs.
        return Capabilities(
            max_channels_in=self._max_channels,
            sample_rates=[1000, 2500, 3333, 5000, 10000],
            dtype="float32",
            notes="Serial/CDC SpikerBox. 10-14bit resolution.",
        )

    def list_available_channels(self, device_id: str) -> List[ChannelInfo]:
        # We'll expose the max likely channels. 
        # The user configures how many they actually want to read.
        return [
            ChannelInfo(id=i, name=f"Channel {i+1}", units="V", range=(-5.0, 5.0))
            for i in range(self._max_channels)
        ]

    def _open_impl(self, device_id: str) -> None:
        if serial is None:
            raise RuntimeError("pyserial module is not installed.")
        
        # 222222 is a common baud rate for BYB, though CDC usually ignores it.
        # 500000 is used by newer Human Human Interface.
        try:
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

        # The BYB protocol often defines sample rate implicitly by channel count 
        # (e.g. 10kHz total bandwidth split by channels), OR by specific commands.
        # We'll try to set the number of channels which is the primary config.
        
        num_channels = len(channels)
        # Ensure channels are contiguous 0..N for this simple driver, or we map them.
        # The protocol just streams N channels. We assume the user wants the first N.
        # If they selected [0, 2], we effectively have to ask for 3 channels to get index 2.
        max_ch_idx = max(channels) if channels else 0
        req_channels = max_ch_idx + 1
        
        # Send configuration commands
        # Stop first to be safe
        self._send_command("h:;") 
        time.sleep(0.05)
        
        # Set number of channels
        self._send_command(f"c:{req_channels};")
        time.sleep(0.05)
        
        # Some devices support sample rate config, others derive it.
        # We'll assume the device gives us roughly 10k / num_channels or fixed 10k.
        # We just echo back what the user asked for, but in reality, 
        # we will timestamp chunks based on arrival if we can't trust rate.
        # For now, let's trust the user or default to 10000.
        
        actual_rate = sample_rate
        
        # Prepare channel infos
        all_chans = self.list_available_channels(self._device_id)
        selected_chans = [all_chans[i] for i in channels if i < len(all_chans)]

        return ActualConfig(
            sample_rate=actual_rate,
            channels=selected_chans,
            chunk_size=chunk_size,
            dtype="float32",
        )

    def _start_impl(self) -> None:
        if not self._ser:
            raise RuntimeError("Serial port not open")
        
        # Send start command
        self._send_command("start:;")
        
        self._producer_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._producer_thread.start()

    def _stop_impl(self) -> None:
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1.0)
        
        # Send halt command
        if self._ser and self._ser.is_open:
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

        num_channels = len(self._active_channel_ids)
        # We might be capturing more channels than active if the user skipped some,
        # but for simplicity, we assume 1:1 mapping to the first N channels here.
        # Protocol frame size: 2 bytes per channel.
        frame_size = num_channels * 2
        
        chunk_size = self.config.chunk_size
        
        # Buffers
        raw_buffer = bytearray()
        sample_buffer = np.zeros((chunk_size, num_channels), dtype=np.float32)
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

            # 2. Process Frames
            # We need at least frame_size bytes
            while len(raw_buffer) >= frame_size:
                # Search for frame alignment
                # Bit 7 (MSB) of the *first byte of the frame* must be 1.
                # Bit 7 of all other bytes in the frame must be 0.
                
                # Peek at the first byte
                if not (raw_buffer[0] & 0x80):
                    # Misaligned or garbage, pop one byte and retry
                    del raw_buffer[0]
                    continue
                
                # Check if we have enough data for a full frame
                if len(raw_buffer) < frame_size:
                    break
                
                # Validate the rest of the bytes have MSB=0
                valid_frame = True
                for i in range(1, frame_size):
                    if raw_buffer[i] & 0x80:
                        # Found a '1' in a position that should be '0'. 
                        # This implies the frame boundary is actually at 'i'.
                        # Discard up to 'i' and restart scan.
                        del raw_buffer[:i]
                        valid_frame = False
                        break
                
                if not valid_frame:
                    continue
                
                # Decode Frame
                # Each channel is 2 bytes. 
                # Byte A: 1xxxxxxx (or 0 for ch>0) -> 7 bits
                # Byte B: 0xxxxxxx -> 7 bits
                # Value = (A & 0x7F) << 7 | (B & 0x7F)
                
                for ch_idx in range(num_channels):
                    high_byte = raw_buffer[ch_idx * 2]
                    low_byte = raw_buffer[ch_idx * 2 + 1]
                    
                    raw_val = ((high_byte & 0x7F) << 7) | (low_byte & 0x7F)
                    
                    # Map 0..16383 (14-bit) to -1.0 .. 1.0 V (approx)
                    # Center is roughly 8192
                    # BYB typically uses 5V range or 3.3V range. 
                    # We'll normalize to a generic +/- 0.5 scale for now (1V pp)
                    # Users can adjust gain in UI.
                    
                    norm_val = (raw_val - 8192) / 8192.0
                    sample_buffer[sample_idx, ch_idx] = norm_val
                
                # Remove used frame
                del raw_buffer[:frame_size]
                
                sample_idx += 1
                
                # Emit chunk if full
                if sample_idx >= chunk_size:
                    self.emit_array(sample_buffer, mono_time=time.monotonic())
                    sample_buffer.fill(0.0)
                    sample_idx = 0

