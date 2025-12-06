# daq/soundcard_source.py
from __future__ import annotations

import logging
import threading
import numpy as np
import time as _time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, Any

logger = logging.getLogger(__name__)

try:
    import miniaudio
except ImportError as e:  # pragma: no cover
    miniaudio = None
    _IMPORT_ERROR = e

from .base_device import (
    BaseDevice,
    Chunk,
    DeviceInfo,
    ChannelInfo,
    Capabilities,
    ActualConfig,
)


class LocalRingBuffer:
    """
    Simple circular buffer for accumulating audio frames without reallocation.
    Optimized for (frames, channels) layout.
    """
    def __init__(self, capacity: int, channels: int, dtype: np.dtype):
        self.capacity = capacity
        self.channels = channels
        self._data = np.zeros((capacity, channels), dtype=dtype)
        self._write_pos = 0
        self._filled = 0

    def write(self, data: np.ndarray) -> None:
        frames = data.shape[0]
        if frames == 0:
            return
        
        # Safety check for massive chunks
        if frames > self.capacity:
            # Reset and take only the latest
            self._write_pos = 0
            self._filled = 0
            data = data[-self.capacity:]
            frames = self.capacity

        # Check for overflow - in this context, we just cap filled at capacity
        # (assuming we are overwriting old unread data, which is bad but better than crashing)
        
        idx = self._write_pos
        end = idx + frames
        if end <= self.capacity:
            self._data[idx:end] = data
        else:
            split = self.capacity - idx
            self._data[idx:] = data[:split]
            self._data[:end-self.capacity] = data[split:]
        
        self._write_pos = (self._write_pos + frames) % self.capacity
        self._filled = min(self.capacity, self._filled + frames)

    @property
    def filled(self) -> int:
        return self._filled

    def read(self, frames: int) -> np.ndarray:
        if frames > self._filled:
            raise ValueError("Not enough data")
        
        # Calculate read position (tail)
        # write_pos points to next write. 
        # So tail is (write_pos - filled) % capacity
        read_pos = (self._write_pos - self._filled) % self.capacity
        
        idx = read_pos
        end = idx + frames
        if end <= self.capacity:
            out = self._data[idx:end].copy()
        else:
            split = self.capacity - idx
            out = np.vstack((self._data[idx:], self._data[:end-self.capacity]))
        
        self._filled -= frames
        return out
    
    def clear(self) -> None:
        self._write_pos = 0
        self._filled = 0


class SoundCardSource(BaseDevice):
    """
    Audio input DAQ using `miniaudio`.

    Notes
    -----
    • Driver callbacks deliver arrays shaped (frames, channels); downstream consumers
      receive `Chunk.samples` shaped (channels, frames).
    • Channel identifiers are simple strings: "In 1", "In 2", ... (1‑based for readability).
    """

    @classmethod
    def device_class_name(cls) -> str:
        return "Sound Card"

    # Toggle to control whether we list all input-capable devices or just the system default.
    _LIST_ALL_DEVICES: bool = False

    @classmethod
    def set_list_all_devices(cls, enabled: bool) -> None:
        cls._LIST_ALL_DEVICES = bool(enabled)

    @classmethod
    def _list_all_devices(cls) -> bool:
        return cls._LIST_ALL_DEVICES

    # ---------- Discovery helpers ---------------------------------------------

    @classmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        """Return input-capable audio devices as generic DeviceInfo objects."""
        if miniaudio is None:
            raise RuntimeError(
                f"`miniaudio` is not available: {_IMPORT_ERROR!r}"
            )
        
        # If not listing all, return a single "System Default" entry
        if not cls._list_all_devices():
            return [
                DeviceInfo(
                    id="default",
                    name="System Default",
                    details={
                        "channels": 2, # Assume stereo for default
                        "sample_rate": 44100,
                        "real_id": None # None means default in miniaudio
                    },
                )
            ]
        
        try:
            devices = miniaudio.Devices().get_captures()
        except Exception as exc:
            logger.warning("Failed to get capture devices: %s", exc)
            return []
            
        out: List[DeviceInfo] = []
        
        for idx, dev in enumerate(devices):
            # dev is a dict: {'name': str, 'id': cdata, 'type': enum, 'formats': list}
            
            # Get max channels from formats
            max_channels = 0
            default_rate = 44100
            if 'formats' in dev and dev['formats']:
                # Find max channels across formats
                for fmt in dev['formats']:
                    max_channels = max(max_channels, fmt.get('channels', 0))
                default_rate = dev['formats'][0].get('samplerate', 44100)
            
            out.append(
                DeviceInfo(
                    id=str(idx), # Use index as stable-ish ID for this session
                    name=dev['name'],
                    details={
                        "channels": max_channels,
                        "sample_rate": default_rate,
                        "real_id": dev['id'] # Store real ID here
                    },
                )
            )
                
        return out

    @staticmethod
    def supported_sample_rates(
        device_index: int,
        probe: Optional[Sequence[int]] = None,
        min_channels: int = 1,
    ) -> List[int]:
        """
        Miniaudio supports high-quality resampling, so effectively all standard rates are supported.
        """
        return [44100, 48000, 88200, 96000]

    # ---------- Instance configuration ----------------------------------------

    def __init__(self, queue_maxsize: int = 64, dtype: str = "float32"):
        if miniaudio is None:
            pass
        super().__init__(queue_maxsize=queue_maxsize)
        self.dtype = dtype
        self._miniaudio_device_id: Any = None # The miniaudio device ID object
        self._n_in: int = 0
        self._chan_names: List[str] = []
        self._buf_lock = threading.Lock()
        self._residual_buffer: Optional[LocalRingBuffer] = None
        self._device: Optional[miniaudio.CaptureDevice] = None
        self._stop_evt = threading.Event()

    # ---------- Required interface --------------------------------------------

    def get_capabilities(self, device_id: str) -> Capabilities:
        if miniaudio is None:
            raise RuntimeError(f"`miniaudio` unavailable: {_IMPORT_ERROR!r}")
        
        # Handle default device
        if device_id == "default":
            return Capabilities(max_channels_in=2, sample_rates=[44100, 48000, 88200, 96000], dtype=self.dtype)
        
        # Resolve device info
        idx = int(device_id)
        captures = miniaudio.Devices().get_captures()
        if idx < 0 or idx >= len(captures):
             raise ValueError(f"Invalid device ID: {device_id}")
        
        dev = captures[idx]
        max_in = 0
        if 'formats' in dev:
            for fmt in dev['formats']:
                max_in = max(max_in, fmt.get('channels', 0))
        
        # Miniaudio handles resampling, so we can support common rates
        supported = [44100, 48000, 88200, 96000]
        
        return Capabilities(max_channels_in=max_in, sample_rates=supported, dtype=self.dtype)

    def list_available_channels(self, device_id: str) -> List[ChannelInfo]:
        if miniaudio is None:
            raise RuntimeError(f"`miniaudio` unavailable: {_IMPORT_ERROR!r}")
            
        # Handle default device
        if device_id == "default":
            return [ChannelInfo(id=i, name=f"In {i+1}", units="V") for i in range(2)]
            
        idx = int(device_id)
        captures = miniaudio.Devices().get_captures()
        if idx < 0 or idx >= len(captures):
             raise ValueError(f"Invalid device ID: {device_id}")
             
        dev = captures[idx]
        max_in = 0
        if 'formats' in dev:
            for fmt in dev['formats']:
                max_in = max(max_in, fmt.get('channels', 0))
        return [ChannelInfo(id=i, name=f"In {i+1}", units="V") for i in range(max_in)]

    # ---------- Lifecycle ------------------------------------------------------

    def _open_impl(self, device_id: str) -> None:
        if miniaudio is None:
            raise RuntimeError(f"`miniaudio` unavailable: {_IMPORT_ERROR!r}")
            
        if device_id == "default":
            self._miniaudio_device_id = None # None means default in miniaudio
            self._n_in = 2 # Assume stereo
            self._chan_names = ["In 1", "In 2"]
            return
            
        idx = int(device_id)
        captures = miniaudio.Devices().get_captures()
        if idx < 0 or idx >= len(captures):
             raise ValueError(f"Invalid device ID: {device_id}")
        
        dev = captures[idx]
        self._miniaudio_device_id = dev['id']
        
        max_in = 0
        if 'formats' in dev:
            for fmt in dev['formats']:
                max_in = max(max_in, fmt.get('channels', 0))
        self._n_in = max_in
        self._chan_names = [f"In {i+1}" for i in range(self._n_in)]

    def _close_impl(self) -> None:
        pass

    def _configure_impl(self, sample_rate: int, channels: Sequence[int], chunk_size: int, **options) -> ActualConfig:
        if miniaudio is None:
            raise RuntimeError(f"`miniaudio` unavailable: {_IMPORT_ERROR!r}")

        if self._miniaudio_device_id is None:
            # Should check if we have a device ID, but it might be None for default device?
            # Actually _open_impl sets it.
            pass

        # Allocate residual buffer (1 second capacity to be safe)
        capacity = max(int(sample_rate), int(chunk_size * 4))
        self._residual_buffer = LocalRingBuffer(capacity, self._n_in, np.dtype(self.dtype))
        
        id_to_info = {ch.id: ch for ch in self._available_channels}
        selected = [id_to_info[c] for c in channels]
        cfg = ActualConfig(sample_rate=sample_rate, channels=selected, chunk_size=chunk_size, dtype=self.dtype)
        return cfg

    def _start_impl(self) -> None:
        if self._device is not None:
            return

        self._stop_evt.clear()

        # Miniaudio capture callback generator
        def capture_generator():
            while True:
                # Yield nothing to receive data
                data_bytes = yield 
                
                # Convert bytes to numpy array
                # Assuming float32 because we request it
                data = np.frombuffer(data_bytes, dtype=np.float32)
                
                # Reshape to (frames, channels)
                # Miniaudio delivers interleaved data
                frames = len(data) // self._n_in
                if frames > 0:
                    data = data.reshape((frames, self._n_in))
                    
                    with self._buf_lock:
                        if self._residual_buffer is None:
                            continue
                        
                        self._residual_buffer.write(data)
                        
                        while self._residual_buffer.filled >= self.config.chunk_size:
                            data_chunk = self._residual_buffer.read(self.config.chunk_size)
                            
                            idxs = [c.id for c in self.get_active_channels()]
                            if not idxs:
                                continue
                            data_chunk = data_chunk[:, idxs]
                            # Emit via base to stamp counters
                            self.emit_array(data_chunk, mono_time=_time.monotonic())

        try:
            # We request FLOAT32 to match our pipeline
            self._device = miniaudio.CaptureDevice(
                device_id=self._miniaudio_device_id,
                nchannels=self._n_in,
                sample_rate=self.config.sample_rate,
                input_format=miniaudio.SampleFormat.FLOAT32,
                buffersize_msec=100 # Reasonable buffer for capture
            )
            
            gen = capture_generator()
            next(gen) # Prime it
            self._device.start(gen)
            
        except Exception as e:
            print(f"Error starting miniaudio capture: {e}")
            self._device = None

    def _stop_impl(self) -> None:
        self._stop_evt.set()
        if self._device and self._device.running:
            self._device.stop()
            self._device.close()
        self._device = None
        
        with self._buf_lock:
            if self._residual_buffer is not None:
                self._residual_buffer.clear()

    # ---------- Internals ------------------------------------------------------
    # No longer needed as we use index-based resolution
