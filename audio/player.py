from __future__ import annotations

import logging
import threading
import queue
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import miniaudio
except ImportError as e:  # pragma: no cover
    miniaudio = None
    _IMPORT_ERR = e

from shared.models import ChunkPointer
from shared.ring_buffer import SharedRingBuffer


@dataclass
class AudioConfig:
    out_samplerate: int = 44_100   # soundcard output Hz
    out_channels: int = 1          # 1=mono (start simple)
    device: Any = None             # None = default output device
    gain: float = 0.35             # careful: spikes can be loud
    blocksize: int = 256           # miniaudio buffer size (frames)
    ring_seconds: float = 0.2      # size of the output ring buffer (seconds)


def list_output_devices(list_all: bool = False) -> List[Dict[str, object]]:
    """Return a list of available output devices using miniaudio."""
    if miniaudio is None:
        return []
    devices: List[Dict[str, object]] = []
    try:
        playback_devices = miniaudio.Devices().get_playbacks()
        for idx, dev in enumerate(playback_devices):
            # Handle both object attributes and dict access (miniaudio version differences)
            if isinstance(dev, dict):
                dev_id = dev.get("id", idx)
                dev_name = dev.get("name", f"Device {idx}")
            else:
                dev_id = getattr(dev, "id", idx)
                dev_name = getattr(dev, "name", f"Device {idx}")
            
            devices.append({"id": dev_id, "label": dev_name, "name": dev_name})
            
            if not list_all:
                # Just return the first one (default)
                break
    except Exception as exc:
        logger.warning("Failed to list output devices: %s", exc)
        return []
    return devices



class AudioPlayer(threading.Thread):
    """
    Consumes Chunk/ChunkPointer-like objects from an audio_queue. Each item must
    resolve to samples shaped (frames, channels) at input_sr. We select one
    channel and play via miniaudio.PlaybackDevice.
    
    OPTIMIZATION: miniaudio handles resampling from input_sr to the hardware
    rate in C, which is significantly faster and lower latency than Python.
    """

    def __init__(
        self,
        audio_queue: "queue.Queue",
        *,
        input_samplerate: int,
        config: AudioConfig = AudioConfig(),
        selected_channel: Optional[int] = 0,  # 0 by default; None = mute
        ring_buffer: Optional[SharedRingBuffer] = None,
    ) -> None:
        super().__init__(name="AudioPlayer", daemon=True)
        if miniaudio is None:
            raise RuntimeError(f"`miniaudio` is not available: {_IMPORT_ERR!r}")

        self.q = audio_queue
        self.in_sr = int(input_samplerate)
        self.cfg = config
        self._selected: Optional[int] = selected_channel
        self._ring_buffer = ring_buffer

        # --- Output ring buffer (mono at INPUT rate - miniaudio handles resampling) ---
        # OPTIMIZATION: Store data at input rate, not output rate.
        # This uses less memory (e.g., 10kHz vs 44.1kHz) and reduces latency.
        ring_len = max(self.cfg.blocksize * 4, int(self.in_sr * self.cfg.ring_seconds))
        self._ring = np.zeros(ring_len, dtype=np.float32)
        self._r_head = 0  # write index
        self._r_tail = 0  # read index
        self._r_lock = threading.Lock()

        self._stop_evt = threading.Event()
        self._device: Optional[miniaudio.PlaybackDevice] = None

    # ---- Public control ------------------------------------------------------

    def set_selected_channel(self, idx: Optional[int]) -> None:
        self._selected = None if idx is None else int(idx)

    def set_ring_buffer(self, ring_buffer: Optional[SharedRingBuffer]) -> None:
        self._ring_buffer = ring_buffer

    def stop(self) -> None:
        self._stop_evt.set()

    # ---- Ring buffer helpers -------------------------------------------------

    def _ring_space(self) -> int:
        if self._r_head >= self._r_tail:
            used = self._r_head - self._r_tail
        else:
            used = self._ring.size - (self._r_tail - self._r_head)
        return self._ring.size - used - 1

    def _ring_available(self) -> int:
        if self._r_head >= self._r_tail:
            return self._r_head - self._r_tail
        return self._ring.size - (self._r_tail - self._r_head)

    def _ring_write(self, x: np.ndarray) -> None:
        n = int(x.size)
        if n == 0:
            return
            
        # Handle case where input is larger than the entire buffer
        if n > self._ring.size:
            # Just take the last self._ring.size samples
            x = x[-self._ring.size:]
            n = self._ring.size
            
        with self._r_lock:
            space = self._ring_space()
            if n > space:
                # drop-oldest to keep latency bounded
                drop = n - space
                self._r_tail = (self._r_tail + drop) % self._ring.size
            end = min(n, self._ring.size - self._r_head)
            self._ring[self._r_head:self._r_head + end] = x[:end]
            rem = n - end
            if rem:
                self._ring[:rem] = x[end:]
            self._r_head = (self._r_head + n) % self._ring.size

    def _ring_read(self, n: int) -> np.ndarray:
        with self._r_lock:
            avail = self._ring_available()
            n = min(n, avail)
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            end = min(n, self._ring.size - self._r_tail)
            out = np.empty(n, dtype=np.float32)
            out[:end] = self._ring[self._r_tail:self._r_tail + end]
            rem = n - end
            if rem:
                out[end:] = self._ring[:rem]
            self._r_tail = (self._r_tail + n) % self._ring.size
            return out

    # ---- Input chunk extraction -----------------------------------------
    def _extract_frames(self, ch) -> Optional[np.ndarray]:
        """
        Accepts a variety of 'chunk-like' objects and returns a float32 array
        shaped (frames, channels). Returns None if the object isn't recognized.
        """
        arr = None

        # Common possibilities:
        if isinstance(ch, ChunkPointer):
            if self._ring_buffer is None:
                return None
            try:
                block = self._ring_buffer.read(ch.start_index, ch.length)
            except Exception as exc:
                logger.debug("Failed to read from ring buffer: %s", exc)
                return None
            arr = np.asarray(block, dtype=np.float32, order="C").T
        elif hasattr(ch, "data"):
            arr = getattr(ch, "data")
        elif hasattr(ch, "frames"):
            arr = getattr(ch, "frames")
        elif hasattr(ch, "samples"):
            # Spikehound Chunk uses .samples shaped (channels, frames).
            # Convert to (frames, channels) for the player.
            arr = np.asarray(getattr(ch, "samples"), dtype=np.float32, order="C").T
        elif isinstance(ch, np.ndarray):
            arr = ch
        elif isinstance(ch, (list, tuple)) and len(ch):
            # e.g., list/tuple of samples or per-channel arrays
            try:
                arr = np.asarray(ch)
            except Exception as exc:
                logger.debug("Failed to convert chunk to array: %s", exc)
                arr = None

        if arr is None:
            return None

        data = np.asarray(arr, dtype=np.float32, order="C")

        # Ensure 2D: (frames, channels)
        if data.ndim == 1:
            data = data[:, None]
        elif data.ndim > 2:
            # Too weird—ignore this chunk
            return None

        return data

    # ---- Resampling removed ---------------------------------------------------
    # OPTIMIZATION: miniaudio handles resampling from in_sr to hardware rate in C.
    # This eliminates Python overhead from np.interp and reduces latency.

    # ---- Miniaudio generator -------------------------------------------------

    def _audio_generator(self):
        """
        Generator that yields audio data for miniaudio.
        """
        required_frames = yield b""  # Initial yield
        
        while True:
            wanted = required_frames
            mono = self._ring_read(wanted)
            
            # Underrun handling: pad with silence
            if mono.size < wanted:
                mono = np.pad(mono, (0, wanted - mono.size))
                
            mono *= self.cfg.gain
            
            # Convert to bytes (float32)
            data_bytes = mono.tobytes()
            
            required_frames = yield data_bytes

    # ---- Thread body ---------------------------------------------------------

    def run(self) -> None:
        # OPTIMIZATION: Use a strict 20ms hardware buffer for low latency.
        # Miniaudio handles resampling, so we don't need to bloat the buffer 
        # based on input rate (which was punishing low-sr devices).
        # We ensure at least 5ms to avoid underruns on busy systems.
        buf_msec = 20
        
        # Configure miniaudio device with INPUT sample rate.
        # miniaudio handles resampling to hardware rate (e.g., 44.1kHz) in C,
        # which is significantly faster and lower latency than Python.
        try:
            self._device = miniaudio.PlaybackDevice(
                device_id=self.cfg.device,
                nchannels=self.cfg.out_channels,
                # OPTIMIZATION: Use input rate. Miniaudio handles resampling to hardware rate.
                sample_rate=self.in_sr,
                output_format=miniaudio.SampleFormat.FLOAT32,
                buffersize_msec=buf_msec,
            )
            
            # Start playback with our generator
            # Generator must be started (primed) before passing to start()
            gen = self._audio_generator()
            next(gen)
            self._device.start(gen)
            
        except Exception as e:
            logger.error(f"Error starting miniaudio device: {e}")
            return

        try:
            while not self._stop_evt.is_set():
                # Drain upstream queue
                try:
                    ch = self.q.get(timeout=0.05)
                except queue.Empty:
                    continue

                data = self._extract_frames(ch)
                if data is None:
                    # Unknown message type - ignore but keep draining.
                    continue
                if self._selected is None or self._selected >= data.shape[1]:
                    # Muted or invalid selection
                    continue
                
                # Extract mono channel
                mono_in = data[:, self._selected]
                
                # OPTIMIZATION: Removed _resample_block() call.
                # Write raw input samples directly to ring buffer.
                # miniaudio handles resampling to hardware rate in C.
                if mono_in.size:
                    # Very soft limiter to avoid clipping if spikes are tall
                    peak = float(np.max(np.abs(mono_in)))
                    if peak > 1.5:
                        mono_in = mono_in / peak
                    self._ring_write(mono_in)
        finally:
            if self._device and self._device.running:
                self._device.stop()
                self._device.close()
            self._device = None
