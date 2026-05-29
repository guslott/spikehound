from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import miniaudio
except ImportError as e:  # pragma: no cover
    miniaudio = None
    _IMPORT_ERR = e


@dataclass
class AudioConfig:
    out_channels: int = 1          # 1=mono (start simple)
    device: Any = None             # None = default output device
    blocksize: int = 256           # miniaudio buffer size (frames)
    ring_seconds: float = 0.2      # size of the output ring buffer (seconds)
    frames_per_write: int = 0      # frames the bridge writes per chunk (0 = unknown)
    playback_buffer_msec: int = 10 # miniaudio playback device buffer (ms)


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
    Mono playback device fed by a software ring buffer.

    The :class:`~core.monitor_audio_bridge.MonitorAudioBridge` writes filtered,
    channel-selected, gained mono samples (at ``input_samplerate``) directly into
    this player's ring via :meth:`_ring_write` from the capture/emitter thread.
    The miniaudio playback callback drains that ring through ``_audio_generator``.

    OPTIMIZATION: miniaudio handles resampling from input_sr to the hardware
    rate in C, which is significantly faster and lower latency than Python.
    """

    def __init__(
        self,
        *,
        input_samplerate: int,
        config: Optional[AudioConfig] = None,
    ) -> None:
        super().__init__(name="AudioPlayer", daemon=True)
        if miniaudio is None:
            raise RuntimeError(f"`miniaudio` is not available: {_IMPORT_ERR!r}")

        self.in_sr = int(input_samplerate)
        self.cfg = config if config is not None else AudioConfig()

        # --- Output ring buffer (mono at INPUT rate - miniaudio handles resampling) ---
        # OPTIMIZATION: Store data at input rate, not output rate.
        # This uses less memory (e.g., 10kHz vs 44.1kHz) and reduces latency.
        #
        # The ring MUST hold at least one full write-block (one source chunk)
        # plus a chunk of scheduler jitter.  Otherwise an oversized write is
        # clamped to the ring size in _ring_write() and the head of every chunk
        # is silently dropped — so we size it to >= 2x the per-write frame count.
        # (Capacity, not fill, grows; average ring latency stays ~half a chunk.)
        ring_len = max(
            self.cfg.blocksize * 4,
            int(self.in_sr * self.cfg.ring_seconds),
            int(self.cfg.frames_per_write) * 2,
        )
        self._ring = np.zeros(ring_len, dtype=np.float32)
        self._r_head = 0   # write index
        self._r_tail = 0   # read index
        # Explicit fill-level counter.  Using head == tail to detect both "empty"
        # and "full" is ambiguous; _r_count is the single authoritative source of
        # truth.  Invariant (always held under _r_lock):
        #   _r_count == number of unread samples in the ring
        #   space    == _ring.size - _r_count   (never < 0)
        #   available == _r_count               (never > _ring.size)
        self._r_count = 0
        self._r_lock = threading.Lock()

        self._stop_evt = threading.Event()
        self._device: Optional[miniaudio.PlaybackDevice] = None
        self._playback_buf_msec: int = 20  # updated in run(); used for latency estimate

    # ---- Public control ------------------------------------------------------

    def stop(self) -> None:
        self._stop_evt.set()

    def estimated_latency_ms(self) -> float:
        """Return a conservative estimate of current end-to-end monitor playback latency.

        Accounts for:
          - Samples currently sitting in the software ring waiting to be consumed
            by the playback callback (variable, reflects real-time queue depth).
          - The miniaudio playback device hardware buffer (fixed, set at device open).

        Does NOT include upstream pipeline latency (dispatcher queue, audio router
        thread, capture device buffer).  Those fixed contributions are tracked
        separately via ``AudioManager.monitor_latency_ms()``.
        """
        with self._r_lock:
            ring_samples = self._r_count
        ring_ms = (ring_samples / self.in_sr) * 1000.0 if self.in_sr > 0 else 0.0
        return ring_ms + float(self._playback_buf_msec)

    # ---- Ring buffer helpers -------------------------------------------------
    # All three methods must be called with _r_lock held (or from __init__).
    # _r_count is the canonical fill level; head/tail track positions only.

    def _ring_space(self) -> int:
        """Writable slots.  space + _r_count == _ring.size, always."""
        return self._ring.size - self._r_count

    def _ring_available(self) -> int:
        """Readable samples.  Equals _r_count directly."""
        return self._r_count

    def _ring_write(self, x: np.ndarray) -> None:
        n = int(x.size)
        if n == 0:
            return

        # Clamp oversized writes to the ring capacity so the drop-oldest
        # logic below can never produce a negative space result.
        if n > self._ring.size:
            x = x[-self._ring.size:]
            n = self._ring.size

        with self._r_lock:
            space = self._ring_space()          # _ring.size - _r_count
            if n > space:
                # Drop-oldest: advance tail to make exactly n slots available.
                drop = n - space
                self._r_tail = (self._r_tail + drop) % self._ring.size
                self._r_count -= drop           # account for discarded samples
            end = min(n, self._ring.size - self._r_head)
            self._ring[self._r_head:self._r_head + end] = x[:end]
            rem = n - end
            if rem:
                self._ring[:rem] = x[end:]
            self._r_head = (self._r_head + n) % self._ring.size
            self._r_count += n                  # n new samples are now readable

    def _ring_read(self, n: int) -> np.ndarray:
        with self._r_lock:
            avail = self._ring_available()      # == _r_count
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
            self._r_count -= n                  # samples have been consumed
            return out

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

            # OPTIMIZATION: Gain is now applied before ring write (see run method)
            # This reduces work in the audio callback for lower latency

            # Convert to bytes (float32)
            data_bytes = mono.tobytes()

            required_frames = yield data_bytes

    # ---- Thread body ---------------------------------------------------------

    def run(self) -> None:
        # OPTIMIZATION: Use a strict 20ms hardware buffer for low latency.
        # Miniaudio handles resampling, so we don't need to bloat the buffer 
        # based on input rate (which was punishing low-sr devices).
        # We ensure at least 5ms to avoid underruns on busy systems.
        # Playback device buffer.  Configurable via AudioConfig.playback_buffer_msec
        # (10 ms default; ~5 ms in the opt-in low-latency monitor mode).  Clamp to
        # a >= 2 ms floor to avoid pathological underruns.  The MonitorAudioBridge
        # writes directly into the ring, so this is the dominant fixed output
        # contribution to monitor latency.
        buf_msec = max(2, int(self.cfg.playback_buffer_msec))
        self._playback_buf_msec = buf_msec

        # Open the playback device at the INPUT sample rate.
        #
        # Rate handling (no forced resampling): when the device/OS can run at
        # in_sr (the common sound-card-monitor case, e.g. 44.1 kHz in -> 44.1 kHz
        # out), miniaudio runs the device at in_sr and its resampler is a
        # pass-through — zero resampling latency.  Only when the device is locked
        # to a different rate (e.g. a 10 kHz BYB input -> 44.1/48 kHz hardware)
        # does miniaudio engage its low-latency linear resampler, in C.  So
        # matching rates already cost nothing, and mismatched rates use the
        # cheapest converter available; there is nothing to gain by resampling
        # ourselves in Python.
        try:
            self._device = miniaudio.PlaybackDevice(
                device_id=self.cfg.device,
                nchannels=self.cfg.out_channels,
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
            # The miniaudio playback callback pulls samples from the ring via the
            # generator above; the MonitorAudioBridge fills that ring from the
            # capture/emitter thread (channel select + gain + clip happen there).
            # This thread only needs to keep the device open until stopped.
            self._stop_evt.wait()
        finally:
            if self._device and self._device.running:
                self._device.stop()
                self._device.close()
            self._device = None
