∑from __future__ import annotations

import threading
import queue
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    sd = None
    _SD_IMPORT_ERR = e


@dataclass
class AudioConfig:
    out_samplerate: int = 44_100
    out_channels: int = 1
    device: Optional[int | str] = None
    gain: float = 0.35
    blocksize: int = 512
    ring_seconds: float = 0.5


class AudioPlayer(threading.Thread):
    """Simplified audio playback thread for SpikeHound."""

    def __init__(
        self,
        audio_queue: "queue.Queue",
        *,
        input_samplerate: int,
        config: AudioConfig = AudioConfig(),
        selected_channel: Optional[int] = 0,
    ) -> None:
        super().__init__(name="AudioPlayer", daemon=True)
        if sd is None:
            raise RuntimeError(f"`sounddevice` not available: {_SD_IMPORT_ERR!r}")

        self.q = audio_queue
        self.in_sr = int(input_samplerate)
        self.cfg = config
        self._selected = selected_channel
        ring_len = max(self.cfg.blocksize * 4, int(self.cfg.out_samplerate * self.cfg.ring_seconds))
        self._ring = np.zeros(ring_len, dtype=np.float32)
        self._r_head = 0
        self._r_tail = 0
        self._r_lock = threading.Lock()
        self._t_in_cursor = 0.0
        self._stop = threading.Event()
        self._stream = None

    # ----- helpers -----
    def _ring_space(self):
        if self._r_head >= self._r_tail:
            used = self._r_head - self._r_tail
        else:
            used = self._ring.size - (self._r_tail - self._r_head)
        return self._ring.size - used - 1

    def _ring_available(self):
        if self._r_head >= self._r_tail:
            return self._r_head - self._r_tail
        return self._ring.size - (self._r_tail - self._r_head)

    def _ring_write(self, x):
        n = int(x.size)
        if n == 0:
            return
        with self._r_lock:
            space = self._ring_space()
            if n > space:
                drop = n - space
                self._r_tail = (self._r_tail + drop) % self._ring.size
            end = min(n, self._ring.size - self._r_head)
            self._ring[self._r_head:self._r_head + end] = x[:end]
            rem = n - end
            if rem:
                self._ring[:rem] = x[end:]
            self._r_head = (self._r_head + n) % self._ring.size

    def _ring_read(self, n):
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

    def _resample_block(self, mono_in):
        if mono_in.size == 0:
            return np.zeros(0, dtype=np.float32)
        N = mono_in.size
        t0 = self._t_in_cursor
        dt_in = 1.0 / self.in_sr
        t = t0 + np.arange(N) * dt_in
        self._t_in_cursor = t[-1] + dt_in
        dt_out = 1.0 / self.cfg.out_samplerate
        n_out = int(np.floor((t[-1] - t[0]) / dt_out)) + 1
        if n_out <= 1:
            return np.zeros(0, dtype=np.float32)
        t_out = np.linspace(t[0], t[0] + (n_out - 1) * dt_out, n_out)
        return np.interp(t_out, t, mono_in).astype(np.float32)

    def _callback(self, outdata, frames, time_info, status):
        if status:
            pass  # xruns/underruns etc.

        wanted = int(frames)
        mono = self._ring_read(wanted)
        if mono.size < wanted:
            mono = np.pad(mono, (0, wanted - mono.size))

        # Apply gain once
        mono = mono * self.cfg.gain

        # sounddevice expects shape (frames, channels)
        if self.cfg.out_channels == 1:
            outdata[:, 0] = mono
        else:
            out = np.tile(mono[:, None], (1, self.cfg.out_channels))
            outdata[:, :] = out

	def stop(self) -> None:
        """Stop the playback thread cleanly."""
        self._stop.set()
	def _callback(self, outdata, frames, time_info, status):
        if status:
            pass  # xruns/underruns etc.

        wanted = int(frames)
        mono = self._ring_read(wanted)
            def stop(self) -> None:
	        self._stop.set()
if mono∑.size < wanted:
            mono = np.pad(mono, (0, wanted - mono.size))

        # Apply gain once
        mono = mono * self.cfg.gain

        # sounddevice expects shape (frames, channels)
        if self.cfg.out_channels == 1:
            outdata[:, 0] = mono
        else:
            out = np.tile(mono[:, None], (1, self.cfg.out_channels))
            outdata[:, :] = out

    def _coerce_to_2d_array(self, raw):
        if isinstance(raw, np.ndarray):
            arr = raw.astype(np.float32, copy=False)
            return arr[:, None] if arr.ndim == 1 else arr
        if hasattr(raw, "data"):
            arr = np.asarray(raw.data, dtype=np.float32)
            return arr[:, None] if arr.ndim == 1 else arr
        for key in ("samples", "y", "buffer", "chunk"):
            if hasattr(raw, key):
                arr = np.asarray(getattr(raw, key), dtype=np.float32)
                return arr[:, None] if arr.ndim == 1 else arr
        if isinstance(raw, (list, tuple)):
            arr = np.asarray(raw, dtype=np.float32)
            return arr[:, None] if arr.ndim == 1 else arr
        return None

    # ----- thread body -----
    def run(self):
        self._stream = sd.OutputStream(
            device=self.cfg.device,
            samplerate=self.cfg.out_samplerate,
            channels=self.cfg.out_channels,
            dtype="float32",
            blocksize=self.cfg.blocksize,
            callback=self._callback,
        )
        self._stream.start()
        try:
            while not self._stop.is_set():
                try:
                    raw = self.q.get(timeout=0.05)
                except queue.Empty:
                    continue
                data = self._coerce_to_2d_array(raw)
                if data is None or data.size == 0:
                    continue
                if self._selected is None or self._selected >= data.shape[1]:
                    continue
                mono_in = data[:, self._selected]
                mono_out = self._resample_block(mono_in)
                if mono_out.size:
                    peak = float(np.max(np.abs(mono_out)))
                    if peak > 1.5:
                        mono_out = mono_out / peak
                    self._ring_write(mono_out)
        finally:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

