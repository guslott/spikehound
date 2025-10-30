from __future__ import annotations

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
    out_samplerate: int = 44_100   # soundcard output Hz
    out_channels: int = 1          # 1=mono (start simple)
    device: Optional[int | str] = None  # None = default output device
    gain: float = 0.35             # careful: spikes can be loud
    blocksize: int = 512           # PortAudio block size (latency knob)
    ring_seconds: float = 0.75      # size of the output ring buffer (seconds)


class AudioPlayer(threading.Thread):
    """
    Consumes Chunk-like objects from an audio_queue. Each chunk must have a
    .data shaped (frames, channels) at input_sr. We select one channel, resample
    to out_sr (linear interpolation), and play via sounddevice.OutputStream.
    """

    def __init__(
        self,
        audio_queue: "queue.Queue",
        *,
        input_samplerate: int,
        config: AudioConfig = AudioConfig(),
        selected_channel: Optional[int] = 0,  # 0 by default; None = mute
    ) -> None:
        super().__init__(name="AudioPlayer", daemon=True)
        if sd is None:
            raise RuntimeError(f"`sounddevice` is not available: {_SD_IMPORT_ERR!r}")

        self.q = audio_queue
        self.in_sr = int(input_samplerate)
        self.cfg = config
        self._selected: Optional[int] = selected_channel

        # --- Output ring buffer (mono at out_sr) ---
        ring_len = max(self.cfg.blocksize * 4, int(self.cfg.out_samplerate * self.cfg.ring_seconds))
        self._ring = np.zeros(ring_len, dtype=np.float32)
        self._r_head = 0  # write index
        self._r_tail = 0  # read index
        self._r_lock = threading.Lock()

        # For interpolation time tracking
        self._t_in_cursor = 0.0  # seconds in input timebase

        self._stop_evt = threading.Event()
        self._stream = None

    # ---- Public control ------------------------------------------------------

    def set_selected_channel(self, idx: Optional[int]) -> None:
        self._selected = None if idx is None else int(idx)

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
        if hasattr(ch, "data"):
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
            except Exception:
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

    # ---- Resampling (linear interpolation) -----------------------------------

    def _resample_block(self, mono_in: np.ndarray) -> np.ndarray:
        """
        Convert mono_in at in_sr -> out_sr using linear interpolation.
        Keeps a rolling input-time cursor so blocks stitch together.
        """
        if mono_in.size == 0:
            return mono_in.astype(np.float32)

        # Input times for this block (seconds)
        N = mono_in.size
        t0 = self._t_in_cursor
        dt_in = 1.0 / self.in_sr
        t = t0 + np.arange(N, dtype=np.float64) * dt_in
        self._t_in_cursor = t[-1] + dt_in  # advance cursor to "just after" this block

        # Output times covering [t[0], t[-1]] at out_sr
        dt_out = 1.0 / self.cfg.out_samplerate
        n_out = int(np.floor((t[-1] - t[0]) / dt_out)) + 1
        if n_out <= 1:
            return np.zeros(0, dtype=np.float32)
        t_out = np.linspace(t[0], t[0] + (n_out - 1) * dt_out, num=n_out, dtype=np.float64)

        # Interpolate
        y = np.interp(t_out, t, mono_in.astype(np.float64, copy=False)).astype(np.float32, copy=False)
        return y

    # ---- PortAudio callback --------------------------------------------------

    def _callback(self, outdata, frames, time_info, status):
        if status:
            # xruns/underruns may be reported here; we just keep streaming
            pass
        wanted = int(frames)
        mono = self._ring_read(wanted)
        if mono.size < wanted:
            mono = np.pad(mono, (0, wanted - mono.size))  # underrun -> silence
        mono *= self.cfg.gain

        # map mono -> (frames,channels) as required by PortAudio/sounddevice
        if self.cfg.out_channels == 1:
            out = mono[:, None] #shape (frames, 1)
        else:
            out = np.tile(mono[:, None], (1, self.cfg.out_channels)) #(frames, C)
        outdata[:] = out

    # ---- Thread body ---------------------------------------------------------

    def run(self) -> None:
        # Open the soundcard stream
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
            while not self._stop_evt.is_set():
                # Drain upstream queue
                try:
                    ch = self.q.get(timeout=0.05)
                except queue.Empty:
                    continue

                data = self._extract_frames(ch)
                if data is None:
                    #Unknown message type - ignore but keep draining.
                    continue
                if self._selected is None or self._selected >= data.shape[1]:
                    #muted or invalid selection
                    continue
                mono_in = data[:, self._selected]
                mono_out = self._resample_block(mono_in)

                if mono_out.size:
                    # very soft limiter to avoid clipping if spikes are tall
                    peak = float(np.max(np.abs(mono_out)))
                    if peak > 1.5:
                        mono_out = mono_out / peak
                    self._ring_write(mono_out)
        finally:
            try:
                if self._stream:
                    self._stream.stop()
                    self._stream.close()
            finally:
                self._stream = None
