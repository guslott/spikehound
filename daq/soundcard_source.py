# daq/soundcard_source.py
from __future__ import annotations

import threading
import queue
import numpy as np
import time as _time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

try:
    import sounddevice as sd  # PortAudio bindings (cross‑platform)
except Exception as e:  # pragma: no cover
    sd = None
    _IMPORT_ERROR = e

from .base_source import DataAcquisitionSource, Chunk, DeviceInfo


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    default_samplerate: float


class SoundCardSource(DataAcquisitionSource):
    """
    Audio input DAQ using PortAudio via `sounddevice`.

    Notes
    -----
    • Produces Chunk.data shaped (chunk_size, n_active_channels) to match the simulator.
    • Channel identifiers are simple strings: "In 1", "In 2", ... (1‑based for readability).
    • If no active channels are selected, chunks are not emitted (same contract as others).
    """

    # ---------- Discovery helpers (class-level) --------------------------------

    @classmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        """Return input-capable audio devices as generic DeviceInfo objects."""
        if sd is None:
            raise RuntimeError(
                f"`sounddevice` is not available: {_IMPORT_ERROR!r}"
            )
        devices = sd.query_devices()
        out: List[DeviceInfo] = []
        for idx, dev in enumerate(devices):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in > 0:
                host = sd.query_hostapis(dev["hostapi"])["name"]
                out.append(
                    DeviceInfo(
                        id=str(idx),
                        name=f"[{idx}] {dev['name']} – {host} ({max_in} ch)",
                        extra={
                            "max_input_channels": max_in,
                            "default_samplerate": float(dev.get("default_samplerate", 0)),
                        },
                    )
                )
        return out

    @classmethod
    def supported_sample_rates(
        cls,
        device_index: int,
        probe: Optional[Sequence[int]] = None,
        min_channels: int = 1,
    ) -> List[int]:
        """
        Probe a set of common sample rates for support on `device_index`.

        PortAudio doesn't provide an exhaustive list; we validate candidates with
        `sd.check_input_settings`. The result is a *likely* supported set.

        Parameters
        ----------
        device_index : int
            Index from `list_input_devices()`.
        probe : sequence[int] | None
            Candidate rates to test. Defaults to a sensible cross‑platform set.
        min_channels : int
            Smallest channel count we require for the test (default 1).

        Returns
        -------
        List[int] of supported rates.
        """
        if sd is None:
            raise RuntimeError(
                f"`sounddevice` is not available: {_IMPORT_ERROR!r}"
            )

        if probe is None:
            probe = (
                8000, 11025, 16000, 22050, 32000,
                44100, 48000, 88200, 96000, 176400, 192000
            )
        ok: List[int] = []
        for sr in probe:
            try:
                sd.check_input_settings(
                    device=device_index, channels=min_channels, samplerate=sr
                )
            except Exception:
                continue
            else:
                ok.append(int(sr))
        return ok

    # ---------- Instance configuration ----------------------------------------

    def __init__(
        self,
        sample_rate: int,
        chunk_size: int,
        device: Union[int, str, None] = None,  # index or name substring; None = default input
        channel_ids: Optional[Sequence[Union[int, str]]] = None,  # indices (0‑based) or "In N"
        dtype: str = "float32",
    ):
        """
        Parameters
        ----------
        sample_rate : int
            Desired sample rate in Hz.
        chunk_size : int
            Frames per Chunk.
        device : int | str | None
            Input device index (int) or case‑insensitive name substring (str).
            None uses the system default input device.
        channel_ids : sequence[int|str] | None
            Specific input channels to acquire. If omitted, you can add later
            via `add_channel("In 1")`, etc.
        dtype : str
            Numpy dtype for audio frames (float32 recommended).
        """
        if sd is None:
            raise RuntimeError(
                f"`sounddevice` is not available: {_IMPORT_ERROR!r}"
            )

        super().__init__(sample_rate=sample_rate, chunk_size=chunk_size)
        self.dtype = dtype

        self._device_index = self._resolve_device(device)
        self._dev_info = sd.query_devices(self._device_index)
        self._n_in = int(self._dev_info["max_input_channels"])
        if self._n_in <= 0:
            raise ValueError("Selected device has no input channels.")

        # Stable, user-friendly channel names: "In 1", "In 2", ...
        self._chan_names: List[str] = [f"In {i+1}" for i in range(self._n_in)]

        # Buffering for the callback → chunkizer path
        self._buf_lock = threading.Lock()
        self._residual = np.zeros((0, self._n_in), dtype=np.float32)

        # Pre-seed active channels if requested
        if channel_ids:
            for ch in channel_ids:
                name = self._normalize_channel_id(ch)
                if name in self._chan_names:
                    self.add_channel(name)

    # ---------- Required interface --------------------------------------------

    def list_available_channels(self) -> list:
        """Return the list of available input channels for the selected device."""
        return list(self._chan_names)

    def run(self):
        """Open an InputStream and package audio into fixed-size Chunk objects."""
        # Validate the requested configuration before opening the stream
        sd.check_input_settings(
            device=self._device_index,
            channels=self._n_in,
            samplerate=self.sample_rate,
        )
        self.reset_counters()
        self._residual = np.zeros((0, self._n_in), dtype=np.float32)

        def _callback(indata, frames, time_info, status):
            # This runs on PortAudio's audio thread; keep it lean & non‑blocking.
            if status:
                # You could log xruns/latency warnings here if desired.
                pass

            # Copy to avoid referencing a temporary buffer owned by PortAudio
            data = np.asarray(indata, dtype=np.float32, order="C")

            with self._buf_lock:
                if self._residual.size == 0:
                    self._residual = data.copy()
                else:
                    # vstack is fine at these block sizes; replace with deque if needed
                    self._residual = np.vstack((self._residual, data))

                # Emit one or more fixed-size chunks
                while self._residual.shape[0] >= self.chunk_size:
                    frames_chunk = self._residual[: self.chunk_size, :]
                    self._residual = self._residual[self.chunk_size :, :]

                    idxs = self._active_indices()
                    if not idxs:
                        continue  # no selection yet

                    # Select active channels and form (chunk_size, n_active)
                    data_chunk = frames_chunk[:, idxs]

                    start, seq = self._next_chunk_meta()
                    ch = Chunk(
                        start_sample=start,
                        mono_time=_time.monotonic(),
                        seq=seq,
                        data=data_chunk,
                    )
                    self._safe_put(self.data_queue, ch)

        # Open the stream with `channels = total input channels`.
        # We downselect to active channels inside the callback to keep mapping simple.
        with sd.InputStream(
            device=self._device_index,
            channels=self._n_in,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,  # a hint; the host may choose a nearby size
            dtype=self.dtype,
            callback=_callback,
        ):
            # Keep our thread alive while the callback runs on PortAudio's thread.
            while self.is_running():
                _time.sleep(0.05)

    # ---------- Internals ------------------------------------------------------

    def _resolve_device(self, device: Union[int, str, None]) -> int:
        """Normalize `device` to an input device index."""
        # Explicit index
        if isinstance(device, int):
            return device

        # If provided a DeviceInfo.id (stringified index), accept it
        if isinstance(device, str) and device.isdigit():
            return int(device)

        # Default input device (tuple: (input_index, output_index))
        default_in = sd.default.device[0] if sd.default.device else None
        if device is None and default_in is not None and default_in >= 0:
            return int(default_in)

        # First available input device as fallback
        if device is None:
            for i, dev in enumerate(sd.query_devices()):
                if int(dev.get("max_input_channels", 0)) > 0:
                    return i
            raise RuntimeError("No input-capable audio device found on this system.")

        # Name substring (case-insensitive)
        name_lc = str(device).lower()
        for i, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) > 0 and name_lc in dev["name"].lower():
                return i
        raise ValueError(f"No input device matching '{device}'.")

    def _normalize_channel_id(self, ch: Union[int, str]) -> str:
        """Accept 0‑based index or 'In N' string, return normalized 'In N'."""
        if isinstance(ch, int):
            if not (0 <= ch < self._n_in):
                raise IndexError(f"Channel index {ch} out of range (0..{self._n_in-1})")
            return f"In {ch+1}"
        if isinstance(ch, str) and ch.startswith("In "):
            # crude but sufficient
            return ch
        if isinstance(ch, str) and ch.isdigit():
            idx = int(ch)
            if not (0 <= idx < self._n_in):
                raise IndexError(f"Channel index {idx} out of range (0..{self._n_in-1})")
            return f"In {idx+1}"
        # fallback raises
        raise ValueError(f"Unrecognized channel identifier: {ch!r}")

    def _active_indices(self) -> List[int]:
        """Map active channel names → 0‑based numeric indices."""
        with self._channel_lock:
            names = list(self.active_channels)
        idxs: List[int] = []
        for name in names:
            try:
                i = self._chan_names.index(self._normalize_channel_id(name))
            except Exception:
                continue
            idxs.append(i)
        return idxs

    @staticmethod
    def _safe_put(q: "queue.Queue[Chunk]", item: Chunk) -> None:
        """Non‑blocking put with drop‑oldest on overflow."""
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                # give up silently; UI stays live
                pass


# Optional quick smoke test: prints devices and exits
if __name__ == "__main__":  # pragma: no cover
    devs = SoundCardSource.list_available_devices()
    for d in devs:
        print(f"{d.id}: {d.name}")
