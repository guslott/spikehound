# daq/soundcard_source.py
from __future__ import annotations

import threading
import numpy as np
import time as _time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

try:
    import sounddevice as sd  # PortAudio bindings (cross‑platform)
except Exception as e:  # pragma: no cover
    sd = None
    _IMPORT_ERROR = e

from .base_device import (
    BaseDevice,
    Chunk,
    DeviceInfo,
    ChannelInfo,
    Capabilities,
    ActualConfig,
)


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    default_samplerate: float


class SoundCardSource(BaseDevice):
    """
    Audio input DAQ using PortAudio via `sounddevice`.

    Notes
    -----
    • Driver callbacks deliver arrays shaped (frames, channels); downstream consumers
      receive `Chunk.samples` shaped (channels, frames).
    • Channel identifiers are simple strings: "In 1", "In 2", ... (1‑based for readability).
    • If no active channels are selected, chunks are not emitted (same contract as others).
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
        if sd is None:
            raise RuntimeError(
                f"`sounddevice` is not available: {_IMPORT_ERROR!r}"
            )
        devices = sd.query_devices()
        out: List[DeviceInfo] = []
        if not cls._list_all_devices():
            # Prefer the system default input; fall back to first input-capable device.
            default_in = sd.default.device[0] if sd.default.device else None
            target_idx = None
            if default_in is not None and default_in >= 0:
                dev = devices[default_in]
                if int(dev.get("max_input_channels", 0)) > 0:
                    target_idx = default_in
            if target_idx is None:
                for idx, dev in enumerate(devices):
                    if int(dev.get("max_input_channels", 0)) > 0:
                        target_idx = idx
                        break
            if target_idx is None:
                return []
            dev = devices[target_idx]
            host = sd.query_hostapis(dev["hostapi"])["name"]
            name = dev["name"]
            out.append(
                DeviceInfo(
                    id=str(target_idx),
                    name=f"System Sound Input - {name}",
                    details={
                        "max_input_channels": int(dev.get("max_input_channels", 0)),
                        "default_samplerate": float(dev.get("default_samplerate", 0)),
                        "host_api": host,
                    },
                )
            )
            return out

        for idx, dev in enumerate(devices):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in > 0:
                host = sd.query_hostapis(dev["hostapi"])["name"]
                out.append(
                    DeviceInfo(
                        id=str(idx),
                        name=f"[{idx}] {dev['name']} – {host} ({max_in} ch)",
                        details={
                            "max_input_channels": max_in,
                            "default_samplerate": float(dev.get("default_samplerate", 0)),
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

    def __init__(self, queue_maxsize: int = 64, dtype: str = "float32"):
        if sd is None:
            # Allow constructing even without sounddevice; discovery will raise later
            pass
        super().__init__(queue_maxsize=queue_maxsize)
        self.dtype = dtype
        self._device_index: Optional[int] = None
        self._dev_info = None
        self._n_in: int = 0
        self._chan_names: List[str] = []
        self._buf_lock = threading.Lock()
        self._residual = np.zeros((0, 1), dtype=np.float32)
        self._stream = None

    # ---------- Required interface --------------------------------------------

    def get_capabilities(self, device_id: str) -> Capabilities:
        if sd is None:
            raise RuntimeError(f"`sounddevice` unavailable: {_IMPORT_ERROR!r}")
        dev = sd.query_devices(int(device_id))
        max_in = int(dev.get("max_input_channels", 0))
        # Offer a curated set of common rates, filtered by device support.
        standard_rates = (22050, 44100, 48000, 96000)
        supported = self.supported_sample_rates(int(device_id), probe=standard_rates, min_channels=1)
        if not supported:
            supported = self.supported_sample_rates(int(device_id), min_channels=1)
        return Capabilities(max_channels_in=max_in, sample_rates=supported, dtype=self.dtype)

    def list_available_channels(self, device_id: str) -> List[ChannelInfo]:
        if sd is None:
            raise RuntimeError(f"`sounddevice` unavailable: {_IMPORT_ERROR!r}")
        dev = sd.query_devices(int(device_id))
        max_in = int(dev.get("max_input_channels", 0))
        return [ChannelInfo(id=i, name=f"In {i+1}", units="V") for i in range(max_in)]

    # ---------- Lifecycle ------------------------------------------------------

    def _open_impl(self, device_id: str) -> None:
        if sd is None:
            raise RuntimeError(f"`sounddevice` unavailable: {_IMPORT_ERROR!r}")
        self._device_index = int(device_id) if str(device_id).isdigit() else self._resolve_device(device_id)
        self._dev_info = sd.query_devices(self._device_index)
        self._n_in = int(self._dev_info["max_input_channels"])
        self._chan_names = [f"In {i+1}" for i in range(self._n_in)]

    def _close_impl(self) -> None:
        # Nothing persistent to close here; start/stop manages the stream
        pass

    def _configure_impl(self, sample_rate: int, channels: Sequence[int], chunk_size: int, **options) -> ActualConfig:
        if sd is None:
            raise RuntimeError(f"`sounddevice` unavailable: {_IMPORT_ERROR!r}")

        if self._device_index is None:
            raise RuntimeError("Device must be opened before configure().")

        try:
            sd.check_input_settings(device=self._device_index, samplerate=sample_rate, channels=len(channels))
        except Exception as exc:
            raise ValueError(f"Sample rate {sample_rate} Hz not supported: {exc}") from exc

        self._residual = np.zeros((0, self._n_in), dtype=np.float32)
        id_to_info = {ch.id: ch for ch in self._available_channels}
        selected = [id_to_info[c] for c in channels]
        cfg = ActualConfig(sample_rate=sample_rate, channels=selected, chunk_size=chunk_size, dtype=self.dtype)
        return cfg

    def _start_impl(self) -> None:
        assert self._device_index is not None and self.config is not None
        if self._stream is not None:
            return

        def _callback(indata, frames, time_info, status):
            if status:
                # Could log xruns
                pass
            data = np.asarray(indata, dtype=np.float32, order="C")
            with self._buf_lock:
                if self._residual.size == 0:
                    self._residual = data.copy()
                else:
                    self._residual = np.vstack((self._residual, data))
                while self._residual.shape[0] >= self.config.chunk_size:
                    frames_chunk = self._residual[: self.config.chunk_size, :]
                    self._residual = self._residual[self.config.chunk_size :, :]
                    idxs = [c.id for c in self.get_active_channels()]
                    if not idxs:
                        continue
                    data_chunk = frames_chunk[:, idxs]
                    # Emit via base to stamp counters
                    self.emit_array(data_chunk, mono_time=_time.monotonic())

        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                self._stream = None
        self._stream = sd.InputStream(
            device=self._device_index,
            channels=self._n_in,
            samplerate=self.config.sample_rate,
            blocksize=self.config.chunk_size,
            dtype=self.dtype,
            callback=_callback,
        )
        self._stream.start()


    def _stop_impl(self) -> None:
        # Stop and close the PortAudio stream synchronously to release the device.
        try:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    try:
                        self._stream.abort()
                    except Exception:
                        pass
                try:
                    self._stream.close()
                finally:
                    self._stream = None
        finally:
            with self._buf_lock:
                self._residual = np.zeros((0, self._n_in if self._n_in else 1), dtype=np.float32)

    # ---------- Internals ------------------------------------------------------

    def _resolve_device(self, device: Union[int, str, None]) -> int:
        """Normalize `device` to an input device index."""
        # Explicit index
        if isinstance(device, int):
            return device

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

    # Optional quick smoke test: prints devices and exits
if __name__ == "__main__":  # pragma: no cover
    devs = SoundCardSource.list_available_devices()
    for d in devs:
        print(f"{d.id}: {d.name}")
