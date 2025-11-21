from __future__ import annotations
import queue, threading, time, os
from typing import Optional
import numpy as np
import h5py


class H5LoggerThread:
    """
    Reads float32 blocks shaped (frames, channels) from a Queue
    and appends them to an extendable HDF5 dataset at /samples.
    """
    def __init__(
        self,
        q: "queue.Queue[np.ndarray]",
        out_path: str,
        sample_rate: float,
        channel_names: list[str],
        channel_units: Optional[list[str]] = None,
        channel_props: Optional[dict[str, dict]] = None,
        block_hint: int = 2048,
        compression: Optional[str] = None,   # e.g. "gzip"
    ) -> None:
        self._q = q
        self._out_path = out_path
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._sample_rate = float(sample_rate)
        self._chn = channel_names
        self._units = channel_units or [""] * len(channel_names)
        self._props = channel_props or {}
        self._block_hint = int(block_hint)
        self._compression = compression

        self._f: Optional[h5py.File] = None
        self._dset = None
        self._written = 0

    def start(self) -> None:
        if self._thr and self._thr.is_alive():
            return
        os.makedirs(os.path.dirname(self._out_path) or ".", exist_ok=True)
        self._f = h5py.File(self._out_path, "w")
        f = self._f

        # Root attrs
        f.attrs["sample_rate"] = self._sample_rate
        f.attrs["dtype"] = "float32"
        f.attrs["created_unix"] = time.time()

        # Channels & units
        f.create_dataset("channels", data=np.array(self._chn, dtype=object), dtype=h5py.string_dtype())
        f.create_dataset("units", data=np.array(self._units, dtype=object), dtype=h5py.string_dtype())

        # Channel properties
        meta = f.create_group("meta")
        chgrp = meta.create_group("channel_props")
        for ch_name, props in self._props.items():
            g = chgrp.create_group(ch_name)
            for k, v in props.items():
                if isinstance(v, str):
                    g.attrs[k] = np.string_(v)
                else:
                    g.attrs[k] = v

        # Extendable dataset for samples
        C = len(self._chn)
        self._dset = f.create_dataset(
            "samples",
            shape=(0, C),
            maxshape=(None, C),
            chunks=(self._block_hint, C),
            dtype="float32",
            compression=self._compression,
        )

        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="H5LoggerThread", daemon=True)
        self._thr.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=join_timeout)
            self._thr = None
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None
            self._dset = None

    def _append_block(self, block: np.ndarray) -> None:
        if self._dset is None:
            return
        frames, C = block.shape
        new_len = self._written + frames
        self._dset.resize((new_len, C))
        self._dset[self._written:new_len, :] = block
        self._written = new_len

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                block = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            if block is None:
                continue
            arr = np.asarray(block, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2:
                continue
            self._append_block(arr)
