from __future__ import annotations

from threading import RLock
from typing import Tuple

import numpy as np


class SharedRingBuffer:
    """
    Thread-safe ring buffer backed by a preallocated NumPy array.

    The last dimension of `shape` defines the capacity of the buffer. Writes
    and reads operate along this axis.
    """

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype | str = np.float32) -> None:
        if not shape:
            raise ValueError("shape must be a non-empty tuple")
        shape = tuple(int(dim) for dim in shape)
        if any(dim <= 0 for dim in shape):
            raise ValueError("all dimensions in shape must be positive")

        self._shape = shape
        self._capacity = shape[-1]
        self._data = np.empty(self._shape, dtype=dtype)
        self._lock = RLock()
        self._write_pos = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the buffer elements."""
        return self._data.dtype

    def write(self, data: np.ndarray) -> int:
        """
        Write `data` into the ring buffer, handling wrap-around.

        Returns the start index where the data was written (mod capacity).
        """
        arr = np.asarray(data, dtype=self._data.dtype)
        if arr.ndim != len(self._shape):
            raise ValueError(f"data must have {len(self._shape)} dimensions, got {arr.ndim}")
        if arr.shape[:-1] != self._shape[:-1]:
            raise ValueError("data shape mismatch on non-ring axes")

        length = arr.shape[-1]
        if length <= 0:
            raise ValueError("length of data must be positive")
        if length > self._capacity:
            raise ValueError("data length exceeds ring buffer capacity")

        with self._lock:
            start = self._write_pos
            end = start + length

            if end <= self._capacity:
                self._data[..., start:end] = arr
            else:
                first = self._capacity - start
                self._data[..., start:] = arr[..., :first]
                self._data[..., : end - self._capacity] = arr[..., first:]

            self._write_pos = (start + length) % self._capacity
            return start

    def read(self, start_index: int, length: int) -> np.ndarray:
        """
        Read `length` samples starting at `start_index`.

        Returns a view when the data is contiguous; returns a copy if the range
        wraps around the end of the buffer.
        """
        if length <= 0:
            raise ValueError("length must be positive")
        if length > self._capacity:
            raise ValueError("length exceeds ring buffer capacity")
        if not 0 <= start_index < self._capacity:
            raise ValueError("start_index out of range")

        with self._lock:
            end = start_index + length
            if end <= self._capacity:
                return self._data[..., start_index:end]

            first = self._capacity - start_index
            tail_len = end - self._capacity
            return np.concatenate(
                (self._data[..., start_index:], self._data[..., :tail_len]),
                axis=-1,
            )


__all__ = ["SharedRingBuffer"]
