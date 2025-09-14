from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import threading
import queue
import time

@dataclass
class Chunk:
    """Metadata and data for a produced chunk.

    All streaming in the project should pass `Chunk` objects exclusively.
    """
    start_sample: int
    mono_time: float
    seq: int
    data: object  # numpy.ndarray


@dataclass(frozen=True)
class DeviceInfo:
    """Generic device descriptor returned by DAQ drivers.

    Fields are intentionally minimal and generic. The `id` is an opaque string
    that the driver understands (e.g., an integer index as a string). The UI may
    display `name` and pass `id` back into the driver when constructing.
    """
    id: str
    name: str
    extra: dict | None = None


class DataAcquisitionSource(ABC, threading.Thread):
    """
    Abstract base class for all data acquisition sources.
    Defines the "contract" for any hardware interface.

    This version now includes timestamping for each data chunk.
    """
    def __init__(self, sample_rate: int, chunk_size: int):
        # Initialize as a daemon thread so it exits when the main program does
        super().__init__(daemon=True)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        # Single queue of Chunk objects for consumers (UI/logger).
        # Bounded to avoid runaway memory; producers may drop-oldest on overflow.
        self.data_queue = queue.Queue(maxsize=200)
        self._is_running = threading.Event()
        
        self.active_channels = []
        # A lock is crucial for safely modifying the channel list from different threads
        self._channel_lock = threading.Lock()
        
        # Sample-accurate counters for chunk metadata
        self._start_sample_counter = 0  # int64 logical sample index of next chunk start
        self._seq_no = 0

    @abstractmethod
    def run(self):
        """The main threaded loop for acquiring data."""
        pass

    @abstractmethod
    def list_available_channels(self) -> list:
        """Returns a list of available physical channels."""
        pass

    # Class-level discovery API so UIs can show device choices before opening
    @classmethod
    @abstractmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        """Enumerate devices supported by this driver on this system."""
        raise NotImplementedError

    def add_channel(self, channel_id):
        """Adds a channel to the list of active channels to be acquired."""
        with self._channel_lock:
            if channel_id not in self.active_channels:
                self.active_channels.append(channel_id)
                # Ensure active channels are sorted for predictable data column ordering
                self.active_channels.sort()
                print(f"Channel {channel_id} added. Active channels: {self.active_channels}")

    def remove_channel(self, channel_id):
        """Removes a channel from the list of active channels."""
        with self._channel_lock:
            if channel_id in self.active_channels:
                self.active_channels.remove(channel_id)
                print(f"Channel {channel_id} removed. Active channels: {self.active_channels}")

    def start(self):
        """Starts the data acquisition thread."""
        if not self.active_channels:
            print("Warning: Starting DAQ source with no active channels.")
        print(f"Starting {self.__class__.__name__}...")
        self._is_running.set()
        super().start()

    def stop(self):
        """Stops the data acquisition thread gracefully."""
        print(f"Stopping {self.__class__.__name__}...")
        self._is_running.clear()
        self.join() # Wait for the thread to finish its current loop and exit

    def is_running(self) -> bool:
        """Returns True if the thread is running."""
        return self._is_running.is_set()

    # --- Helpers for subclasses -------------------------------------------------
    def _next_chunk_meta(self) -> tuple[int, int]:
        """Returns `(start_sample, seq)` for the next chunk and advances counters."""
        start = self._start_sample_counter
        seq = self._seq_no
        self._start_sample_counter += self.chunk_size
        self._seq_no += 1
        return start, seq

    def reset_counters(self):
        """Reset sample and sequence counters (e.g., on rearm)."""
        self._start_sample_counter = 0
        self._seq_no = 0
