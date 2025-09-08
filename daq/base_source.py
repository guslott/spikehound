from abc import ABC, abstractmethod
import threading
import queue

class DataAcquisitionSource(ABC, threading.Thread):
    """
    Abstract base class for all data acquisition sources.
    Defines the "contract" for any hardware interface.
    """
    def __init__(self, sample_rate: int, chunk_size: int):
        # Initialize as a daemon thread so it exits when the main program does
        super().__init__(daemon=True)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        # Create a thread-safe queue with a max size to prevent memory issues
        self.data_queue = queue.Queue(maxsize=1000) # Increased maxsize slightly
        self._is_running = threading.Event()
        
        self.active_channels = []
        # A lock is crucial for safely modifying the channel list from different threads
        self._channel_lock = threading.Lock()

    @abstractmethod
    def run(self):
        """The main threaded loop for acquiring data."""
        pass

    @abstractmethod
    def list_available_channels(self) -> list:
        """Returns a list of available physical channels."""
        pass

    def add_channel(self, channel_id):
        """Adds a channel to the list of active channels to be acquired."""
        with self._channel_lock:
            if channel_id not in self.active_channels:
                self.active_channels.append(channel_id)
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

