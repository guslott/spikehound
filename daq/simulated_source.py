# daq/simulated_source.py
import numpy as np
import time
import queue
from .base_source import DataAcquisitionSource

class SimulatedPhysiologySource(DataAcquisitionSource):
    """
    A simulated data source that generates physiological-like signals.
    It now respects the active channel list and provides timestamps.
    """
    def __init__(self, sample_rate: int, chunk_size: int):
        super().__init__(sample_rate, chunk_size)
            
        # This is where we will eventually build our signal generation logic
        self._spike_templates = {} # To be populated later
        self._noise_level = 0.01

    def list_available_channels(self) -> list:
        """Returns the list of channels this simulated device can produce."""
        return ['Simulated Extracellular 1', 'Simulated Extracellular 2', 'Simulated Intracellular']

    def run(self):
        """
        The main loop for generating simulated data.
        """
        chunk_duration = self.chunk_size / self.sample_rate

        # This loop generates data until the thread is stopped
        while self.is_running():
            start_time = time.perf_counter() # Use a high-precision clock

            # Safely get the number of active channels
            with self._channel_lock:
                num_channels = len(self.active_channels)

            if num_channels == 0:
                time.sleep(chunk_duration) # Still sleep to avoid a busy loop
                continue

            # --- Signal Generation Logic Will Go Here ---
            # For now, just generate white noise for the correct number of channels.
            data_chunk = np.random.randn(self.chunk_size, num_channels) * self._noise_level
            
            # Put data into the queue without blocking
            try:
                self.data_queue.put_nowait(data_chunk)
            except queue.Full:
                print("Warning: DAQ queue is full. Dropping oldest data chunk.")
                # To prevent backlog, discard the oldest chunk and add the new one
                self.data_queue.get_nowait() 
                self.data_queue.put_nowait(data_chunk)



            # Wait precisely to maintain the sample rate
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            sleep_time = chunk_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)