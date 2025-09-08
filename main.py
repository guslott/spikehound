import time
import queue
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from daq.simulated_source import SimulatedPhysiologySource

# --- Configuration ---
SAMPLE_RATE = 10000  # Hz
CHUNK_SIZE = 1000      # Samples per chunk
NUM_CHANNELS = 1     # We'll start with one channel
PLOT_DURATION_S = 0.1 # How many seconds of data to display on the scope

# Calculate the total number of samples to store for the plot
PLOT_SAMPLES = int(PLOT_DURATION_S * SAMPLE_RATE)

# --- Global variables ---
sim_source = None
# Use a numpy array as a circular buffer for performance
data_buffer = np.zeros((PLOT_SAMPLES, NUM_CHANNELS))
# Create the time vector for the x-axis once
time_vector = np.linspace(0, PLOT_DURATION_S, PLOT_SAMPLES)

# --- Matplotlib Setup ---
fig, ax = plt.subplots()
# Initialize plot with empty data, it will be populated by the init function
line, = ax.plot([], [], lw=1)

def init_plot():
    """Initializes the plot aesthetics and data."""
    ax.set_xlim(0, PLOT_DURATION_S)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Live Simulated Electrophysiology")
    ax.grid(True)
    line.set_data(time_vector, data_buffer[:, 0])
    return line,

def update_plot(frame):
    """
    This function is called by FuncAnimation to update the plot.
    """
    global data_buffer
    
    new_data_chunks = []
    # Drain the queue of all available data chunks
    while True:
        try:
            data_chunk = sim_source.data_queue.get_nowait()
            new_data_chunks.append(data_chunk)
        except queue.Empty:
            break # No more data in the queue
    
    if not new_data_chunks:
        return line, # No new data, no need to redraw

    # Concatenate all new chunks into a single array
    new_data = np.concatenate(new_data_chunks)
    num_new_samples = new_data.shape[0]

    # Roll the buffer to the left to make space for the new data
    data_buffer = np.roll(data_buffer, -num_new_samples, axis=0)
    # Insert the new data at the end of the buffer
    data_buffer[-num_new_samples:, :] = new_data
    
    # Update the plot line with the new buffer content
    line.set_ydata(data_buffer[:, 0]) # Plot the first channel
    
    return line,

def on_close(event):
    """Callback function for when the plot window is closed."""
    print("Plot window closed. Stopping DAQ source...")
    if sim_source and sim_source.is_running():
        sim_source.stop()

def main():
    """
    Main entry point for the application.
    """
    global sim_source
    
    # Define which channels we want to acquire from the start.
    initial_channels = ['Simulated Extracellular 1']
    
    # Instantiate and configure the data source.
    sim_source = SimulatedPhysiologySource(
        sample_rate=SAMPLE_RATE, 
        chunk_size=CHUNK_SIZE
    )
    print(sim_source.list_available_channels())
    sim_source.add_channel(initial_channels[0])
    

    print("--- SpikeHound Live Plotter ---")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Chunk Size: {CHUNK_SIZE} samples")
    print(f"Data Rate: {SAMPLE_RATE / CHUNK_SIZE} chunks/sec")
    print(f"Plot Refresh Interval: 50 ms (~20 Hz)")
    print("---------------------------------\n")

    # Start the data acquisition thread
    sim_source.start()
    
    # Connect the close event to our shutdown function
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Create the animation object. 
    # interval=50 means the plot will try to refresh every 50ms.
    # blit=True is a performance optimization.
    ani = animation.FuncAnimation(
        fig, 
        update_plot, 
        init_func=init_plot, 
        blit=True, 
        interval=50,
        save_count=50 # Internal buffer size for FuncAnimation
    )
    
    # Show the plot. This call is blocking and will run until the window is closed.
    plt.show()
    
    print("Application finished.")

if __name__ == "__main__":
    main()
