import sys
import time
import pytest
import numpy as np
from daq.labjack_device import LabJackDevice

# This test requires the actual labjack-ljm library to be installed.
# It uses the special "-2" identifier for Demo Mode.

def test_labjack_demo_mode():
    try:
        import labjack.ljm as ljm
    except ImportError:
        pytest.skip("labjack-ljm not installed")

    print("LJM library found. Attempting to open Demo Mode device...")
    
    # 1. Initialize
    dev = LabJackDevice()
    
    # 2. Open Demo Device
    # Identifier "-2" is LJM_DEMO_MODE
    try:
        dev.open("-2")
    except Exception as e:
        pytest.fail(f"Failed to open Demo Mode device: {e}")
        
    print("Opened Demo Device.")
    
    # 3. Configure
    # Demo mode supports standard rates.
    try:
        config = dev.configure(sample_rate=1000, channels=[0, 1], chunk_size=100)
        print(f"Configured: {config}")
    except Exception as e:
        pytest.fail(f"Failed to configure: {e}")
        
    # 4. Start
    try:
        dev.start()
        print("Stream started.")
    except Exception as e:
        pytest.fail(f"Failed to start stream: {e}")
        
    # 5. Consume data
    # Collect for 2 seconds
    start_time = time.time()
    total_frames = 0
    
    try:
        while time.time() - start_time < 2.0:
            ptr = dev.data_queue.get(timeout=1.0)
            total_frames += ptr.length
            
            # Verify data
            rb = dev.get_buffer()
            data = rb.read(ptr.start_index, ptr.length)
            assert data.shape == (2, ptr.length)
            
            # In demo mode, data is usually random or simulated sine waves
            # Just check it's not all zeros (though it might be if configured that way)
            # print(f"Received {ptr.length} frames")
            
    except Exception as e:
        pytest.fail(f"Error during streaming: {e}")
    finally:
        dev.stop()
        dev.close()
        print("Closed device.")
        
    assert total_frames > 0, "No data received from Demo Mode device"
    print(f"Total frames received: {total_frames}")

if __name__ == "__main__":
    test_labjack_demo_mode()
