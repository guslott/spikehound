import sys
import threading
import time
import queue
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Mock labjack.ljm before importing the device
mock_ljm = MagicMock()
mock_ljm.constants.dtANY = 0
mock_ljm.constants.ctANY = 0
mock_ljm.constants.ctTCP = 2
mock_ljm.constants.ctETHERNET = 3
mock_ljm.constants.ctWIFI = 4
mock_ljm.constants.ctUSB = 1

class MockLJMError(Exception):
    pass
mock_ljm.LJMError = MockLJMError

# Mock listAll return values
# num_found, aDeviceTypes, aConnectionTypes, aSerialNumbers, aIPAddresses
mock_ljm.listAll.return_value = (
    1, [7], [1], [470012345], [0]
)
mock_ljm.numberToIP.return_value = "0.0.0.0"
mock_ljm.namesToAddresses.return_value = ([0, 2], [0, 0]) # Address for AIN0, AIN1
mock_ljm.eStreamStart.return_value = 1000.0 # Actual scan rate

# Mock eStreamRead to return some data
# (aData, DeviceScanBacklog, LJMScanBacklog)
# Return 10 scans of 2 channels = 20 samples
mock_data = [float(i) for i in range(20)]
mock_ljm.eStreamRead.return_value = (mock_data, 0, 0)

# Setup parent mock
mock_labjack = MagicMock()
mock_labjack.ljm = mock_ljm

# Apply the mock to sys.modules
with patch.dict(sys.modules, {"labjack": mock_labjack, "labjack.ljm": mock_ljm}):
    from daq.labjack_device import LabJackDevice

def test_labjack_lifecycle():
    # 1. List devices
    devices = LabJackDevice.list_available_devices()
    assert len(devices) == 1
    assert devices[0].id == "470012345"
    assert devices[0].details["driver_name"] == "LabJack T-Series"

    # 2. Initialize
    dev = LabJackDevice()
    
    # 3. Open
    dev.open(devices[0].id)
    mock_ljm.openS.assert_called_with("ANY", "ANY", "470012345")
    
    # 4. Configure
    # 1000 Hz, 2 channels (0, 1)
    config = dev.configure(sample_rate=1000, channels=[0, 1], chunk_size=10)
    assert config.sample_rate == 1000
    assert len(config.channels) == 2
    
    # 5. Start
    dev.start()
    mock_ljm.eStreamStart.assert_called()
    assert dev.running
    
    # 6. Consume data
    # We expect ChunkPointers in the queue
    try:
        ptr = dev.data_queue.get(timeout=2.0)
        assert ptr.length > 0
        
        # Verify data in ring buffer
        rb = dev.get_buffer()
        # We can't easily check the exact data without knowing where the pointer points vs the mock data generation
        # But we can check that rb.read(ptr.start_index, ptr.length) returns valid data
        data = rb.read(ptr.start_index, ptr.length)
        assert data.shape == (2, ptr.length) # (channels, frames)
        
    except queue.Empty:
        pytest.fail("Data queue empty")
        
    # 7. Stop
    dev.stop()
    mock_ljm.eStreamStop.assert_called()
    assert not dev.running
    
    # 8. Close
    dev.close()
    mock_ljm.close.assert_called()

if __name__ == "__main__":
    # Manually run the test function if executed as script
    try:
        test_labjack_lifecycle()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
