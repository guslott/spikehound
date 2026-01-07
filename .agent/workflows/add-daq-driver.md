---
description: Add a new DAQ device driver for hardware data acquisition
---

# Add a New DAQ Driver

Follow these steps to add support for a new data acquisition device.

## Prerequisites
- Device SDK/library documentation
- Python bindings for the device (or create using ctypes/cffi)

## Steps

1. **Read the DAQ documentation**
   - Review `daq/daq_readme.md` for the complete contract and patterns
   - Study `daq/simulated_source.py` as the reference implementation

2. **Create the driver file**
   ```
   daq/my_device_source.py
   ```

3. **Implement the driver class**
   - Subclass `BaseDevice` from `daq/base_device.py`
   - Implement these 5 hooks:
     - `_open_impl(self, device_id: str)` - Open device handle
     - `_close_impl(self)` - Release device handle  
     - `_configure_impl(self, sample_rate, channels, chunk_size, **opts)` - Configure hardware
     - `_start_impl(self)` - Start producer thread/callback
     - `_stop_impl(self)` - Stop producer thread
   - Implement discovery methods:
     - `list_available_devices()` - Return list of `DeviceInfo`
     - `get_capabilities(device_id)` - Return `Capabilities`
     - `list_available_channels(device_id)` - Return list of `ChannelInfo`

4. **Emit data using `emit_array()`**
   ```python
   # In your producer loop:
   data = np.zeros((chunk_size, n_channels), dtype=np.float32)
   self.emit_array(data, device_time=optional_hw_timestamp)
   ```

5. **Register the driver in `daq/registry.py`**
   ```python
   from .my_device_source import MyDeviceSource
   
   DRIVER_CLASSES.append(MyDeviceSource)
   ```

6. **Update `SpikeHound.spec` for binary builds**
   
   PyInstaller cannot auto-discover dynamically loaded modules. Add your driver to the `hiddenimports` list:
   ```python
   hiddenimports=[
       "daq.simulated_source",
       "daq.backyard_brains",
       "daq.soundcard_source",
       "daq.file_source",
       "daq.my_device_source",  # <-- Add your new driver here
       ...
   ],
   ```
   > **Note**: Without this step, the driver will work when running from source (`python main.py`) but will be missing from packaged builds (`pyinstaller --clean --noconfirm SpikeHound.spec`).

7. **Test the driver**
   - Run the application and verify device appears in dropdown
   - Connect and verify data streams correctly
   - Check `stats()` for drops/xruns

## Verification Checklist
- [ ] Device appears in device list
- [ ] Capabilities show correct sample rates and channels
- [ ] Data streams without errors
- [ ] Clean shutdown (no hanging threads)
- [ ] Stats show minimal drops/xruns
