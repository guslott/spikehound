from __future__ import annotations

import logging
import threading
import time
from typing import List, Sequence, Any, Optional, Dict

import numpy as np

from shared.models import (
    Capabilities,
    ChannelInfo,
    DeviceInfo,
    ActualConfig,
)
from daq.base_device import BaseDevice

# Try to import labjack-ljm
try:
    from labjack import ljm
    LJM_AVAILABLE = True
except ImportError:
    LJM_AVAILABLE = False
    ljm = None

logger = logging.getLogger(__name__)


class LabJackDevice(BaseDevice):
    """
    Driver for LabJack T7/T4 devices using the LJM library.
    Supports streaming analog inputs.
    """

    @classmethod
    def device_class_name(cls) -> str:
        return "LabJack T-Series"

    @classmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        if not LJM_AVAILABLE:
            return []
        
        devices = []
        try:
            # List all connected T7 and T4 devices
            # dtANY=0, ctANY=0 -> Scan all supported devices and connections
            # But usually we want T7/T4 specifically. LJM_dtANY is fine.
            # LJM_ListAll returns arrays of info
            num_found, aDeviceTypes, aConnectionTypes, aSerialNumbers, aIPAddresses = ljm.listAll(
                ljm.constants.dtANY, ljm.constants.ctANY
            )

            for i in range(num_found):
                dt = aDeviceTypes[i]
                ct = aConnectionTypes[i]
                sn = aSerialNumbers[i]
                
                # Filter for T4 (4) and T7 (7)
                if dt not in (4, 7):
                    continue
                    
                model = "T7" if dt == 7 else "T4"
                conn_type = "USB"
                if ct == ljm.constants.ctTCP:
                    conn_type = "TCP"
                elif ct == ljm.constants.ctETHERNET:
                    conn_type = "Ethernet"
                elif ct == ljm.constants.ctWIFI:
                    conn_type = "WiFi"
                
                # Create a unique ID that can be used to open the device
                # We'll use the serial number as the primary ID
                dev_id = str(sn)
                
                devices.append(DeviceInfo(
                    id=dev_id,
                    name=f"LabJack {model} ({sn})",
                    vendor="LabJack",
                    details={
                        "driver_name": cls.device_class_name(),
                        "device_type": dt,
                        "connection_type": ct,
                        "ip_address": ljm.numberToIP(aIPAddresses[i]) if ct != ljm.constants.ctUSB else None
                    }
                ))
                
        except Exception as e:
            logger.error(f"Error listing LabJack devices: {e}")
            
        return devices

    def __init__(self, queue_maxsize: int = 64) -> None:
        super().__init__(queue_maxsize)
        self._handle: Optional[int] = None
        self._scan_rate: float = 0.0
        self._scans_per_read: int = 0
        self._stream_thread: Optional[threading.Thread] = None
        self._active_addresses: List[int] = []

    def get_capabilities(self, device_id: str) -> Capabilities:
        # T7 supports a wide range of rates.
        # Max is ~100k samples/s aggregate.
        # We'll expose a reasonable set of standard rates.
        return Capabilities(
            sample_rates=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
            channel_counts=[1, 2, 4, 8, 14], # T7 has 14 AINs
            bit_depths=[16], # Effective resolution varies, but 16 is a safe placeholder
            formats=["float32"] # LJM returns floats
        )

    def list_available_channels(self, device_id: str) -> List[ChannelInfo]:
        # T7 has AIN0-AIN13 available on screw terminals/DB37
        # T4 has AIN0-AIN3 (HV) and AIN4-AIN11 (LV)
        # For simplicity, we'll list AIN0-AIN13 for now.
        # In a real implementation, we might query the device type to be more specific.
        
        channels = []
        for i in range(14):
            channels.append(ChannelInfo(
                id=i,
                name=f"AIN{i}"
            ))
        return channels

    def _open_impl(self, device_id: str) -> None:
        if not LJM_AVAILABLE:
            raise RuntimeError("labjack-ljm library not installed")
        
        try:
            # Open by serial number
            # dtANY, ctANY, identifier=serial_number
            self._handle = ljm.openS("ANY", "ANY", device_id)
            
            # Get device info to confirm
            info = ljm.getHandleInfo(self._handle)
            logger.info(f"Opened LabJack device: {info}")
            
        except ljm.LJMError as e:
            raise RuntimeError(f"Failed to open LabJack device {device_id}: {e}")

    def _close_impl(self) -> None:
        if self._handle is not None:
            try:
                ljm.close(self._handle)
            except Exception as e:
                logger.error(f"Error closing LabJack handle: {e}")
            self._handle = None

    def _configure_impl(
        self,
        sample_rate: int,
        channels: Sequence[int],
        chunk_size: int,
        **options: Any,
    ) -> ActualConfig:
        if self._handle is None:
            raise RuntimeError("Device not open")

        # 1. Setup scan list
        # Map channel IDs (0, 1, ...) to LJM names/addresses
        # AIN0 is address 0, AIN1 is 2, etc? No, LJM names map to addresses.
        # AIN0=0, AIN1=2, ... wait, LJM addresses for AIN are 0, 2, 4... for T7?
        # Actually, let's use names to be safe, or look up addresses.
        # LJM names: "AIN0", "AIN1", ...
        # We need to convert these to addresses for streamStart
        
        ljm_names = [f"AIN{cid}" for cid in channels]
        aScanList = ljm.namesToAddresses(len(ljm_names), ljm_names)[0]
        self._active_addresses = aScanList
        
        # 2. Configure stream
        # We don't actually start stream here, just validate and store params
        # But LJM stream start takes the rate and updates it.
        # We'll do a dry run or just trust the requested rate for now, 
        # and actual rate will be confirmed in start().
        
        self._scan_rate = float(sample_rate)
        
        # Scans per read: how many scans to fetch at once.
        # chunk_size is a good target.
        self._scans_per_read = chunk_size
        
        # Check if valid rate (rough check)
        if sample_rate > 100000:
             logger.warning(f"Sample rate {sample_rate} might be too high for LabJack T7")

        return ActualConfig(
            sample_rate=sample_rate,
            channels=list(channels),
            chunk_size=chunk_size,
            dtype="float32"
        )

    def _start_impl(self) -> None:
        if self._handle is None:
            raise RuntimeError("Device not open")
            
        # Start the stream
        try:
            scan_rate = self._scan_rate
            scans_per_read = self._scans_per_read
            num_addresses = len(self._active_addresses)
            
            # Ensure stream is stopped before starting
            try:
                ljm.eStreamStop(self._handle)
            except ljm.LJMError:
                pass
                
            # LJM_eStreamStart(Handle, ScansPerRead, NumAddresses, aScanList, ScanRate)
            # Returns actual scan rate
            actual_scan_rate = ljm.eStreamStart(
                self._handle,
                scans_per_read,
                num_addresses,
                self._active_addresses,
                scan_rate
            )
            
            logger.info(f"Stream started: requested {scan_rate} Hz, actual {actual_scan_rate} Hz")
            self._scan_rate = actual_scan_rate
            
            # Start the consumer thread
            self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._stream_thread.start()
            
        except ljm.LJMError as e:
            raise RuntimeError(f"Failed to start stream: {e}")

    def _stop_impl(self) -> None:
        # The thread checks self._stop_event, so we just wait for it
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            if self._stream_thread.is_alive():
                logger.warning("Stream thread did not exit cleanly")
            self._stream_thread = None
            
        if self._handle is not None:
            try:
                ljm.eStreamStop(self._handle)
            except ljm.LJMError as e:
                # If it's already stopped, that's fine
                if "STREAM_NOT_RUNNING" not in str(e):
                    logger.error(f"Error stopping stream: {e}")

    def _stream_loop(self) -> None:
        """
        Continuously read from LJM stream and emit to the ring buffer.
        """
        if self._handle is None:
            return

        num_addresses = len(self._active_addresses)
        scans_per_read = self._scans_per_read
        
        # Pre-allocate buffer for LJM to write into? 
        # LJM python wrapper returns the data as a list/array.
        
        logger.debug("Entering stream loop")
        
        while not self._stop_event.is_set():
            try:
                # LJM_eStreamRead(Handle) -> (aData, DeviceScanBacklog, LJMScanBacklog)
                # Note: The python wrapper might behave slightly differently depending on version.
                # Usually: ret = ljm.eStreamRead(handle)
                # ret is (aData, deviceScanBacklog, ljmScanBacklog)
                
                ret = ljm.eStreamRead(self._handle)
                aData = ret[0]
                device_backlog = ret[1]
                ljm_backlog = ret[2]
                
                # Check for skipped scans / overflows
                if device_backlog > 1000 or ljm_backlog > 1000:
                    self.note_xrun()
                
                # aData is a flat list of floats: [scan0_ch0, scan0_ch1, ..., scan1_ch0, ...]
                # We need to reshape it to (frames, channels)
                # len(aData) should be scans_per_read * num_addresses
                
                # Convert to numpy array
                data_flat = np.array(aData, dtype=np.float32)
                
                # Reshape: (scans, channels)
                # LJM returns interleaved data
                num_samples = len(data_flat)
                num_scans = num_samples // num_addresses
                
                if num_scans > 0:
                    data_chunk = data_flat.reshape((num_scans, num_addresses))
                    self.emit_array(data_chunk)
                
            except ljm.LJMError as e:
                if self._stop_event.is_set():
                    break
                    
                # Check for specific errors like NO_SCANS_RETURNED (if configured)
                # or STREAM_NOT_RUNNING
                err_str = str(e)
                if "LJME_NO_SCANS_RETURNED" in err_str:
                    continue
                elif "LJME_STREAM_NOT_RUNNING" in err_str:
                    logger.error("Stream stopped unexpectedly")
                    break
                else:
                    logger.error(f"LJM stream error: {e}")
                    self.note_xrun()
                    # Back off slightly to avoid hammering
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Unexpected error in stream loop: {e}")
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
