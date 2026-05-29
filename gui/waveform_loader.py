from __future__ import annotations
from typing import TYPE_CHECKING
from collections import Counter
import numpy as np
from PySide6 import QtCore, QtGui

class WaveformLoader(QtCore.QThread):
    """
    Background thread to gather and align waveform data for visualization.
    Avoids blocking the GUI when processing thousands of events.
    """
    data_ready = QtCore.Signal(str, object, list, object)  # class_name, waveforms, aligned_samples, median
    progress = QtCore.Signal(int)
    
    def __init__(
        self, 
        event_ids: list[int], 
        event_details: dict[int, dict[str, object]], 
        class_name: str, 
        class_color: QtGui.QColor,
        parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.event_ids = event_ids
        # Note: We access event_details by reference. Since this is read-only for existing keys
        # and we are in a GIL-protected environment, this is generally safe for this purpose.
        self.event_details = event_details
        self.class_name = class_name
        self.class_color = class_color
        self._is_canceled = False
        
    def cancel(self) -> None:
        self._is_canceled = True
        
    def run(self) -> None:
        waveforms: list[tuple[np.ndarray, np.ndarray]] = []
        total = len(self.event_ids)
        
        # 1. Gather raw waveforms
        for idx, event_id in enumerate(self.event_ids):
            if self._is_canceled:
                return
            if idx % 100 == 0:
                self.progress.emit(int((idx / total) * 50))
                
            details = self.event_details.get(event_id)
            if not details:
                continue
            times = details.get("times")
            samples = details.get("samples")
            if times is None or samples is None:
                continue
                
            arr_t = np.asarray(times, dtype=np.float64)
            arr_s = np.asarray(samples, dtype=np.float32)
            if arr_t.size == 0 or arr_s.size == 0 or arr_t.size != arr_s.size:
                continue
                
            # Baseline correct
            baseline = float(np.median(arr_s)) if arr_s.size else 0.0
            s_rel = arr_s - baseline
            # Time relative to start
            t_rel = arr_t - arr_t[0]
            waveforms.append((t_rel, s_rel))
            
        if not waveforms:
            self.data_ready.emit(self.class_name, self.class_color, [], None)
            return
            
        # 2. Align and calculate median (common length)
        self.progress.emit(60)
        lengths = [min(t.size, s.size) for t, s in waveforms]
        if not lengths:
            self.data_ready.emit(self.class_name, self.class_color, [], None)
            return
            
        # Find most common length
        length_counts = Counter(lengths)
        if not length_counts:
            target_len = 0
        else:
            target_len = max(length_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            
        aligned_samples: list[np.ndarray] = []
        plot_time_axis: np.ndarray | None = None
        
        processed_count = 0
        total_waveforms = len(waveforms)
        
        for times, samples in waveforms:
            processed_count += 1
            if processed_count % 100 == 0:
                if self._is_canceled:
                    return
                # Map 60-90% progress
                p = 60 + int((processed_count / total_waveforms) * 30)
                self.progress.emit(p)
                
            if times.size < target_len or samples.size < target_len or target_len <= 0:
                continue
                
            # Use copy only if needed, but here we are creating new arrays anyway
            # We only need the samples for the median and plotting
            s_trim = samples[:target_len]
            aligned_samples.append(s_trim)
            
            if plot_time_axis is None:
                plot_time_axis = times[:target_len]
        
        self.progress.emit(95)
        median_waveform = None
        if aligned_samples:
            stack = np.stack(aligned_samples, axis=0)
            median_waveform = np.median(stack, axis=0)
            
        self.progress.emit(100)
        self.data_ready.emit(self.class_name, self.class_color, waveforms, median_waveform)
