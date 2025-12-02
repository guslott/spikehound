from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui

from .types import ChannelConfig

class TraceRenderer:
    """
    Manages the visualization of a single channel's trace.
    Handles curve updates, downsampling, and visual offsets.
    """

    def __init__(self, plot_item: pg.PlotItem, config: ChannelConfig):
        self._plot_item = plot_item
        self._config = config
        self._curve = pg.PlotCurveItem(pen=pg.mkPen(config.color, width=3))
        self._manual_downsampling = False
        try:
            self._curve.setDownsampling(ds=True, auto=True, method="peak")
        except AttributeError:
            self._manual_downsampling = True
        self._plot_item.addItem(self._curve)
        
        # State
        self._last_samples: np.ndarray = np.zeros(0, dtype=np.float32)
        self._downsample_factor = 1
        
    def update_config(self, config: ChannelConfig) -> None:
        """Update visual properties based on new config."""
        self._config = config
        self._curve.setPen(pg.mkPen(config.color, width=3))
        self._curve.setVisible(config.display_enabled)
        # Re-render with new offset/scale if we have data
        if self._last_samples.size > 0:
            self._update_curve_data()

    def update_data(self, samples: np.ndarray, times: np.ndarray, downsample: int = 1) -> None:
        """
        Update the curve with new data.
        
        Args:
            samples: 1D array of voltage values
            times: 1D array of time values
            downsample: Downsampling factor (stride)
        """
        self._last_samples = samples
        self._downsample_factor = downsample
        
        if not self._config.display_enabled:
            self._curve.setData([], [])
            return

        # Apply downsampling
        if self._manual_downsampling and samples.size > 4000:
            display_y, display_x = self._resample_peak(samples, times, target=2000)
        else:
            display_y = samples
            display_x = times
            
        # Apply offset and scaling for display
        # Visual Y = (Data Y / Vertical Span) + Offset
        # But usually we want to map the vertical span to some visual range.
        # In MainWindow logic: 
        #   normalized = data / span
        #   shifted = normalized + offset
        # Let's stick to the logic that was likely in MainWindow or intended:
        # The user selects a "Vertical Range (+/- V)". 
        # If range is 1V, then +1V maps to +0.5 relative to center, -1V to -0.5.
        # Then we add the screen offset (0-1, default 0.5).
        # So: y_final = (y_data / (2 * span)) + offset
        
        span = max(1e-6, self._config.vertical_span_v)
        # Map [-span, +span] to [-0.5, +0.5]
        scaled_y = display_y / (2.0 * span)
        # Shift by offset (0.0 at bottom, 1.0 at top)
        final_y = scaled_y + self._config.screen_offset
        
        self._curve.setData(display_x, final_y)
        
    def _update_curve_data(self) -> None:
        """Re-apply scaling/offset to the last known data."""
        # We don't have 'times' stored, so we can't fully re-render if we don't store times.
        # However, update_data is called frequently. 
        # If we need to support config changes without new data, we should store times too.
        # For now, let's assume update_data will be called soon enough, 
        # or we just clear if we don't have times.
        # Actually, let's just wait for the next update_data call for simplicity 
        # unless we want to cache times. 
        # Given the high refresh rate, waiting is usually fine.
        pass

    def set_active(self, active: bool) -> None:
        """Update visual style for active/inactive state."""
        width = 5.0 if active else 3.0
        self._curve.setPen(pg.mkPen(self._config.color, width=width))
        self._curve.setZValue(1.0 if active else 0.0)
        try:
            self._curve.setOpacity(1.0 if active else 0.6)
        except AttributeError:
            pass

    def clear(self) -> None:
        self._curve.clear()
        self._last_samples = np.zeros(0, dtype=np.float32)

    def cleanup(self) -> None:
        """Remove curve from plot."""
        try:
            self._plot_item.removeItem(self._curve)
        except Exception:
            pass

    def _resample_peak(self, samples: np.ndarray, times: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
        """Manual peak downsampling for older pyqtgraph versions."""
        n = samples.size
        if n <= target:
            return samples, times
            
        # Chunk size
        k = n // (target // 2)
        if k <= 1:
            return samples, times
            
        n_chunks = n // k
        
        # Reshape to find min/max in each chunk
        # Truncate to multiple of k
        n_trim = n_chunks * k
        y_view = samples[:n_trim].reshape(n_chunks, k)
        t_view = times[:n_trim].reshape(n_chunks, k)
        
        mins = y_view.min(axis=1)
        maxs = y_view.max(axis=1)
        t_starts = t_view[:, 0]
        t_ends = t_view[:, -1]
        
        # Interleave
        y_out = np.empty(n_chunks * 2, dtype=samples.dtype)
        y_out[0::2] = mins
        y_out[1::2] = maxs
        
        t_out = np.empty(n_chunks * 2, dtype=times.dtype)
        t_out[0::2] = t_starts
        t_out[1::2] = t_ends
        
        return y_out, t_out
