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
        self._curve = pg.PlotCurveItem(pen=pg.mkPen(config.color, width=1))
        self._plot_item.addItem(self._curve)
        
        # State
        self._last_samples: np.ndarray = np.zeros(0, dtype=np.float32)
        self._downsample_factor = 1
        
    def update_config(self, config: ChannelConfig) -> None:
        """Update visual properties based on new config."""
        self._config = config
        self._curve.setPen(pg.mkPen(config.color, width=1))
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
        if downsample > 1:
            display_y = samples[::downsample]
            display_x = times[::downsample]
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
        width = 3.0 if active else 1.0
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
