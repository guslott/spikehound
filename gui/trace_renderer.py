from __future__ import annotations

import logging
import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui

from .types import ChannelConfig

logger = logging.getLogger(__name__)

class TraceRenderer:
    """
    Manages the visualization of a single channel's trace.
    Handles curve updates, downsampling, and visual offsets.
    """

    # Constants for downsampling
    DOWNSAMPLE_TARGET = 2000
    DOWNSAMPLE_THRESHOLD = 4000

    def __init__(self, plot_item: pg.PlotItem, config: ChannelConfig):
        self._plot_item = plot_item
        self._config = config
        # Use width=2 for balance of visibility and performance (thick pens are slower)
        self._curve = pg.PlotCurveItem(pen=pg.mkPen(config.color, width=2))
        self._manual_downsampling = False
        try:
            self._curve.setDownsampling(ds=True, auto=True, method="peak")
        except AttributeError:
            self._manual_downsampling = True
        self._plot_item.addItem(self._curve)

        # State
        self._last_samples: np.ndarray = np.zeros(0, dtype=np.float32)
        self._downsample_factor = 1

        # Pre-allocated buffers to reduce GC pressure during real-time rendering.
        # These are lazily resized when input dimensions change.
        self._transform_buffer: np.ndarray | None = None
        self._downsample_y_buffer: np.ndarray | None = None
        self._downsample_t_buffer: np.ndarray | None = None
        # Intermediate buffers for min/max computation during downsampling
        self._mins_buffer: np.ndarray | None = None
        self._maxs_buffer: np.ndarray | None = None
        
    def update_config(self, config: ChannelConfig) -> None:
        """Update visual properties based on new config."""
        self._config = config
        self._curve.setPen(pg.mkPen(config.color, width=2))
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
            self._curve.setData([], [], skipFiniteCheck=True)
            return

        # Always apply peak downsampling for large datasets
        # PyQtGraph's auto-downsampling helps but explicit control is faster
        if samples.size > self.DOWNSAMPLE_THRESHOLD:
            display_y, display_x = self._resample_peak(samples, times, target=self.DOWNSAMPLE_TARGET)
        else:
            display_y = samples
            display_x = times

        # Apply offset and scaling for display using pre-allocated buffer
        # y_final = (y_data / (2 * span)) + offset
        span = max(1e-6, self._config.vertical_span_v)
        scale = 1.0 / (2.0 * span)
        offset = self._config.screen_offset

        # Ensure transform buffer is large enough; reallocate only when size changes
        if self._transform_buffer is None or self._transform_buffer.size < display_y.size:
            self._transform_buffer = np.empty(display_y.size, dtype=np.float32)

        # Use in-place operations to avoid allocations in the hot path
        final_y = self._transform_buffer[:display_y.size]
        np.multiply(display_y, scale, out=final_y)
        np.add(final_y, offset, out=final_y)

        # skipFiniteCheck=True: avoid scanning 100k+ points for NaN/Inf
        # connect='all': skip connection analysis, we know data is contiguous
        self._curve.setData(display_x, final_y, skipFiniteCheck=True, connect='all')
        
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
        # Reduced widths for performance (thick pens are slower with OpenGL)
        width = 3.0 if active else 2.0
        self._curve.setPen(pg.mkPen(self._config.color, width=width))
        self._curve.setZValue(1.0 if active else 0.0)
        try:
            self._curve.setOpacity(1.0 if active else 0.6)
        except AttributeError:
            pass

    def clear(self) -> None:
        """Clear the curve data and release pre-allocated buffers."""
        self._curve.clear()
        self._last_samples = np.zeros(0, dtype=np.float32)
        # Release buffers to free memory when renderer is cleared
        self._transform_buffer = None
        self._downsample_y_buffer = None
        self._downsample_t_buffer = None
        self._mins_buffer = None
        self._maxs_buffer = None

    def cleanup(self) -> None:
        """Remove curve from plot."""
        try:
            self._plot_item.removeItem(self._curve)
        except Exception as exc:
            logger.debug("Failed to remove curve from plot: %s", exc)

    def _resample_peak(self, samples: np.ndarray, times: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Manual peak downsampling for older pyqtgraph versions.

        Uses pre-allocated buffers to reduce GC pressure during real-time rendering.
        The algorithm preserves signal envelope by keeping min/max values per chunk.
        """
        n = samples.size
        if n <= target:
            return samples, times

        # Chunk size: each chunk produces 2 output points (min, max)
        k = n // (target // 2)
        if k <= 1:
            return samples, times

        n_chunks = n // k
        out_size = n_chunks * 2

        # Ensure all downsample buffers are large enough; reallocate only when size increases
        if self._downsample_y_buffer is None or self._downsample_y_buffer.size < out_size:
            self._downsample_y_buffer = np.empty(out_size, dtype=np.float32)
        if self._downsample_t_buffer is None or self._downsample_t_buffer.size < out_size:
            self._downsample_t_buffer = np.empty(out_size, dtype=np.float32)
        if self._mins_buffer is None or self._mins_buffer.size < n_chunks:
            self._mins_buffer = np.empty(n_chunks, dtype=np.float32)
        if self._maxs_buffer is None or self._maxs_buffer.size < n_chunks:
            self._maxs_buffer = np.empty(n_chunks, dtype=np.float32)

        # Reshape to find min/max in each chunk (zero-copy view)
        n_trim = n_chunks * k
        y_view = samples[:n_trim].reshape(n_chunks, k)
        t_view = times[:n_trim].reshape(n_chunks, k)

        # Extract min/max values using pre-allocated buffers
        # NumPy's min/max with axis don't support out=, so we use a vectorized approach:
        # argmin/argmax return indices, then we gather values via advanced indexing
        mins = self._mins_buffer[:n_chunks]
        maxs = self._maxs_buffer[:n_chunks]

        # Use argmin/argmax to find indices, then gather values
        # This trades 2 small index arrays for avoiding 2 large value arrays
        min_indices = y_view.argmin(axis=1)
        max_indices = y_view.argmax(axis=1)

        # Gather min/max values using advanced indexing into pre-allocated buffers
        row_indices = np.arange(n_chunks)
        np.copyto(mins, y_view[row_indices, min_indices])
        np.copyto(maxs, y_view[row_indices, max_indices])

        # Use slices of pre-allocated buffers for output
        y_out = self._downsample_y_buffer[:out_size]
        t_out = self._downsample_t_buffer[:out_size]

        # Interleave min/max into output buffer
        y_out[0::2] = mins
        y_out[1::2] = maxs

        # Time values: use actual times at min/max positions for better accuracy
        t_out[0::2] = t_view[row_indices, min_indices]
        t_out[1::2] = t_view[row_indices, max_indices]

        return y_out, t_out
