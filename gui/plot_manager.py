"""PlotManager - Manages waveform visualization and data flow for the oscilloscope.

Extracted from MainWindow to provide a focused component for renderer lifecycle,
data processing, and dispatcher integration.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from .trace_renderer import TraceRenderer
from .trigger_controller import TriggerController
from .types import ChannelConfig

logger = logging.getLogger(__name__)


class PlotManager(QtCore.QObject):
    """Manages waveform visualization, renderer lifecycle, and data flow.
    
    Responsibilities:
    - Manage TraceRenderer instances for each channel
    - Process incoming data from the dispatcher
    - Handle streaming and trigger modes
    - Track plot refresh rate and chunk statistics
    """
    
    # Signals to notify MainWindow of state changes
    statusUpdated = QtCore.Signal(dict)
    sampleRateChanged = QtCore.Signal(float)
    
    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        trigger_controller: TriggerController,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._plot_widget = plot_widget
        self._trigger_controller = trigger_controller
        
        # Renderer management
        self._renderers: Dict[int, TraceRenderer] = {}
        
        # Channel state
        self._channel_ids_current: List[int] = []
        self._channel_names: List[str] = []
        self._channel_configs: Dict[int, ChannelConfig] = {}
        self._channel_last_samples: Dict[int, np.ndarray] = {}
        self._channel_display_buffers: Dict[int, np.ndarray] = {}
        self._last_times: np.ndarray = np.zeros(0, dtype=np.float32)
        self._active_channel_id: Optional[int] = None
        
        # Display state
        self._current_sample_rate: float = 0.0
        self._current_window_sec: float = 1.0
        
        # Plot refresh rate tracking
        self._plot_refresh_hz: float = 60.0  # 60Hz for smooth updates
        self._plot_interval: float = 1.0 / self._plot_refresh_hz
        self._last_plot_refresh: float = 0.0
        self._actual_plot_refresh_hz: float = 0.0
        self._plot_refresh_count: int = 0
        self._plot_refresh_last_calc: float = time.perf_counter()
        
        # Chunk rate statistics
        self._chunk_rate: float = 0.0
        self._chunk_mean_samples: float = 0.0
        self._chunk_accum_count: int = 0
        self._chunk_accum_samples: int = 0
        self._chunk_rate_window: float = 1.0
        self._chunk_last_rate_update: float = time.perf_counter()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def chunk_rate(self) -> float:
        return self._chunk_rate
    
    @property
    def chunk_mean_samples(self) -> float:
        return self._chunk_mean_samples
    
    @property
    def sample_rate(self) -> float:
        return self._current_sample_rate
    
    @property
    def window_sec(self) -> float:
        return self._current_window_sec
    
    @property
    def actual_plot_refresh_hz(self) -> float:
        return self._actual_plot_refresh_hz
    
    @property
    def renderers(self) -> Dict[int, TraceRenderer]:
        """Access to renderers for MainWindow styling updates."""
        return self._renderers
    
    @property
    def channel_last_samples(self) -> Dict[int, np.ndarray]:
        return self._channel_last_samples
    
    @property
    def last_times(self) -> np.ndarray:
        return self._last_times

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    def set_plot_refresh_hz(self, hz: float) -> None:
        """Set the target plot refresh rate."""
        self._plot_refresh_hz = max(float(hz), 1.0)
        self._plot_interval = 1.0 / self._plot_refresh_hz
    
    def set_window_sec(self, seconds: float) -> None:
        """Set the display window duration."""
        self._current_window_sec = max(float(seconds), 1e-3)
        plot_item = self._plot_widget.getPlotItem()
        plot_item.setXRange(0.0, self._current_window_sec, padding=0.0)
    
    def set_active_channel(self, channel_id: Optional[int]) -> None:
        """Set which channel is currently active for styling."""
        self._active_channel_id = channel_id
        self._apply_active_channel_style()
    
    def update_channel_configs(self, configs: Dict[int, ChannelConfig]) -> None:
        """Update the channel configurations reference."""
        self._channel_configs = configs

    # -------------------------------------------------------------------------
    # Renderer Management
    # -------------------------------------------------------------------------
    
    def ensure_renderers_for_ids(
        self,
        channel_ids: Sequence[int],
        channel_names: Sequence[str],
        configs: Dict[int, ChannelConfig],
    ) -> None:
        """Synchronize the TraceRenderers with the current active channel list."""
        plot_item = self._plot_widget.getPlotItem()
        self._channel_configs = configs
        
        # If no channels, clear everything and return
        if not channel_ids:
            for renderer in self._renderers.values():
                renderer.cleanup()
            self._renderers.clear()
            self._channel_display_buffers.clear()
            self._channel_names = []
            self._channel_ids_current = []
            self._active_channel_id = None
            self._update_plot_y_range()
            return

        # Update current IDs and names
        self._channel_ids_current = list(channel_ids)
        self._channel_names = list(channel_names) if channel_names else [f"Ch {i}" for i in channel_ids]

        # Remove renderers for channels no longer present
        current_set = set(channel_ids)
        for cid in list(self._renderers.keys()):
            if cid not in current_set:
                self._renderers[cid].cleanup()
                del self._renderers[cid]
                if cid in self._channel_display_buffers:
                    del self._channel_display_buffers[cid]

        # Create renderers for new channels
        for cid, name in zip(channel_ids, self._channel_names):
            config = configs.get(cid)
            if config is None:
                continue
            if cid not in self._renderers:
                renderer = TraceRenderer(plot_item, config)
                self._renderers[cid] = renderer
            else:
                # Update existing renderer config
                self._renderers[cid].update_config(config)

        # Force plot update
        plot_item.update()
        self._refresh_channel_layout()
    
    def _refresh_channel_layout(self) -> None:
        """Rebuild plot curves to track active channels and refresh styling."""
        if not self._channel_ids_current:
            for renderer in self._renderers.values():
                renderer.cleanup()
            self._renderers.clear()
            self._update_plot_y_range()
            return

        # Remove orphaned renderers
        for cid in list(self._renderers.keys()):
            if cid not in self._channel_ids_current:
                self._renderers[cid].cleanup()
                del self._renderers[cid]

        # Ensure all current channels have renderers
        plot_item = self._plot_widget.getPlotItem()
        for cid, name in zip(self._channel_ids_current, self._channel_names):
            config = self._channel_configs.get(cid)
            if config is None:
                continue
            if cid not in self._renderers:
                self._renderers[cid] = TraceRenderer(plot_item, config)
        
        self._apply_active_channel_style()
        self._update_plot_y_range()
    
    def reset_scope_for_channels(
        self,
        channel_ids: Sequence[int],
        channel_names: Sequence[str],
        configs: Dict[int, ChannelConfig],
        window_sec: float,
    ) -> None:
        """Reset the scope display for a new set of channels."""
        self.ensure_renderers_for_ids(channel_ids, channel_names, configs)
        self._current_window_sec = max(window_sec, 1e-3)
        plot_item = self._plot_widget.getPlotItem()
        plot_item.setXRange(0.0, self._current_window_sec, padding=0.0)
        for renderer in self._renderers.values():
            renderer.clear()
        self._update_plot_y_range()
        self._ensure_active_channel_focus()
        self._current_sample_rate = 0.0
        self._chunk_rate = 0.0
        self._chunk_mean_samples = 0.0
        self._chunk_accum_count = 0
        self._chunk_accum_samples = 0
        self._chunk_last_rate_update = time.perf_counter()
    
    def clear_scope_display(self) -> None:
        """Clear all channels and reset the scope to initial state."""
        plot_item = self._plot_widget.getPlotItem()
        try:
            plot_item.clear()
        except Exception as e:
            logger.debug("Failed to clear plot item: %s", e)
        
        # Clear renderers
        for renderer in self._renderers.values():
            renderer.cleanup()
        self._renderers.clear()
        
        self._channel_display_buffers.clear()
        self._channel_ids_current = []
        self._channel_names = []
        self._active_channel_id = None
        self._apply_active_channel_style()
        self._update_plot_y_range()
        self._current_sample_rate = 0.0
        self._chunk_rate = 0.0
        self._chunk_mean_samples = 0.0
        self._chunk_accum_count = 0
        self._chunk_accum_samples = 0
        self._chunk_last_rate_update = time.perf_counter()
    
    def update_channel_display(self, channel_id: int) -> None:
        """Re-render a single channel's curve using the last raw samples and current offset/range."""
        if channel_id not in self._channel_configs:
            return
        config = self._channel_configs[channel_id]
        renderer = self._renderers.get(channel_id)
        if renderer is None:
            return
            
        # Update renderer config
        renderer.update_config(config)
        
        # Re-push data if we have it
        raw = self._channel_last_samples.get(channel_id)
        if raw is not None and raw.size > 0 and self._last_times.size > 0:
            if raw.shape[-1] == self._last_times.shape[0]:
                renderer.update_data(raw, self._last_times, downsample=1)
        
        self._apply_active_channel_style()

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------
    
    def process_streaming(
        self,
        data: np.ndarray,
        times_arr: np.ndarray,
        sample_rate: float,
        window_sec: float,
        channel_ids: List[int],
        now: float,
    ) -> None:
        """Process data in streaming (non-triggered) mode."""
        should_redraw = (now - self._last_plot_refresh) >= self._plot_interval

        if data.ndim != 2 or data.size == 0:
            for renderer in self._renderers.values():
                renderer.clear()
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            self._chunk_rate = 0.0
            self._chunk_mean_samples = 0.0
            self._chunk_accum_count = 0
            return

        # Update renderers
        # data is (channels, samples), times_arr is (samples,)
        active_samples: Dict[int, np.ndarray] = {}
        ds = 1  # Downsampling handled by TraceRenderer
            
        for i, cid in enumerate(channel_ids):
            if i >= data.shape[0]:
                break
            
            channel_data = data[i]
            active_samples[cid] = channel_data
            
            if cid in self._renderers:
                self._renderers[cid].update_data(channel_data, times_arr, downsample=ds)

        self._channel_last_samples = active_samples
        if should_redraw:
            self._apply_active_channel_style()
            if window_sec > 0:
                self._plot_widget.getPlotItem().setXRange(0, window_sec, padding=0)
            self._update_plot_y_range()
            self._last_plot_refresh = now
            
            # Track actual plot refresh rate
            self._plot_refresh_count += 1
            elapsed = now - self._plot_refresh_last_calc
            if elapsed >= 1.0:
                self._actual_plot_refresh_hz = self._plot_refresh_count / elapsed
                self._plot_refresh_count = 0
                self._plot_refresh_last_calc = now

        self._current_sample_rate = sample_rate
        self._current_window_sec = window_sec
        
        # Notify of sample rate change
        if sample_rate > 0:
            self.sampleRateChanged.emit(sample_rate)
    
    def process_trigger_mode(
        self,
        data: np.ndarray,
        times_arr: np.ndarray,
        sample_rate: float,
        window_sec: float,
        channel_ids: List[int],
        now: float,
        trigger_mode: str,
        trigger_channel_id: Optional[int],
        pretrigger_line: object,
    ) -> None:
        """Process data in triggered mode (single or continuous)."""
        if data.ndim != 2 or data.size == 0:
            # If we represent a triggered state with existing display data,
            # we should update the display regardless of new input.
            if self._trigger_controller.display_data is not None:
                self._current_sample_rate = sample_rate
                self._current_window_sec = window_sec
                self._render_trigger_display(channel_ids, window_sec, pretrigger_line)
                return
            
            # Otherwise fall back to streaming (which clears/resets)
            self.process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)
            return
            
        tc = self._trigger_controller
        if sample_rate > 0 and abs(sample_rate - tc.sample_rate) > 1e-6:
            tc._window_sec = window_sec
            tc.update_sample_rate(sample_rate)

        chunk_samples = data.T  # shape (samples, channels)
        if chunk_samples.ndim != 2 or chunk_samples.size == 0:
            self.process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)
            return
        
        # Append to trigger history
        self._append_trigger_history(chunk_samples)

        monitor_idx = None
        if trigger_channel_id is not None and trigger_channel_id in channel_ids:
            monitor_idx = channel_ids.index(trigger_channel_id)

        # Check for hold expiry
        if (
            tc.display_data is not None
            and trigger_mode != "single"
            and now >= tc._hold_until
        ):
            tc._display = None
            tc._display_times = None
            if pretrigger_line is not None:
                pretrigger_line.setVisible(False)
            tc._hold_until = 0.0

        # Detect trigger crossing
        if monitor_idx is not None and tc.should_arm(now) and tc.display_data is None:
            cross_idx = tc.detect_crossing(chunk_samples[:, monitor_idx])
            if cross_idx is not None:
                chunk_start_abs = tc._history_total - chunk_samples.shape[0]
                tc.start_capture(chunk_start_abs, cross_idx)
        elif monitor_idx is not None and tc.display_data is None:
            # Maintain previous value even if not armed
            tc.detect_crossing(chunk_samples[:, monitor_idx])

        tc.finalize_capture()

        if tc.display_data is not None:
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            self._render_trigger_display(channel_ids, window_sec, pretrigger_line)
            return

        if trigger_mode == "single":
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            return

        if trigger_mode == "continuous":
            self._current_sample_rate = sample_rate
            self._current_window_sec = window_sec
            return

        self.process_streaming(data, times_arr, sample_rate, window_sec, channel_ids, now)
    
    def _append_trigger_history(self, chunk_samples: np.ndarray) -> None:
        """Append samples to trigger history."""
        if chunk_samples.size == 0:
            return
        tc = self._trigger_controller
        tc._history.append(chunk_samples)
        tc._history_length += chunk_samples.shape[0]
        tc._history_total += chunk_samples.shape[0]
        tc._max_chunk = max(tc._max_chunk, chunk_samples.shape[0])
        # Keep 3x window to prevent evicting tails before capture
        max_keep = tc._window_samples * 3
        while tc._history_length > max_keep and tc._history:
            left = tc._history.popleft()
            tc._history_length -= left.shape[0]
    
    def _render_trigger_display(
        self,
        channel_ids: List[int],
        window_sec: float,
        pretrigger_line: object,
    ) -> None:
        """Render the captured triggered waveform."""
        tc = self._trigger_controller
        if tc.display_data is None:
            return
        window = max(window_sec, 1e-6)
        data = tc.display_data
        n = data.shape[0]
        sr = tc.sample_rate if tc.sample_rate > 0 else self._current_sample_rate
        if sr <= 0 and window > 0 and n > 0:
            sr = n / window
        if sr <= 0:
            sr = max(n / max(window, 1e-6), 1.0)
        dt = 1.0 / sr
        time_axis = np.arange(n, dtype=np.float32) * float(dt)
        tc._display_times = np.asarray(time_axis, dtype=np.float32)
        
        # Enforce view range to match the configured window
        if window > 0:
            self._plot_widget.getPlotItem().setXRange(0, window, padding=0)
        
        # data is (samples, channels) from history deque
        for idx, cid in enumerate(channel_ids):
            if idx >= data.shape[1]:
                break
            if cid in self._renderers:
                self._renderers[cid].update_data(data[:, idx], time_axis, downsample=1)

        self._channel_last_samples = {
            cid: data[:, i].astype(np.float32)
            for i, cid in enumerate(channel_ids)
            if i < data.shape[1]
        }
        self._last_times = time_axis
        self._last_plot_refresh = time.perf_counter()
        
        # Update pretrigger line position
        if pretrigger_line is not None and tc.display_pre_samples > 0:
            pre_time = tc.display_pre_samples * dt
            pretrigger_line.setValue(pre_time)
            pretrigger_line.setVisible(True)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def register_chunk(self, data: np.ndarray) -> None:
        """Register a chunk arrival for rate statistics."""
        if data.ndim != 2 or data.size == 0:
            return
        self._chunk_accum_count += 1
        self._chunk_accum_samples += data.shape[1]
        now = time.perf_counter()
        elapsed = now - self._chunk_last_rate_update
        if elapsed >= self._chunk_rate_window:
            self._chunk_rate = self._chunk_accum_count / elapsed if elapsed > 0 else 0.0
            if self._chunk_accum_count > 0:
                self._chunk_mean_samples = self._chunk_accum_samples / self._chunk_accum_count
            else:
                self._chunk_mean_samples = 0.0
            self._chunk_accum_count = 0
            self._chunk_accum_samples = 0
            self._chunk_last_rate_update = now

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------
    
    def _update_plot_y_range(self) -> None:
        """Fix the normalized viewport to [0.0, 1.0]."""
        plot_item = self._plot_widget.getPlotItem()
        plot_item.setYRange(0.0, 1.0, padding=0.0)
    
    def _apply_active_channel_style(self) -> None:
        """Update visual styles based on active channel."""
        for cid, renderer in self._renderers.items():
            is_active = cid == self._active_channel_id
            renderer.set_active(is_active)
    
    def _ensure_active_channel_focus(self) -> None:
        """Ensure active channel is valid."""
        if self._channel_ids_current:
            if self._active_channel_id not in self._channel_ids_current:
                self._active_channel_id = self._channel_ids_current[0]
        else:
            self._active_channel_id = None
        self._apply_active_channel_style()
    
    def transform_to_screen(
        self, raw_data: np.ndarray, span_v: float, offset_pct: float
    ) -> np.ndarray:
        """Transform raw voltage data to normalized screen coordinates."""
        span = max(float(span_v), 1e-9)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.asarray(raw_data, dtype=np.float32) / span + float(offset_pct)
        return np.nan_to_num(result, nan=offset_pct, posinf=offset_pct, neginf=offset_pct)

    def get_channel_at_y(self, y: float) -> Optional[int]:
        """Return the channel with screen offset closest to y, if within range."""
        candidates = []
        for cid, config in self._channel_configs.items():
            dist = abs(float(y) - config.screen_offset)
            candidates.append((dist, cid))
        
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: x[0])
        dist, cid = candidates[0]
        
        # 5% tolerance
        if dist > 0.05:
            return None
        return cid
