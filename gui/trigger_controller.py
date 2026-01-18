"""TriggerController - Manages trigger detection and state for the oscilloscope.

Extracted from MainWindow to provide a focused component for trigger configuration,
threshold crossings detection, and triggered waveform capture.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore

from shared.models import TriggerConfig

logger = logging.getLogger(__name__)



class TriggerController(QtCore.QObject):
    """
    Manages trigger detection, state, and captured waveforms.
    
    Responsibilities:
    - Store trigger configuration (mode, channel, threshold, pretrigger)
    - Maintain sample history for pretrigger capture
    - Detect threshold crossings
    - Capture and hold triggered waveforms for display
    
    Does NOT own UI widgets - MainWindow still owns those and calls
    controller methods when UI state changes.
    """

    # Emitted when trigger configuration changes
    configChanged = QtCore.Signal(object)  # TriggerConfig
    
    # Emitted when a triggered capture is ready for display
    captureReady = QtCore.Signal()

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        
        # Configuration state
        self._mode: str = "stream"
        self._channel_id: Optional[int] = None
        self._threshold: float = 0.0
        self._pre_seconds: float = 0.01
        self._window_sec: float = 1.0
        
        # Sample tracking
        self._pre_samples: int = 0
        self._window_samples: int = 1
        self._last_sample_rate: float = 0.0
        
        # Alignment config
        self._alignment_mode: str = "simple"  # "simple" or "peak"
        self._alignment_search_window_sec: float = 0.002
        
        # History buffer for pretrigger
        self._history: Deque[np.ndarray] = deque()
        self._history_length: int = 0
        self._history_total: int = 0
        self._max_chunk: int = 0
        
        # Detection state
        self._prev_value: float = 0.0
        self._capture_start_abs: Optional[int] = None
        self._capture_end_abs: Optional[int] = None
        
        # Display state
        self._display: Optional[np.ndarray] = None
        self._display_times: Optional[np.ndarray] = None
        self._display_pre_samples: int = 0
        self._hold_until: float = 0.0
        self._single_armed: bool = False

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode
    
    @property
    def channel_id(self) -> Optional[int]:
        return self._channel_id
    
    @property
    def threshold(self) -> float:
        return self._threshold
    
    @property
    def pre_seconds(self) -> float:
        return self._pre_seconds
    
    @property
    def window_sec(self) -> float:
        return self._window_sec
    
    @property
    def is_triggered_mode(self) -> bool:
        """True if in single or continuous trigger mode (not stream)."""
        return self._mode in ("single", "continuous")
    
    @property
    def alignment_mode(self) -> str:
        return self._alignment_mode
    
    @alignment_mode.setter
    def alignment_mode(self, value: str) -> None:
        self._alignment_mode = value
    
    @property
    def display_data(self) -> Optional[np.ndarray]:
        """Currently captured triggered waveform, or None."""
        return self._display
    
    @property
    def display_times(self) -> Optional[np.ndarray]:
        """Time axis for display_data, or None."""
        return self._display_times
    
    @property
    def display_pre_samples(self) -> int:
        """Number of pretrigger samples in current display."""
        return self._display_pre_samples
    
    @property
    def sample_rate(self) -> float:
        """Last known sample rate."""
        return self._last_sample_rate

    def configure(
        self,
        *,
        mode: str = "stream",
        channel_id: Optional[int] = None,
        threshold: float = 0.0,
        pre_seconds: float = 0.01,
        window_sec: float = 1.0,
        reset_state: bool = True,
    ) -> None:
        """
        Update trigger configuration.
        
        Args:
            mode: "stream", "single", or "continuous"
            channel_id: ID of the channel to trigger on
            threshold: Trigger threshold in volts
            pre_seconds: Pretrigger window in seconds
            window_sec: Total capture window in seconds
            reset_state: If True, reset capture state (history, display)
        """
        self._mode = mode
        self._channel_id = channel_id
        self._threshold = threshold
        self._pre_seconds = pre_seconds
        self._window_sec = window_sec
        
        if reset_state:
            self.reset_state()
        
        config = TriggerConfig(
            channel_index=channel_id,
            threshold=threshold,
            hysteresis=0.0,
            pretrigger_frac=pre_seconds,
            window_sec=window_sec,
            mode=mode,
        )
        self.configChanged.emit(config)

    def arm_single(self) -> None:
        """Arm single-shot trigger mode."""
        self.clear_display()
        self._single_armed = True

    def disarm_single(self) -> None:
        """Disarm single-shot trigger."""
        self._single_armed = False

    def update_sample_rate(self, sample_rate: float) -> None:
        """Update sample rate and recalculate sample counts."""
        if sample_rate <= 0:
            return
        self._last_sample_rate = sample_rate
        self._pre_samples = max(int(self._pre_seconds * sample_rate), 0)
        self._window_samples = max(int(self._window_sec * sample_rate), 1)

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset all capture state (history, display, etc.)."""
        self._history.clear()
        self._history_length = 0
        self._history_total = 0
        self._max_chunk = 0
        self._last_sample_rate = 0.0
        self._prev_value = 0.0
        self._capture_start_abs = None
        self._capture_end_abs = None
        self._display = None
        self._display_times = None
        self._hold_until = 0.0
        self._display_pre_samples = 0
        if self._mode != "single":
            self._single_armed = False

    def clear_display(self) -> None:
        """Clear only the display without resetting history."""
        self._display = None
        self._display_times = None
        self._hold_until = 0.0

    # -------------------------------------------------------------------------
    # Sample processing and detection
    # -------------------------------------------------------------------------

    def push_samples(
        self,
        chunk_samples: np.ndarray,
        sample_rate: float,
        window_sec: float,
    ) -> None:
        """
        Push a chunk of samples into the trigger history buffer.
        
        Args:
            chunk_samples: 1D array of samples from the trigger channel
            sample_rate: Current sample rate
            window_sec: Current display window in seconds
        """
        if chunk_samples.size == 0:
            return
            
        # Recalculate timing if sample rate changed
        if sample_rate != self._last_sample_rate:
            self._last_sample_rate = sample_rate
            self._pre_samples = max(int(self._pre_seconds * sample_rate), 0)
            self._window_samples = max(int(window_sec * sample_rate), 1)
            
            # Clear history on sample rate change
            self._history.clear()
            self._history_length = 0
            self._history_total = 0
            self._max_chunk = 0
        
        self._history.append(chunk_samples)
        self._history_length += chunk_samples.shape[0]
        self._history_total += chunk_samples.shape[0]
        self._max_chunk = max(self._max_chunk, chunk_samples.shape[0])
        
        # Keep 3x the trigger window to prevent evicting tails before capture
        max_keep = self._window_samples * 3
        while self._history_length > max_keep and self._history:
            left = self._history.popleft()
            self._history_length -= left.shape[0]

    def detect_crossing(self, samples: np.ndarray) -> Optional[int]:
        """
        Detect rising threshold crossing in samples.
        
        Args:
            samples: 1D array of samples to check
            
        Returns:
            Index of first crossing, or None if no crossing detected
        """
        threshold = self._threshold
        prev = self._prev_value
        
        for idx, sample in enumerate(samples):
            if prev < threshold <= sample:
                self._prev_value = float(samples[-1])
                return idx
            prev = sample
            
        self._prev_value = float(samples[-1])
        return None

    def should_arm(self, now: float) -> bool:
        """
        Check if trigger detection should be active.
        
        Args:
            now: Current time from time.perf_counter()
            
        Returns:
            True if should check for trigger crossings
        """
        # Hold display for a while after capture
        if self._display is not None and now < self._hold_until:
            return False
        # Already capturing
        if self._capture_start_abs is not None:
            return False
        # Mode-specific arming
        if self._mode == "continuous":
            return True
        if self._mode == "single":
            return self._single_armed
        return False

    def start_capture(self, chunk_start_abs: int, trigger_idx: int) -> None:
        """
        Start a trigger capture at the given position.
        
        Args:
            chunk_start_abs: Absolute sample index where current chunk starts
            trigger_idx: Index within chunk where crossing occurred
        """
        window = self._window_samples
        if window <= 0:
            return
            
        pre = self._pre_samples
        
        # Add alignment padding if needed
        align_pad = 0
        if self._alignment_mode == "peak":
            align_pad = int(self._alignment_search_window_sec * self._last_sample_rate)
            # Ensure at least a few samples for search
            align_pad = max(align_pad, 5)

        earliest_abs = self._history_total - self._history_length
        # Request data starting earlier to allow for alignment search
        start_abs = max(chunk_start_abs + trigger_idx - pre - align_pad, earliest_abs)
        
        self._capture_start_abs = start_abs
        # Capture enough for window + 2*padding (search left and right)
        self._capture_end_abs = start_abs + window + 2 * align_pad
        
        if self._mode == "single":
            self._single_armed = False

    def finalize_capture(self) -> bool:
        """
        Finalize a pending capture if enough samples have arrived.
        
        Returns:
            True if capture was finalized, False otherwise
        """
        if self._capture_start_abs is None or self._capture_end_abs is None:
            return False
        if self._history_total < self._capture_end_abs:
            return False
        if not self._history:
            return False
        
        # Calculate absolute range
        earliest_abs = self._history_total - self._history_length
        start_abs = max(self._capture_start_abs, earliest_abs)
        end_abs = start_abs + self._window_samples
        
        # Collect relevant chunks
        relevant_chunks: List[np.ndarray] = []
        current_abs = earliest_abs
        
        for chunk in self._history:
            chunk_len = chunk.shape[0]
            chunk_end = current_abs + chunk_len
            
            if chunk_end > start_abs and current_abs < end_abs:
                relevant_chunks.append(chunk)
                
            current_abs += chunk_len
            if current_abs >= end_abs:
                break
        
        if not relevant_chunks:
            return False
        
        # Concatenate relevant chunks
        data = np.concatenate(relevant_chunks, axis=0)
        
        # Find absolute start of first chunk
        scan_abs = earliest_abs
        data_start_abs = earliest_abs
        for chunk in self._history:
            if chunk is relevant_chunks[0]:
                data_start_abs = scan_abs
                break
            scan_abs += chunk.shape[0]
        
        # Slice to get the window
        start_idx = start_abs - data_start_abs
        
        # Perform alignment if requested
        if self._alignment_mode == "peak":
            align_pad = int(self._alignment_search_window_sec * self._last_sample_rate)
            align_pad = max(align_pad, 5)
            
            # The nominal trigger point is at start_idx + align_pad + pre_samples
            nominal_trigger = start_idx + align_pad + self._pre_samples
            
            # Search window around nominal trigger
            # We look +/- align_pad
            s_start = max(0, nominal_trigger - align_pad)
            s_end = min(data.shape[0], nominal_trigger + align_pad)
            
            search_region = data[s_start:s_end]
            if search_region.size > 0:
                # Find peak offset relative to start of search region
                peak_offset = np.argmax(np.abs(search_region))
                
                # Absolute index of the peak in data
                peak_abs_idx = s_start + peak_offset
                
                # We want peak_abs_idx to end up at index 'pre_samples' in the final snippet
                # snippet = data[start:end]
                # peak_abs_idx - start = pre_samples
                # start = peak_abs_idx - pre_samples
                
                final_start = peak_abs_idx - self._pre_samples
                start_idx = final_start

        end_idx = start_idx + self._window_samples
        
        # Handle start before available data (if peak shifted left past start)
        pad_front = 0
        if start_idx < 0:
            pad_front = -start_idx
            start_idx = 0
            
        if end_idx > data.shape[0]:
            end_idx = data.shape[0]
            
        snippet = data[start_idx:end_idx]
        
        if pad_front > 0:
            if snippet.ndim == 1:
                padding = np.zeros(pad_front, dtype=snippet.dtype)
            else:
                padding = np.zeros((pad_front, snippet.shape[1]), dtype=snippet.dtype)
            snippet = np.concatenate([padding, snippet], axis=0)
        
        # Pad if needed
        if snippet.shape[0] < self._window_samples:
            pad = self._window_samples - snippet.shape[0]
            if snippet.shape[0] > 0:
                last_row = snippet[-1:]
            else:
                ndim = data.ndim
                if ndim == 1:
                    last_row = np.zeros(1, dtype=np.float32)
                else:
                    last_row = np.zeros((1, data.shape[1]), dtype=np.float32)
            snippet = np.concatenate([snippet, np.repeat(last_row, pad, axis=0)], axis=0)
            
        if snippet.shape[0] == 0:
            ndim = data.ndim
            if ndim == 1:
                snippet = np.zeros(self._window_samples, dtype=np.float32)
            else:
                snippet = np.zeros((self._window_samples, data.shape[1]), dtype=np.float32)
        
        self._display = snippet
        self._display_times = None
        self._display_pre_samples = min(self._pre_samples, max(snippet.shape[0] - 1, 0))
        
        # Hold display for duration of window
        if self._last_sample_rate > 0:
            duration = self._window_samples / self._last_sample_rate
        else:
            duration = self._window_sec
        self._hold_until = time.perf_counter() + max(duration, 1e-3)
        
        # Reset capture pointers
        self._capture_start_abs = None
        self._capture_end_abs = None
        
        self.captureReady.emit()
        return True

    def get_display_times(self, window_sec: float) -> np.ndarray:
        """
        Generate time axis for current display data.
        
        Args:
            window_sec: Display window in seconds
            
        Returns:
            Time array aligned with display data
        """
        if self._display is None:
            return np.zeros(0, dtype=np.float32)
            
        n = self._display.shape[0]
        sr = self._last_sample_rate if self._last_sample_rate > 0 else 10000.0
        pre = self._display_pre_samples
        
        # Time axis: sample index 'pre' should correspond exactly to t=0.0
        # Use arange to correctly compute: time[i] = (i - pre) / sr
        return (np.arange(n, dtype=np.float32) - pre) / sr
