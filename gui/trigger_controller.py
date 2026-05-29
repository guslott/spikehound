"""TriggerController - Manages trigger detection and state for the oscilloscope.

Extracted from MainWindow to provide a focused component for trigger configuration,
threshold crossings detection, and triggered waveform capture.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, List, Optional

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
        self._edge: str = "rising"
        self._hysteresis: float = 0.0

        # Sample tracking
        self._pre_samples: int = 0
        self._window_samples: int = 1
        self._last_sample_rate: float = 0.0
        
        # History buffer for pretrigger
        self._history: Deque[np.ndarray] = deque()
        self._history_length: int = 0
        self._history_total: int = 0
        self._max_chunk: int = 0
        
        # Detection state
        self._prev_value: float = 0.0
        # Hysteresis priming: a trigger only fires once the signal has retreated
        # past the far side of the noise-reject band (re-primed on every reset).
        self._rise_primed: bool = True
        self._fall_primed: bool = True
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
    def edge(self) -> str:
        """Trigger edge: 'rising', 'falling', or 'either'."""
        return self._edge

    @property
    def hysteresis(self) -> float:
        """Noise-reject band in volts (0 disables hysteresis)."""
        return self._hysteresis

    @property
    def is_triggered_mode(self) -> bool:
        """True if in single or repeated trigger mode (not stream)."""
        return self._mode in ("single", "repeated")

    @property
    def display_data(self) -> Optional[np.ndarray]:
        """Currently captured triggered waveform, or None."""
        return self._display
    
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
        edge: str = "rising",
        hysteresis: float = 0.0,
        reset_state: bool = True,
        preserve_display_on_reset: bool = False,
    ) -> None:
        """
        Update trigger configuration.

        Args:
            mode: "stream", "single", or "repeated"
            channel_id: ID of the channel to trigger on
            threshold: Trigger threshold in volts
            pre_seconds: Pretrigger window in seconds
            window_sec: Total capture window in seconds
            edge: Trigger edge - "rising", "falling", or "either"
            hysteresis: Noise-reject band in volts (0 disables it)
            reset_state: If True, reset capture state (history, display)
            preserve_display_on_reset: If True and reset_state=True, keep the
                currently displayed capture while resetting trigger internals.
        """
        self._mode = mode
        self._channel_id = channel_id
        self._threshold = threshold
        self._pre_seconds = pre_seconds
        self._window_sec = window_sec
        self._edge = edge if edge in ("rising", "falling", "either") else "rising"
        self._hysteresis = max(float(hysteresis), 0.0)

        sample_rate = self._last_sample_rate
        if reset_state:
            self.reset_state(clear_display=not preserve_display_on_reset)
        if sample_rate > 0:
            self.update_sample_rate(sample_rate)

        config = TriggerConfig(
            channel_index=channel_id,
            threshold=threshold,
            hysteresis=self._hysteresis,
            pretrigger_sec=pre_seconds,
            window_sec=window_sec,
            mode=mode,
        )
        self.configChanged.emit(config)

    def arm_single(self) -> None:
        """Arm single-shot trigger mode."""
        self.clear_display()
        self._single_armed = True

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

    def reset_state(self, *, clear_display: bool = True) -> None:
        """Reset trigger internals.

        Args:
            clear_display: If False, preserve the currently displayed capture
                so UI interactions can continue to reference it.
        """
        self._history.clear()
        self._history_length = 0
        self._history_total = 0
        self._max_chunk = 0
        self._prev_value = 0.0
        self._rise_primed = True
        self._fall_primed = True
        self._capture_start_abs = None
        self._capture_end_abs = None
        if clear_display:
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

        sample_rate_changed = abs(sample_rate - self._last_sample_rate) > 1e-6
        desired_pre_samples = max(int(self._pre_seconds * sample_rate), 0)
        desired_window_samples = max(int(self._window_sec * sample_rate), 1)
        timing_changed = (
            desired_pre_samples != self._pre_samples
            or desired_window_samples != self._window_samples
        )

        # Refresh timing whenever the configured pretrigger/window changes, not
        # only when the device sample rate changes.
        if sample_rate_changed or timing_changed:
            self._last_sample_rate = sample_rate
            self._pre_samples = desired_pre_samples
            self._window_samples = desired_window_samples

            if sample_rate_changed:
                # History stored at the previous sample cadence is not
                # compatible with the new rate.
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
        Detect the first threshold crossing in ``samples`` for the configured
        edge ("rising", "falling", or "either"), honoring the hysteresis
        (noise-reject) band.

        Args:
            samples: 1D array of samples to check

        Returns:
            Index of the first qualifying crossing, or None.
        """
        if samples.size == 0:
            return None

        threshold = self._threshold
        hyst = self._hysteresis
        edge = self._edge

        # Prepend the carried-over previous value for a shifted comparison.
        extended = np.empty(samples.size + 1, dtype=np.float32)
        extended[0] = self._prev_value
        extended[1:] = samples
        prev = extended[:-1]
        cur = extended[1:]
        self._prev_value = float(samples[-1])

        if hyst <= 0.0:
            # Fast vectorized path (no priming state needed).
            candidates: List[int] = []
            if edge in ("rising", "either"):
                rises = np.where((prev < threshold) & (cur >= threshold))[0]
                if rises.size:
                    candidates.append(int(rises[0]))
            if edge in ("falling", "either"):
                falls = np.where((prev > threshold) & (cur <= threshold))[0]
                if falls.size:
                    candidates.append(int(falls[0]))
            return min(candidates) if candidates else None

        return self._detect_with_hysteresis(prev, cur, threshold, hyst, edge)

    def _detect_with_hysteresis(
        self,
        prev: np.ndarray,
        cur: np.ndarray,
        threshold: float,
        hyst: float,
        edge: str,
    ) -> Optional[int]:
        """Stateful edge detection with a noise-reject band.

        A rising trigger only fires once the signal has dipped below
        ``threshold - hyst`` since the last fire (and symmetrically for
        falling). This rejects repeated triggering on noise hovering at the
        threshold. Runs a short per-sample scan (detection is active only while
        waiting for a trigger), so it stays simple rather than vectorized.
        """
        rise_arm = threshold - hyst
        fall_arm = threshold + hyst
        want_rise = edge in ("rising", "either")
        want_fall = edge in ("falling", "either")
        for i in range(cur.shape[0]):
            p = float(prev[i])
            c = float(cur[i])
            if want_rise:
                if c < rise_arm:
                    self._rise_primed = True
                if self._rise_primed and p < threshold <= c:
                    self._rise_primed = False
                    return i
            if want_fall:
                if c > fall_arm:
                    self._fall_primed = True
                if self._fall_primed and p > threshold >= c:
                    self._fall_primed = False
                    return i
        return None

    def update_baseline(self, samples: np.ndarray) -> None:
        """Track the latest sample (and re-prime hysteresis) without firing.

        Used while detection is intentionally idle (e.g. single mode that isn't
        armed yet) so the edge/hysteresis state stays current for when
        detection resumes -- without consuming a crossing.
        """
        if samples.size == 0:
            return
        if self._hysteresis > 0.0:
            if self._edge in ("rising", "either") and float(samples.min()) < self._threshold - self._hysteresis:
                self._rise_primed = True
            if self._edge in ("falling", "either") and float(samples.max()) > self._threshold + self._hysteresis:
                self._fall_primed = True
        self._prev_value = float(samples[-1])

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
        if self._mode == "repeated":
            return True
        if self._mode == "single":
            return self._single_armed
        return False

    def check_hold_expiry(self, now: float, is_single_mode: bool) -> bool:
        """
        Check if hold period has expired and clear display if needed.

        Args:
            now: Current time from time.perf_counter()
            is_single_mode: True if in single trigger mode

        Returns:
            True if display was cleared due to expiry, False otherwise
        """
        if self._display is None or is_single_mode:
            return False
        # A capture that is still streaming in (progressive sweep) has not begun
        # its hold yet; never clear it mid-sweep.
        if self._capture_start_abs is not None:
            return False
        if now >= self._hold_until:
            self._display = None
            self._display_times = None
            self._hold_until = 0.0
            return True
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

        earliest_abs = self._history_total - self._history_length
        # Trigger point is at chunk_start_abs + trigger_idx
        # We want to capture 'pre' samples before this and 'window - pre' after
        start_abs = max(chunk_start_abs + trigger_idx - pre, earliest_abs)

        self._capture_start_abs = start_abs
        # Capture exactly the window duration
        self._capture_end_abs = start_abs + window

        if self._mode == "single":
            self._single_armed = False

    def advance_capture(self) -> bool:
        """Extend the on-screen capture with whatever samples have arrived.

        Progressive ("sweep") rendering: after a trigger fires, the displayed
        window grows from the trigger point as data streams in, instead of
        waiting for the whole window to buffer. Each call republishes the
        partial capture (emitting ``captureReady``).

        Returns True once the full window has been captured -- at which point
        the display is latched (hold started, capture pointers cleared) so the
        controller can re-arm (repeated) or stay put (single). Returns False
        while the sweep is still filling, or when there is nothing to show yet.
        """
        if self._capture_start_abs is None or self._capture_end_abs is None:
            return False
        if not self._history:
            return False

        earliest_abs = self._history_total - self._history_length
        start_abs = max(self._capture_start_abs, earliest_abs)
        # Only render up to the samples that have actually arrived.
        end_abs = min(self._capture_start_abs + self._window_samples, self._history_total)
        if end_abs <= start_abs:
            return False
        available = end_abs - start_abs

        # Collect chunks covering [start_abs, end_abs)
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

        total_len = sum(chunk.shape[0] for chunk in relevant_chunks)
        if relevant_chunks[0].ndim == 1:
            data = np.empty(total_len, dtype=np.float32)
        else:
            data = np.empty((total_len, relevant_chunks[0].shape[1]), dtype=np.float32)
        offset = 0
        for chunk in relevant_chunks:
            chunk_len = chunk.shape[0]
            data[offset:offset + chunk_len] = chunk
            offset += chunk_len

        # Absolute index of the first collected chunk.
        scan_abs = earliest_abs
        data_start_abs = earliest_abs
        for chunk in self._history:
            if chunk is relevant_chunks[0]:
                data_start_abs = scan_abs
                break
            scan_abs += chunk.shape[0]

        start_idx = start_abs - data_start_abs
        end_idx = start_idx + available
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

        if snippet.shape[0] == 0:
            return False

        # No back-padding: the not-yet-arrived tail of the window stays blank so
        # the trace visibly sweeps left-to-right.
        self._display = snippet
        self._display_times = None
        self._display_pre_samples = min(self._pre_samples, max(snippet.shape[0] - 1, 0))

        complete = self._history_total >= self._capture_end_abs
        if complete:
            # Latch: hold the finished window, then allow re-arm per mode.
            if self._last_sample_rate > 0:
                duration = self._window_samples / self._last_sample_rate
            else:
                duration = self._window_sec
            self._hold_until = time.perf_counter() + max(duration, 1e-3)
            self._capture_start_abs = None
            self._capture_end_abs = None

        self.captureReady.emit()
        return complete
