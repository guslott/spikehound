"""
Reference implementations for property-based testing.

These are deliberately simple, obviously-correct implementations used to
verify the production code via differential testing. They prioritize
correctness and clarity over performance.

Design for AI-maintainability:
- Each reference model matches the API of its production counterpart
- Implementations use basic Python/numpy without clever optimizations
- Operations are traceable for debugging test failures
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


class ReferenceRingBuffer:
    """Simple list-based ring buffer for property testing.

    This implementation is intentionally naive to serve as a "ground truth"
    for verifying SharedRingBuffer. It stores all writes and provides
    reads by slicing the logical sequence.

    Key invariants being tested:
    1. Reads return correct data even after wraparound
    2. Old data is overwritten but new data is always accessible
    3. dtype is preserved
    """

    def __init__(self, capacity: int, n_channels: int = 1, dtype: np.dtype = np.float32):
        self._capacity = capacity
        self._n_channels = n_channels
        self._dtype = dtype

        # Simple deque-style tracking using a list
        # Each element is a (channels, samples) block
        self._data: List[np.ndarray] = []
        self._total_written = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        return self._total_written

    def write(self, data: np.ndarray) -> int:
        """Write data and return the logical start position."""
        data = np.asarray(data, dtype=self._dtype)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        assert data.shape[0] == self._n_channels, "Channel count mismatch"

        start_pos = self._total_written
        n_samples = data.shape[1]

        # Append to our log
        self._data.append(data.copy())
        self._total_written += n_samples

        # Prune old data beyond capacity
        self._prune()

        return start_pos % self._capacity

    def _prune(self) -> None:
        """Remove data older than capacity."""
        while self._get_available() > self._capacity:
            if self._data:
                oldest = self._data[0]
                excess = self._get_available() - self._capacity
                if oldest.shape[1] <= excess:
                    self._data.pop(0)
                else:
                    # Partial removal from oldest block
                    self._data[0] = oldest[:, excess:]
            else:
                break

    def _get_available(self) -> int:
        """Total samples currently stored."""
        return sum(d.shape[1] for d in self._data)

    def read_latest(self, n_samples: int) -> np.ndarray:
        """Read the most recent n_samples."""
        available = self._get_available()
        n_samples = min(n_samples, available)

        if n_samples == 0:
            return np.zeros((self._n_channels, 0), dtype=self._dtype)

        # Collect from the end
        result_parts = []
        remaining = n_samples

        for block in reversed(self._data):
            if remaining <= 0:
                break
            take = min(remaining, block.shape[1])
            result_parts.append(block[:, -take:])
            remaining -= take

        result_parts.reverse()
        return np.concatenate(result_parts, axis=1)

    def get_all_logical(self) -> np.ndarray:
        """Return all data currently in buffer as contiguous array."""
        if not self._data:
            return np.zeros((self._n_channels, 0), dtype=self._dtype)
        return np.concatenate(self._data, axis=1)


class ReferenceThresholdDetector:
    """Simple threshold detector for property testing.

    This is a minimal implementation that:
    1. Detects crossings of a threshold
    2. Enforces refractory period
    3. Records detection times

    Properties being verified:
    1. All detections are above threshold
    2. No detections within refractory period of each other
    3. Detection times are monotonically increasing
    """

    def __init__(
        self,
        threshold: float,
        refractory_samples: int,
        polarity: int = -1,  # -1 for negative, +1 for positive
    ):
        self._threshold = threshold
        self._refractory = refractory_samples
        self._polarity = polarity
        self._last_detection: Optional[int] = None

    def detect(self, samples: np.ndarray) -> List[int]:
        """Detect threshold crossings in 1D signal.

        Args:
            samples: 1D array of samples.

        Returns:
            List of sample indices where detections occurred.
        """
        samples = np.asarray(samples, dtype=np.float64).flatten()
        detections = []

        for i, s in enumerate(samples):
            # Check polarity
            if self._polarity < 0:
                crosses = s < self._threshold
            else:
                crosses = s > self._threshold

            if not crosses:
                continue

            # Check refractory
            if self._last_detection is not None:
                if i - self._last_detection < self._refractory:
                    continue

            detections.append(i)
            self._last_detection = i

        return detections

    def reset(self) -> None:
        """Reset detector state."""
        self._last_detection = None


@dataclass
class DetectionTestCase:
    """Test case for spike detection validation.

    Contains a signal with known spike locations, allowing comparison
    of detected spikes against ground truth.
    """
    signal: np.ndarray
    ground_truth_samples: np.ndarray  # Sample indices of actual spikes
    sample_rate: float
    description: str = ""

    def tolerance_samples(self, tolerance_ms: float = 0.5) -> int:
        """Convert ms tolerance to sample count."""
        return int(tolerance_ms * self.sample_rate / 1000.0)


def create_detection_oracle(
    ground_truth: np.ndarray,
    detected: np.ndarray,
    tolerance: int,
) -> Tuple[int, int, int]:
    """Compare detected events against ground truth.

    Args:
        ground_truth: Array of true spike sample indices.
        detected: Array of detected spike sample indices.
        tolerance: Maximum allowed distance (samples) for a match.

    Returns:
        Tuple of (true_positives, false_positives, false_negatives).
    """
    ground_truth = np.asarray(ground_truth, dtype=np.int64)
    detected = np.asarray(detected, dtype=np.int64)

    if ground_truth.size == 0 and detected.size == 0:
        return (0, 0, 0)

    if ground_truth.size == 0:
        return (0, len(detected), 0)

    if detected.size == 0:
        return (0, 0, len(ground_truth))

    # Match detections to ground truth (greedy nearest-neighbor)
    matched_gt = set()
    matched_det = set()

    for d_idx, d in enumerate(detected):
        distances = np.abs(ground_truth - d)
        nearest_idx = int(np.argmin(distances))
        nearest_dist = distances[nearest_idx]

        if nearest_dist <= tolerance and nearest_idx not in matched_gt:
            matched_gt.add(nearest_idx)
            matched_det.add(d_idx)

    true_positives = len(matched_gt)
    false_positives = len(detected) - len(matched_det)
    false_negatives = len(ground_truth) - len(matched_gt)

    return (true_positives, false_positives, false_negatives)
