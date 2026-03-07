from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Deque, List, Optional, Sequence, Tuple

from .types import AnalysisEvent


class EventRingBuffer:
    """
    Thread-safe, bounded ring buffer for AnalysisEvent objects.

    Analysis workers push into the buffer, while visualization code can either
    peek at the most recent events or drain the buffer entirely.

    [H1] Performance: a monotonic write counter lets consumers pull only new
    events since their last poll without copying the full buffer or sorting.
    """

    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._buffer: Deque[AnalysisEvent] = deque(maxlen=self._capacity)
        self._lock = Lock()
        # Monotonic write sequence – incremented on every push.  Used by
        # pull_events_since() to cheaply identify the new-event boundary.
        self._write_seq: int = 0

    def push(self, event: AnalysisEvent) -> None:
        """
        Append an event to the buffer, dropping the oldest entry if the buffer
        is full.
        """
        with self._lock:
            if len(self._buffer) == self._capacity:
                self._buffer.popleft()
            self._buffer.append(event)
            self._write_seq += 1

    def drain(self) -> List[AnalysisEvent]:
        """
        Remove and return all buffered events in chronological order.
        """
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
            return events

    def peek_all(self) -> List[AnalysisEvent]:
        """
        Return a snapshot of the buffered events without clearing the buffer.
        """
        with self._lock:
            return list(self._buffer)

    def extend(self, events: Sequence[AnalysisEvent]) -> None:
        """
        Convenience helper to push a batch of events.

        [H8] Acquires the lock once for the entire batch instead of once per
        event to reduce lock contention when detectors fire multiple events.
        """
        with self._lock:
            for event in events:
                if len(self._buffer) == self._capacity:
                    self._buffer.popleft()
                self._buffer.append(event)
                self._write_seq += 1

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def write_seq(self) -> int:
        """Current monotonic write counter (useful for fast 'anything new?' checks)."""
        with self._lock:
            return self._write_seq

    def pull_since(self, last_event_id: Optional[int] = None) -> Tuple[List[AnalysisEvent], Optional[int], int]:
        """Return only events newer than *last_event_id* without a full copy or sort.

        Because events are always appended in monotonically increasing ID order
        and the deque preserves insertion order, we can walk backward from the
        tail to find the first event with ``id > last_event_id`` and slice only
        that tail.  This is O(new) instead of the old O(n + n log n) path.

        Returns ``(new_events, new_last_event_id, write_seq)`` so callers can
        use *write_seq* as a fast-path "has anything changed?" check.
        """
        with self._lock:
            seq = self._write_seq
            buf = self._buffer
            n = len(buf)
            if n == 0:
                return [], last_event_id, seq

            if last_event_id is None:
                # First poll — return everything (same as old behaviour).
                events = list(buf)
                return events, events[-1].id, seq

            # Fast path: nothing new since last poll
            if buf[-1].id <= last_event_id:
                return [], last_event_id, seq

            # Walk backward to find split point.  In the common case
            # (a handful of new events since last poll) this touches only a
            # few elements.
            split = 0  # assume everything is new (e.g. old events were evicted)
            for i in range(n - 1, -1, -1):
                if buf[i].id <= last_event_id:
                    split = i + 1
                    break

            # Slice only the new tail.  ``list(itertools.islice(buf, …))``
            # would still iterate from the head; direct indexing on deque is
            # O(1) per element but we'd need a loop.  A small list-comp is
            # the simplest efficient approach.
            new_events = [buf[j] for j in range(split, n)]
            if new_events:
                return new_events, new_events[-1].id, seq
            return [], last_event_id, seq


class AnalysisEvents:
    """
    Consumer-facing view over the EventRingBuffer that lets callers pull only
    the events that arrived since their last poll without mutating the buffer.
    """

    def __init__(self, buffer: EventRingBuffer) -> None:
        self._buffer = buffer

    def pull_events(self, last_event_id: Optional[int] = None) -> Tuple[List[AnalysisEvent], Optional[int]]:
        """Pull new events since *last_event_id*.

        [H1] Delegates to ``EventRingBuffer.pull_since()`` which avoids the
        full-buffer copy and O(n log n) sort that the original implementation
        performed on every poll.  The return signature is unchanged for
        backward compatibility.
        """
        events, new_id, _seq = self._buffer.pull_since(last_event_id)
        return events, new_id
