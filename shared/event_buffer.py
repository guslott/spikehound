from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Deque, List, Optional, Sequence, Tuple

from .types import Event


class EventRingBuffer:
    """
    Thread-safe, bounded ring buffer for Event objects.

    Analysis workers push into the buffer, while visualization code can either
    peek at the most recent events or drain the buffer entirely.
    """

    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._buffer: Deque[Event] = deque(maxlen=self._capacity)
        self._lock = Lock()

    def push(self, event: Event) -> None:
        """
        Append an event to the buffer, dropping the oldest entry if the buffer
        is full.
        """
        with self._lock:
            if len(self._buffer) == self._capacity:
                self._buffer.popleft()
            self._buffer.append(event)

    def drain(self) -> List[Event]:
        """
        Remove and return all buffered events in chronological order.
        """
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
            return events

    def peek_all(self) -> List[Event]:
        """
        Return a snapshot of the buffered events without clearing the buffer.
        """
        with self._lock:
            return list(self._buffer)

    def extend(self, events: Sequence[Event]) -> None:
        """
        Convenience helper to push a batch of events.
        """
        for event in events:
            self.push(event)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class AnalysisEvents:
    """
    Consumer-facing view over the EventRingBuffer that lets callers pull only
    the events that arrived since their last poll without mutating the buffer.
    """

    def __init__(self, buffer: EventRingBuffer) -> None:
        self._buffer = buffer

    def pull_events(self, last_event_id: Optional[int] = None) -> Tuple[List[Event], Optional[int]]:
        events = self._buffer.peek_all()
        if last_event_id is not None:
            events = [ev for ev in events if ev.id > last_event_id]
        if not events:
            return [], last_event_id
        events.sort(key=lambda ev: ev.id)
        return events, events[-1].id
