from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class AnalysisSettings:
    event_window_ms: float = 10.0


class AnalysisSettingsStore:
    """
    Thread-safe settings container that allows multiple producers/consumers to
    observe changes (e.g., GUI controls updating worker parameters).
    """

    def __init__(self, initial: Optional[AnalysisSettings] = None) -> None:
        self._settings = initial or AnalysisSettings()
        self._lock = threading.Lock()
        self._subscribers: Dict[int, Callable[[AnalysisSettings], None]] = {}
        self._next_token = 0

    def get(self) -> AnalysisSettings:
        with self._lock:
            return self._settings

    def update(self, **kwargs) -> AnalysisSettings:
        with self._lock:
            new_settings = replace(self._settings, **kwargs)
            self._settings = new_settings
            callbacks = list(self._subscribers.values())
        for callback in callbacks:
            try:
                callback(new_settings)
            except Exception:
                # Best-effort notifications; a bad subscriber should not break updates.
                continue
        return new_settings

    def subscribe(self, callback: Callable[[AnalysisSettings], None], *, replay: bool = True) -> Callable[[], None]:
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._subscribers[token] = callback
            snapshot = self._settings
        if replay:
            callback(snapshot)

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(token, None)

        return unsubscribe


__all__ = ["AnalysisSettings", "AnalysisSettingsStore"]
