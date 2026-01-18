"""Qt signal adapter for Dispatcher tick events.

This adapter bridges the core Dispatcher's callback-based notification system
to Qt signals, enabling the GUI to receive tick events via Qt's signal/slot
mechanism. This keeps PySide6 dependencies out of the core module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from PySide6 import QtCore

if TYPE_CHECKING:
    from core.dispatcher import Dispatcher


class DispatcherSignals(QtCore.QObject):
    """Qt signals for dispatcher events."""
    tick = QtCore.Signal(dict)


def connect_dispatcher_signals(dispatcher: "Dispatcher") -> tuple[DispatcherSignals, Callable[[], None]]:
    """Create Qt signal bridge for dispatcher tick events.
    
    Args:
        dispatcher: The dispatcher to connect to.
        
    Returns:
        A tuple of (signals object, unsubscribe function).
    """
    signals = DispatcherSignals()
    unsubscribe = dispatcher.add_tick_callback(signals.tick.emit)
    return signals, unsubscribe
