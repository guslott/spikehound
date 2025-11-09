"""
Shared data structures available to both the analysis back end and the GUI.
"""

from .event_buffer import AnalysisEvents, EventRingBuffer
from .types import Event

__all__ = ["AnalysisEvents", "Event", "EventRingBuffer"]
