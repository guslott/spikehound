"""
Shared data structures available to both the analysis back end and the GUI.
"""

from .ring_buffer import SharedRingBuffer
from .types import AnalysisEvent

__all__ = ["AnalysisEvent", "SharedRingBuffer"]
