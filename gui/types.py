"""GUI type definitions for channel configuration and display settings.

This module defines data classes used to configure per-channel display properties
including colors, vertical scaling, filter settings, and feature toggles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from PySide6 import QtGui

@dataclass
class ChannelConfig:
    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(0, 0, 139))
    display_enabled: bool = True
    vertical_span_v: float = 1.0
    screen_offset: float = 0.5
    notch_enabled: bool = False
    notch_freq: float = 60.0
    highpass_enabled: bool = False
    highpass_freq: float = 10.0
    lowpass_enabled: bool = False
    lowpass_freq: float = 1_000.0
    listen_enabled: bool = False
    analyze_enabled: bool = False
    channel_name: str = ""
