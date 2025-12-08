"""Compat shim: re-export shared models."""

from shared.models import (
    ActualConfig,
    Capabilities,
    ChannelInfo,
    Chunk,
    DeviceInfo,
    EndOfStream,
    DetectionEvent,
    TriggerConfig,
)

__all__ = [
    "Chunk",
    "DetectionEvent",
    "EndOfStream",
    "TriggerConfig",
    "ChannelInfo",
    "DeviceInfo",
    "Capabilities",
    "ActualConfig",
]
