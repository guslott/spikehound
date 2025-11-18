"""Compat shim: re-export shared models."""

from shared.models import (
    ActualConfig,
    Capabilities,
    ChannelInfo,
    Chunk,
    DeviceInfo,
    EndOfStream,
    Event,
    TriggerConfig,
)

__all__ = [
    "Chunk",
    "Event",
    "EndOfStream",
    "TriggerConfig",
    "ChannelInfo",
    "DeviceInfo",
    "Capabilities",
    "ActualConfig",
]
