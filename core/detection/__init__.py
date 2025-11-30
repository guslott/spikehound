from .base import (
    DETECTOR_REGISTRY,
    DetectorParameter,
    EventDetector,
    register_detector,
)
from .threshold import AmpThresholdDetector

__all__ = [
    "EventDetector",
    "DetectorParameter",
    "DETECTOR_REGISTRY",
    "register_detector",
    "AmpThresholdDetector",
]
