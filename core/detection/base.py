from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Type

from shared.models import Event


@dataclass
class DetectorParameter:
    name: str
    default: float | int | bool
    min: float | None = None
    max: float | None = None
    help: str = ""


class EventDetector(Protocol):
    name: str
    display_name: str

    @property
    def parameters(self) -> Mapping[str, DetectorParameter]:
        ...

    def configure(self, **params) -> None:
        ...

    def reset(self, sample_rate: float, n_channels: int) -> None:
        """Called when acquisition (re)starts."""
        ...

    def process_chunk(self, chunk) -> Iterable[Event]:
        """Return any new events detected in this chunk."""
        ...

    def finalize(self) -> Iterable[Event]:
        """Flush any trailing events at stop (optional)."""
        ...


DETECTOR_REGISTRY: Dict[str, Type[EventDetector]] = {}


def register_detector(cls: Type[EventDetector]) -> Type[EventDetector]:
    if not hasattr(cls, "name"):
        raise ValueError(f"Detector {cls} must have a 'name' attribute")
    DETECTOR_REGISTRY[cls.name] = cls
    return cls
