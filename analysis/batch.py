from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from shared.models import Chunk
from shared.types import AnalysisEvent


@dataclass(frozen=True)
class AnalysisBatch:
    """A routed chunk plus the analysis events detected within it."""

    chunk: Chunk
    events: Sequence[AnalysisEvent]


__all__ = ["AnalysisBatch"]
