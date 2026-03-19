from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from shared.models import Chunk
from shared.types import AnalysisEvent


@dataclass(frozen=True)
class AnalysisBatch:
    """Sole worker-to-analysis-UI payload: one routed chunk plus its events."""

    chunk: Chunk
    events: Sequence[AnalysisEvent]


__all__ = ["AnalysisBatch"]
