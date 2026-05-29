"""F10: TriggerConfig.pretrigger_sec is seconds (0 <= pretrigger_sec <=
window_sec), not a [0,1] fraction. These lock in the corrected semantics.
"""
from __future__ import annotations

import pytest

from shared.models import TriggerConfig


def _cfg(pretrigger_sec: float, window_sec: float) -> TriggerConfig:
    return TriggerConfig(
        channel_index=0,
        threshold=0.5,
        hysteresis=0.0,
        pretrigger_sec=pretrigger_sec,
        window_sec=window_sec,
        mode="repeated",
    )


def test_pretrigger_seconds_above_one_is_valid_within_window() -> None:
    # Previously rejected as a fraction > 1; now valid as seconds <= window.
    cfg = _cfg(pretrigger_sec=1.5, window_sec=3.0)
    assert cfg.pretrigger_sec == pytest.approx(1.5)


def test_pretrigger_seconds_equal_to_window_is_valid() -> None:
    cfg = _cfg(pretrigger_sec=0.2, window_sec=0.2)
    assert cfg.pretrigger_sec == pytest.approx(0.2)


def test_pretrigger_seconds_exceeding_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="pretrigger_sec"):
        _cfg(pretrigger_sec=2.0, window_sec=1.0)


def test_negative_pretrigger_is_rejected() -> None:
    with pytest.raises(ValueError, match="pretrigger_sec"):
        _cfg(pretrigger_sec=-0.1, window_sec=1.0)
