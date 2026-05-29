"""Regression test for _update_status per-tick waste (finding 7.3)."""
from __future__ import annotations

from types import SimpleNamespace

from gui.main_window import MainWindow


class _SpyController:
    def __init__(self) -> None:
        self.stats_calls = 0
        self.depth_calls = 0

    def dispatcher_stats(self):
        self.stats_calls += 1
        return {}

    def queue_depths(self):
        self.depth_calls += 1
        return {}


def _status_stub(controller, status_labels):
    return SimpleNamespace(
        _chunk_accum_count=1,  # non-zero -> skip the stale-rate clear branch
        _chunk_last_rate_update=0.0,
        _chunk_rate_window=1.0,
        _chunk_rate=0.0,
        _chunk_mean_samples=0.0,
        _status_labels=status_labels,
        _last_status_update=0.0,
        _controller=controller,
        _current_sample_rate=20000.0,
    )


def test_update_status_skips_core_locks_when_no_status_labels():
    """With no status labels wired up, _update_status must NOT call the
    lock-taking dispatcher_stats()/queue_depths() on every ~60 Hz tick (7.3)."""
    controller = _SpyController()
    stub = _status_stub(controller, status_labels={})

    # Call the method unbound on a lightweight stub (no Qt widgets needed).
    MainWindow._update_status(stub, 0)

    assert controller.stats_calls == 0, "should not call dispatcher_stats() with no labels"
    assert controller.depth_calls == 0, "should not call queue_depths() with no labels"


def test_update_status_throttles_when_labels_present():
    """When labels exist, the status work is throttled rather than run every tick:
    a second immediate call is skipped by the throttle window."""
    controller = _SpyController()
    stub = _status_stub(controller, status_labels={"x": object()})

    MainWindow._update_status(stub, 0)   # first call does the work
    first_stats = controller.stats_calls
    MainWindow._update_status(stub, 0)   # immediate second call -> throttled

    assert first_stats == 1, "first call should refresh status once"
    assert controller.stats_calls == 1, "an immediate second call should be throttled"
