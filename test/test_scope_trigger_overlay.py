"""Regression tests for the scope trigger overlay (threshold/pretrigger lines).

These guard two bugs that made the black threshold line fail to appear in
Single/Repeated trigger mode:

  * The overlay InfiniteLines were orphaned from the plot by
    ``PlotManager.clear_scope_display`` calling ``plot_item.clear()`` (on device
    disconnect / hardware rescan). ``setVisible(True)`` then flipped the flag
    but drew nothing, because the item was no longer in the scene. Note that
    ``isVisible()`` returns True on an orphaned line, so visibility alone is not
    a sufficient assertion -- we check scene membership (``plot_item.items``).

  * Entering a triggered mode with no trigger channel raised in
    ``TriggerConfig.__post_init__`` (channel_index=None), aborting the config
    signal before the visual update ran.
"""
from __future__ import annotations

from PySide6 import QtWidgets

from gui.scope_widget import ScopeWidget
from gui.plot_manager import PlotManager
from gui.trigger_controller import TriggerController
from gui.trigger_control_widget import TriggerControlWidget


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _plot_items(scope: ScopeWidget) -> list:
    return scope.plot_widget.getPlotItem().items


def test_threshold_line_is_black_and_movable() -> None:
    _app()
    scope = ScopeWidget()
    pen = scope.threshold_line.pen
    assert (pen.color().red(), pen.color().green(), pen.color().blue()) == (0, 0, 0)
    assert scope.threshold_line.movable is True


def test_overlay_lines_survive_clear_scope_display() -> None:
    """clear_scope_display must NOT detach the persistent overlay lines."""
    _app()
    scope = ScopeWidget()
    manager = PlotManager(scope.plot_widget, TriggerController())

    assert scope.threshold_line in _plot_items(scope)
    assert scope.pretrigger_line in _plot_items(scope)

    manager.clear_scope_display()  # simulates a device disconnect / rescan

    assert scope.threshold_line in _plot_items(scope)
    assert scope.pretrigger_line in _plot_items(scope)


def test_ensure_overlay_items_reattaches_after_external_clear() -> None:
    """If anything detaches the lines, ensure_overlay_items() restores them."""
    _app()
    scope = ScopeWidget()
    scope.plot_widget.getPlotItem().clear()  # brute-force detach (worst case)

    assert scope.threshold_line not in _plot_items(scope)

    scope.ensure_overlay_items()

    assert scope.threshold_line in _plot_items(scope)
    assert scope.pretrigger_line in _plot_items(scope)


def test_threshold_visible_and_in_scene_after_clear_cycle() -> None:
    """The reported symptom: after a clear, showing the line must actually draw."""
    _app()
    scope = ScopeWidget()
    manager = PlotManager(scope.plot_widget, TriggerController())
    manager.clear_scope_display()

    scope.ensure_overlay_items()
    scope.set_threshold(0.5, visible=True)

    assert scope.threshold_line.isVisible() is True
    assert scope.threshold_line in _plot_items(scope)  # the part that used to fail


def test_threshold_line_drag_emits_value() -> None:
    """Dragging the line (sigPositionChanged) forwards the new value."""
    _app()
    scope = ScopeWidget()
    seen: list[float] = []
    scope.thresholdChanged.connect(lambda v: seen.append(v))

    scope.threshold_line.setValue(0.7)

    assert seen and abs(seen[-1] - 0.7) < 1e-6


def test_enter_triggered_mode_without_channel_falls_back_to_stream() -> None:
    """Selecting Single/Repeated with no channel must not raise; coerce to stream."""
    _app()
    controller = TriggerController()
    widget = TriggerControlWidget(controller)
    widget.update_channels([])  # no channels available

    # Force the (normally disabled) Single radio on, as a stray UI state would.
    widget.trigger_mode_single.setChecked(True)
    widget._on_config_changed()  # must not raise

    assert controller.mode == "stream"
    assert widget.trigger_mode_stream.isChecked() is True


def test_enter_single_mode_with_channel_keeps_single() -> None:
    """Sanity: with a valid channel, Single mode is honored (and emits config)."""
    _app()
    controller = TriggerController()
    widget = TriggerControlWidget(controller)
    widget.update_channels([("Ch 0", 0)])

    widget.trigger_mode_single.setChecked(True)
    widget._on_config_changed()

    assert controller.mode == "single"
    assert controller.channel_id == 0
