from __future__ import annotations

from PySide6 import QtWidgets

from gui.analysis_tab import AnalysisTab
from gui.trigger_control_widget import TriggerControlWidget
from gui.trigger_controller import TriggerController


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_trigger_window_combo_includes_005_seconds() -> None:
    _app()
    widget = TriggerControlWidget(TriggerController())
    values = [float(widget.window_combo.itemData(i)) for i in range(widget.window_combo.count())]
    labels = [widget.window_combo.itemText(i) for i in range(widget.window_combo.count())]

    assert 0.05 in values
    assert labels[values.index(0.05)] == "0.05"


def test_analysis_tab_accepts_005_second_scope_width() -> None:
    _app()
    widget = AnalysisTab("Channel 1", 20_000.0)

    widget.update_scale(0.05, 1.0)

    assert widget._scope_window_sec == 0.05
