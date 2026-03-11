from __future__ import annotations

from PySide6 import QtWidgets

from gui.scope_widget import ScopeWidget


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_scope_widget_exposes_popout_button_and_signal() -> None:
    _app()
    widget = ScopeWidget()
    emissions: list[bool] = []
    widget.popoutRequested.connect(lambda: emissions.append(True))

    assert widget.popout_button.toolTip() == "Open the active scope trace in a separate waveform window."
    assert "128, 0, 32" in widget.popout_button.styleSheet()

    widget.popout_button.click()

    assert emissions == [True]
