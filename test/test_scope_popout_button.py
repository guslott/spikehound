from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from gui.analysis_tab import ClusterWaveformDialog
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


def test_waveform_dialog_supports_multicolor_scope_mode() -> None:
    _app()
    t = np.linspace(0.0, 0.05, 16, dtype=np.float64)
    dialog = ClusterWaveformDialog(
        None,
        "Scope traces",
        [
            (t, np.linspace(0.2, 0.4, 16, dtype=np.float32)),
            (t, np.linspace(0.6, 0.8, 16, dtype=np.float32), None),
        ],
        show_median=False,
        y_label="Scope position",
        y_units="",
    )

    assert dialog._median_waveform is None
    assert len(dialog._aligned_samples) == 2
    assert len(dialog._aligned_colors) == 2


def test_waveform_dialog_supports_summary_only_mode() -> None:
    _app()
    t = np.linspace(-0.01, 0.01, 21, dtype=np.float64)
    dialog = ClusterWaveformDialog(
        None,
        "Autocorrelogram",
        [],
        show_median=False,
        y_label="Count",
        y_units="",
        summary_waveforms=[
            ("Autocorrelogram", t, np.arange(t.size, dtype=np.float32), None),
        ],
        count_override=3,
    )

    assert dialog._plot_time_axis is not None
    assert dialog._export_time_axis is not None
    assert "3 events" in dialog.windowTitle()
