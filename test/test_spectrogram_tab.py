"""Regression tests for gui/spectrogram_tab.py (finding 7.1)."""
from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from gui.spectrogram_tab import SpectrogramTab


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _FakePlotManager:
    def __init__(self, channel_id: int, n: int = 1024, sr: float = 10000.0) -> None:
        self.channel_last_samples = {
            channel_id: np.random.randn(n).astype(np.float32)
        }
        self.last_times = None
        self.sample_rate = sr
        self.window_sec = 1.0


class _FakeMainWindow:
    def __init__(self, channel_id: int) -> None:
        self._plot_manager = _FakePlotManager(channel_id)


def test_spectrogram_realloc_is_double_wide_no_indexerror(monkeypatch) -> None:
    """Finding 7.1: when the freq-bin count changes outside _on_fft_size_changed,
    the spectrogram buffer must be reallocated DOUBLE-wide.

    The buffer uses a double-wide circular-write trick (each column is written at
    both ``col`` and ``col + _spec_columns``). A single-wide realloc on the timer
    hot path used to index out of bounds on the very next write, throwing an
    IndexError inside the QTimer callback.
    """
    _app()
    channel_id = 0
    widget = SpectrogramTab(_FakeMainWindow(channel_id), channel_id, "ch0")
    try:
        # Treat the tab as visible so _on_timer does real FFT/spectrogram work.
        monkeypatch.setattr(widget, "isVisible", lambda: True)

        # Simulate the drift the finding describes: the buffer's bin count no
        # longer matches freqs.shape[0], forcing the realloc path. Use a non-zero
        # write cursor to mimic steady state.
        widget._spec_data = np.full(
            (999, 2 * widget._spec_columns), -120.0, dtype=np.float32
        )
        widget._spec_col_idx = 5

        # Must not raise (previously: IndexError at the `col + _spec_columns` write).
        widget._on_timer()

        needed_bins = widget._fft_size // 2 + 1
        assert widget._spec_data.shape == (needed_bins, 2 * widget._spec_columns), (
            "realloc must be double-wide to satisfy the circular-write contract"
        )
        # The realloc must reset the write cursor (mirroring _on_fft_size_changed).
        assert 0 <= widget._spec_col_idx < widget._spec_columns

        # A second tick (steady-state write, no realloc) must also stay in bounds.
        widget._on_timer()
    finally:
        widget.close()
        _app().processEvents()
