"""SpectrogramTab - Per-channel spectrogram visualization.

Provides a compact three-panel view for a single channel:
- Left: Amplitude vs Frequency (FFT snapshot)
- Right: Amplitude vs Time (last window of data)  
- Bottom: Spectrogram (Frequency vs Time, color = power)

Data is driven from MainWindow's PlotManager data.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

# Match the app's color scheme
SCOPE_BACKGROUND_COLOR = QtGui.QColor(211, 230, 204)  # Light green from analysis tab


class SpectrogramTab(QtWidgets.QWidget):
    """
    Compact three-panel view for frequency analysis:

    Top-Left:  Amplitude vs Frequency (FFT snapshot)
    Top-Right: Amplitude vs Time (waveform)
    Bottom:    Spectrogram (Frequency vs Time, color = power)
    """

    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        channel_id: int,
        channel_name: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._channel_id = channel_id
        self._channel_name = channel_name

        # FFT / spectrogram parameters
        self._fft_size = 1024
        self._spec_columns = 200
        self._window = np.hanning(self._fft_size).astype(np.float32)

        # Display controls - will be set from dropdowns
        self._spec_dynamic_range_db = 60.0
        self._max_freq_hz = 5000.0

        # State
        self._sample_rate: float = 0.0
        self._cached_nyquist: float = 0.0
        self._freqs: Optional[np.ndarray] = None
        self._time_axis: np.ndarray = np.linspace(-1.0, 0.0, self._spec_columns)

        self._spec_data: np.ndarray = np.full(
            (self._fft_size // 2 + 1, self._spec_columns),
            -120.0,
            dtype=np.float32,
        )

        # Baseline subtraction
        self._use_baseline_subtraction = True
        self._baseline_alpha = 0.02
        self._baseline_db: Optional[np.ndarray] = None
        self._db_ceil = 30.0
        self._db_floor = -80.0

        # Build UI
        self._build_ui()

        # Timer for updates
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)  # ~20 Hz
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Header row: Title + Status + Controls
        header = self._build_header()
        main_layout.addWidget(header)

        # Top row: FFT and Time plots side-by-side
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)

        fft_container = self._build_fft_plot()
        top_row.addWidget(fft_container, stretch=1)

        time_container = self._build_time_plot()
        top_row.addWidget(time_container, stretch=1)

        main_layout.addLayout(top_row, stretch=2)

        # Bottom: Spectrogram (larger)
        spec_container = self._build_spectrogram()
        main_layout.addWidget(spec_container, stretch=5)

    def _build_header(self) -> QtWidgets.QWidget:
        """Build compact header with title, status, and controls in one row."""
        header = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(12)

        # Title
        title = QtWidgets.QLabel(f"<b>{self._channel_name}</b> – Frequency Analysis")
        title.setStyleSheet("font-size: 13px; color: #333;")
        layout.addWidget(title)

        # Status label
        self._sr_label = QtWidgets.QLabel("Fs: — Hz")
        self._sr_label.setStyleSheet("""
            font-size: 11px;
            color: #555;
            background: #E8E8E8;
            border: 1px solid #CCC;
            border-radius: 3px;
            padding: 2px 6px;
        """)
        layout.addWidget(self._sr_label)

        layout.addStretch()

        # Compact controls
        layout.addWidget(QtWidgets.QLabel("Freq Range:"))
        self._freq_combo = QtWidgets.QComboBox()
        self._freq_combo.setMinimumWidth(100)
        self._freq_combo.currentIndexChanged.connect(self._on_freq_changed)
        layout.addWidget(self._freq_combo)

        layout.addWidget(QtWidgets.QLabel("FFT:"))
        self._fft_combo = QtWidgets.QComboBox()
        for size in [256, 512, 1024, 2048, 4096]:
            self._fft_combo.addItem(f"{size}", size)
        self._fft_combo.setCurrentIndex(2)  # 1024
        self._fft_combo.currentIndexChanged.connect(self._on_fft_size_changed)
        layout.addWidget(self._fft_combo)

        layout.addWidget(QtWidgets.QLabel("Range:"))
        self._range_combo = QtWidgets.QComboBox()
        for db in [40, 60, 80, 100]:
            self._range_combo.addItem(f"{db} dB", db)
        self._range_combo.setCurrentIndex(1)  # 60 dB
        self._range_combo.currentIndexChanged.connect(self._on_range_changed)
        layout.addWidget(self._range_combo)

        # Baseline subtraction checkbox
        self._baseline_check = QtWidgets.QCheckBox("Baseline Sub")
        self._baseline_check.setChecked(True)
        self._baseline_check.toggled.connect(self._on_baseline_toggled)
        layout.addWidget(self._baseline_check)

        return header

    def _build_fft_plot(self) -> QtWidgets.QWidget:
        """Build the FFT frequency spectrum plot."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        label = QtWidgets.QLabel("Amplitude vs Frequency")
        label.setStyleSheet("font-weight: bold; font-size: 10px; color: #444;")
        layout.addWidget(label)

        self.fft_plot = pg.PlotWidget(enableMenu=False)
        self.fft_plot.setBackground(SCOPE_BACKGROUND_COLOR)
        self.fft_plot.showGrid(x=True, y=True, alpha=0.4)
        self.fft_plot.setLabel("bottom", "Frequency (Hz)")
        self.fft_plot.setLabel("left", "Amplitude")
        self.fft_plot.setMouseEnabled(x=False, y=True)
        self.fft_plot.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color=(60, 60, 60), width=1))
        self.fft_plot.getPlotItem().getAxis('left').setPen(pg.mkPen(color=(60, 60, 60), width=1))

        self._fft_curve = self.fft_plot.plot(
            pen=pg.mkPen((0, 100, 0), width=1.5),
            fillLevel=0,
            fillBrush=pg.mkBrush(0, 150, 0, 30)
        )

        try:
            self.fft_plot.hideButtons()
        except Exception:
            pass

        layout.addWidget(self.fft_plot)
        return container

    def _build_time_plot(self) -> QtWidgets.QWidget:
        """Build the time domain waveform plot."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        label = QtWidgets.QLabel("Amplitude vs Time")
        label.setStyleSheet("font-weight: bold; font-size: 10px; color: #444;")
        layout.addWidget(label)

        self.time_plot = pg.PlotWidget(enableMenu=False)
        self.time_plot.setBackground(SCOPE_BACKGROUND_COLOR)
        self.time_plot.showGrid(x=True, y=True, alpha=0.4)
        self.time_plot.setLabel("bottom", "Time (s)")
        self.time_plot.setLabel("left", "V")
        self.time_plot.setMouseEnabled(x=False, y=True)
        self.time_plot.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color=(60, 60, 60), width=1))
        self.time_plot.getPlotItem().getAxis('left').setPen(pg.mkPen(color=(60, 60, 60), width=1))

        self._time_curve = self.time_plot.plot(
            pen=pg.mkPen((0, 0, 139), width=1)
        )

        try:
            self.time_plot.hideButtons()
        except Exception:
            pass

        layout.addWidget(self.time_plot)
        return container

    def _build_spectrogram(self) -> QtWidgets.QWidget:
        """Build the spectrogram (waterfall) display without histogram."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        label = QtWidgets.QLabel("Spectrogram (Frequency vs Time)")
        label.setStyleSheet("font-weight: bold; font-size: 10px; color: #444;")
        layout.addWidget(label)

        # Create PlotItem
        view = pg.PlotItem()
        view.setLabel("bottom", "Time (s)")
        view.setLabel("left", "Frequency (Hz)")
        view.showGrid(x=True, y=True, alpha=0.3)
        view.getAxis('bottom').setPen(pg.mkPen(color=(60, 60, 60), width=1))
        view.getAxis('left').setPen(pg.mkPen(color=(60, 60, 60), width=1))

        self.spec_view = pg.ImageView(view=view)

        # HIDE all the extra UI elements including histogram
        try:
            self.spec_view.ui.menuBtn.hide()
            self.spec_view.ui.roiBtn.hide()
            self.spec_view.ui.roiPlot.hide()
            self.spec_view.ui.histogram.hide()  # Hide the color bar/histogram
        except Exception:
            pass

        # Light background
        self.spec_view.ui.graphicsView.setBackgroundBrush(
            pg.mkBrush(SCOPE_BACKGROUND_COLOR)
        )

        # Colormap
        try:
            cmap = pg.colormap.get("viridis")
        except Exception:
            cmap = pg.colormap.get("inferno")
        self.spec_view.setColorMap(cmap)

        self.spec_view.getView().invertY(False)
        self.spec_view.getView().setAspectLocked(False)

        self._update_image()

        layout.addWidget(self.spec_view)
        return container

    # ------------------------------------------------------------------
    # Control Handlers
    # ------------------------------------------------------------------

    def _update_freq_options(self, nyquist: float) -> None:
        """Update frequency range dropdown based on current Nyquist."""
        if abs(nyquist - self._cached_nyquist) < 1.0:
            return  # No significant change

        self._cached_nyquist = nyquist
        current_val = self._max_freq_hz

        # Build list of valid frequency options
        presets = [500, 1000, 2000, 5000, 10000, 20000]
        options: List[Tuple[str, float]] = []

        for freq in presets:
            if freq < nyquist - 1:  # Only include if strictly less than Nyquist
                if freq >= 1000:
                    label = f"{freq // 1000} kHz"
                else:
                    label = f"{freq} Hz"
                options.append((label, float(freq)))

        # Add Nyquist as the max option (with actual value, not "Nyquist")
        nyq_int = int(nyquist)
        if nyq_int >= 1000:
            nyq_label = f"{nyq_int // 1000} kHz"
            if nyq_int % 1000 != 0:
                nyq_label = f"{nyq_int / 1000:.1f} kHz"
        else:
            nyq_label = f"{nyq_int} Hz"
        options.append((nyq_label, float(nyq_int)))

        # Update combo box
        self._freq_combo.blockSignals(True)
        self._freq_combo.clear()
        
        best_idx = 0
        for i, (label, value) in enumerate(options):
            self._freq_combo.addItem(label, value)
            if value <= current_val:
                best_idx = i

        # Select the best matching option
        self._freq_combo.setCurrentIndex(best_idx)
        self._max_freq_hz = self._freq_combo.currentData() or nyquist
        self._freq_combo.blockSignals(False)

    def _on_freq_changed(self, index: int) -> None:
        """Handle frequency range selection."""
        val = self._freq_combo.currentData()
        if val:
            self._max_freq_hz = float(val)
            self._update_freq_range()

    def _on_fft_size_changed(self, index: int) -> None:
        """Handle FFT size change."""
        new_size = self._fft_combo.currentData()
        if new_size and new_size != self._fft_size:
            self._fft_size = new_size
            self._window = np.hanning(self._fft_size).astype(np.float32)
            self._spec_data = np.full(
                (self._fft_size // 2 + 1, self._spec_columns),
                -120.0,
                dtype=np.float32,
            )
            self._baseline_db = None
            self._freqs = None

    def _on_range_changed(self, index: int) -> None:
        """Handle dynamic range change."""
        val = self._range_combo.currentData()
        if val:
            self._spec_dynamic_range_db = float(val)
            self._update_image()

    def _on_baseline_toggled(self, checked: bool) -> None:
        """Handle baseline subtraction toggle."""
        self._use_baseline_subtraction = checked
        self._baseline_db = None

    def _update_freq_range(self) -> None:
        """Update frequency axis ranges on FFT and spectrogram."""
        fmax = self._max_freq_hz
        self.fft_plot.getPlotItem().setXRange(0, fmax, padding=0)
        try:
            self.spec_view.getView().setYRange(0, fmax, padding=0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Timer Callback
    # ------------------------------------------------------------------

    def _on_timer(self) -> None:
        """Update plots with latest data."""
        mw = self._main_window

        # Get samples from PlotManager
        plot_manager = getattr(mw, "_plot_manager", None)
        if plot_manager is not None:
            samples_dict = getattr(plot_manager, "channel_last_samples", {}) or {}
            times = getattr(plot_manager, "last_times", None)
        else:
            samples_dict = getattr(mw, "_channel_last_samples", {}) or {}
            times = getattr(mw, "_last_times", None)

        # Get sample rate
        sr = float(getattr(mw, "_current_sample_rate", 0.0) or 0.0)
        if sr <= 0 and self._sample_rate > 0:
            sr = float(self._sample_rate)

        # Update status and frequency options
        if sr > 0:
            nyquist = sr / 2.0
            self._sr_label.setText(f"Fs: {sr/1000:.1f} kHz")
            self._update_freq_options(nyquist)
        else:
            self._sr_label.setText("Fs: — Hz (waiting)")
            return

        # Check for data
        if not samples_dict or self._channel_id not in samples_dict:
            return

        y = np.asarray(samples_dict[self._channel_id], dtype=np.float32)
        if y.size < 4:
            return

        # Time axis
        if times is not None:
            t = np.asarray(times, dtype=np.float32)
            if t.size == y.size and t.size > 1:
                dt = float(t[-1] - t[0]) / max(t.size - 1, 1)
                if dt > 0:
                    self._sample_rate = 1.0 / dt
                t_rel = t - t[-1]
            else:
                t_rel = None
        else:
            t_rel = None

        sr = float(getattr(mw, "_current_sample_rate", 0.0) or 0.0)
        if sr <= 0:
            sr = self._sample_rate if self._sample_rate > 0 else 10_000.0

        window_sec = float(getattr(mw, "_current_window_sec", 1.0) or 1.0)

        # --- Time Domain Plot ---
        if t_rel is None:
            t_rel = np.linspace(-window_sec, 0.0, y.size, dtype=np.float32)

        self._time_curve.setData(t_rel, y)
        self.time_plot.getPlotItem().setXRange(-window_sec, 0.0, padding=0)

        # --- FFT Plot ---
        if y.size >= self._fft_size:
            segment = y[-self._fft_size:]
        else:
            segment = np.zeros(self._fft_size, dtype=np.float32)
            segment[-y.size:] = y

        windowed = segment * self._window
        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

        freqs = np.fft.rfftfreq(self._fft_size, d=1.0 / sr)
        self._freqs = freqs
        self._fft_curve.setData(freqs, mag)

        # Set FFT X range to match selected frequency range
        fmax = min(self._max_freq_hz, sr / 2)
        self.fft_plot.getPlotItem().setXRange(0, fmax, padding=0)

        # --- Spectrogram ---
        if freqs is None or freqs.size == 0:
            return

        needed_bins = freqs.shape[0]
        if self._spec_data.shape[0] != needed_bins:
            self._spec_data = np.full(
                (needed_bins, self._spec_columns),
                -120.0,
                dtype=np.float32,
            )

        # Baseline subtraction
        if self._use_baseline_subtraction:
            if self._baseline_db is None or self._baseline_db.shape != mag_db.shape:
                self._baseline_db = mag_db.copy()
            else:
                a = float(self._baseline_alpha)
                self._baseline_db = (1.0 - a) * self._baseline_db + a * mag_db
            disp_db = mag_db - self._baseline_db
        else:
            disp_db = mag_db

        mag_db_clipped = np.clip(disp_db, self._db_floor, self._db_ceil)

        self._spec_data = np.roll(self._spec_data, -1, axis=1)
        self._spec_data[:, -1] = mag_db_clipped.astype(np.float32, copy=False)

        self._time_axis = np.linspace(-window_sec, 0.0, self._spec_columns, dtype=np.float32)

        self._update_image()

        # Keep spectrogram in sync with frequency range
        self.spec_view.getView().setYRange(0, fmax, padding=0)

    # ------------------------------------------------------------------
    # Image Update
    # ------------------------------------------------------------------

    def _update_image(self) -> None:
        """Push spectrogram buffer to ImageView."""
        if self._spec_data.size == 0:
            return

        img = self._spec_data.astype(np.float32, copy=False)

        # Fixed levels based on dynamic range selection
        low = self._db_ceil - self._spec_dynamic_range_db
        levels = (float(low), float(self._db_ceil))

        self.spec_view.setImage(
            img.T,
            autoLevels=False,
            levels=levels,
            autoRange=False,
            autoHistogramRange=False,
        )

        if self._freqs is None or self._freqs.size < 2:
            return

        t0 = float(self._time_axis[0])
        t1 = float(self._time_axis[-1])
        f0 = float(self._freqs[0])
        f1 = float(self._freqs[-1])

        rect = QtCore.QRectF(min(t0, t1), min(f0, f1), abs(t1 - t0), abs(f1 - f0))
        self.spec_view.getImageItem().setRect(rect)

        v = self.spec_view.getView()
        v.setAspectLocked(False)
        v.setXRange(t0, t1, padding=0.0)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Stop timer when closed."""
        self._timer.stop()
        super().closeEvent(event)
