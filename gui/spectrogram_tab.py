from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


class SpectrogramTab(QtWidgets.QWidget):
    """
    Three-panel view for a single channel, driven directly from MainWindow's data:

    Top:    Amplitude vs Frequency (FFT snapshot)
    Middle: Amplitude vs Time (last window of data)
    Bottom: Spectrogram (Frequency vs Time, color = power)
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

        # Display controls
        self._spec_dynamic_range_db = 80.0
        self._spec_floor_db = -120.0

        # State
        self._sample_rate: float = 0.0
        self._freqs: Optional[np.ndarray] = None
        self._time_axis: np.ndarray = np.linspace(-1.0, 0.0, self._spec_columns)

        self._spec_data: np.ndarray = np.full(
            (self._fft_size // 2 + 1, self._spec_columns),
            self._spec_floor_db,
            dtype=np.float32,
        )

        # ---- Better visibility controls ----
        self._use_baseline_subtraction = True
        self._baseline_alpha = 0.02
        self._baseline_db: Optional[np.ndarray] = None

        # baseline-subtracted display can be positive
        self._db_ceil = 30.0

        # levels (keep this stable at first; auto-percentile can go flat when image is mostly floor)
        self._auto_levels = False
        self._level_smooth_alpha = 0.15
        self._level_low: Optional[float] = None
        self._level_high: Optional[float] = None

        # Build UI ONCE
        self._build_ui()

        # Timer ONCE
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Title
        title = QtWidgets.QLabel(f"Spectrogram – {self._channel_name}")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        # --- Status line: Sample rate + Nyquist ---
        self._sr_label = QtWidgets.QLabel("Fs: — Hz    Nyquist: — Hz")
        self._sr_label.setAlignment(QtCore.Qt.AlignCenter)
        self._sr_label.setStyleSheet("font-size: 11px; color: #DDD; background: #222; padding: 4px;")
        layout.addWidget(self._sr_label)

        # --- Controls row ---
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QtWidgets.QLabel("Freq Range"))

        self._range_combo = QtWidgets.QComboBox()
        # data = ("mode", value) where mode is "nyquist" or "fixed"
        self._range_combo.addItem("0–Nyquist", ("nyquist", 0.0))
        self._range_combo.addItem("0–5000 Hz (default)", ("fixed", 5000.0))
        self._range_combo.addItem("0–2000 Hz", ("fixed", 2000.0))
        self._range_combo.addItem("0–1000 Hz", ("fixed", 1000.0))
        self._range_combo.setCurrentIndex(1)  # default = 0–5000
        self._range_combo.currentIndexChanged.connect(self._on_controls_changed)
        controls.addWidget(self._range_combo)

        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Contrast (dB)"))

        self._contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._contrast_slider.setRange(40, 140)  # dB range
        self._contrast_slider.setValue(int(round(self._spec_dynamic_range_db)))
        self._contrast_slider.setFixedWidth(160)
        self._contrast_slider.valueChanged.connect(self._on_controls_changed)
        controls.addWidget(self._contrast_slider)

        self._contrast_value = QtWidgets.QLabel(f"{int(self._spec_dynamic_range_db)}")
        self._contrast_value.setFixedWidth(30)
        controls.addWidget(self._contrast_value)

        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Floor (dB)"))

        self._floor_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._floor_slider.setRange(-180, -20)
        self._floor_slider.setValue(int(round(self._spec_floor_db)))
        self._floor_slider.setFixedWidth(160)
        self._floor_slider.valueChanged.connect(self._on_controls_changed)
        controls.addWidget(self._floor_slider)

        self._floor_value = QtWidgets.QLabel(f"{int(self._spec_floor_db)}")
        self._floor_value.setFixedWidth(40)
        controls.addWidget(self._floor_value)

        controls.addStretch(1)
        layout.addLayout(controls)

        # =======================
        # Top: FFT (Amplitude vs Frequency)
        # =======================
        self.fft_plot = pg.PlotWidget(enableMenu=False)

        try:
            self.fft_plot.hideButtons()
        except Exception:
            pass

        self.fft_plot.setBackground("k")
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self.fft_plot.setLabel("left", "Amplitude", units="dB")
        self.fft_plot.setMouseEnabled(x=False, y=False)
        self._fft_curve = self.fft_plot.plot(
            pen=pg.mkPen((0, 255, 0), width=2)
        )
        layout.addWidget(self.fft_plot, stretch=2)

        # =======================
        # Middle: Time-domain trace
        # =======================
        self.time_plot = pg.PlotWidget(enableMenu=False)
        try:
            self.time_plot.hideButtons()
        except Exception:
            pass

        self.time_plot.setBackground("k")
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.time_plot.setLabel("bottom", "Time", units="s")
        self.time_plot.setLabel("left", "Amplitude", units="V")
        self.time_plot.setMouseEnabled(x=False, y=False)
        self._time_curve = self.time_plot.plot(
            pen=pg.mkPen((0, 200, 255), width=1)
        )
        layout.addWidget(self.time_plot, stretch=2)

        # =======================
        # Bottom: Spectrogram
        # =======================
        view = pg.PlotItem()
        view.setLabel("bottom", "Time", units="s")
        view.setLabel("left", "Frequency", units="Hz")
        view.showGrid(x=True, y=True, alpha=0.25)

        self.spec_view = pg.ImageView(view=view)

        # Hide unwanted UI elements
        try:
            self.spec_view.ui.menuBtn.hide()
            self.spec_view.ui.roiBtn.hide()
            self.spec_view.ui.roiPlot.hide()
        except Exception:
            pass

        # Shrink histogram so spectrogram fills the space
        try:
            self.spec_view.ui.histogram.setMaximumWidth(80)
            self.spec_view.ui.histogram.setMinimumWidth(60)
        except Exception:
            pass

        # Dark background
        self.spec_view.ui.graphicsView.setBackgroundBrush(
            pg.mkBrush("k")
        )

        # High-contrast colormap
        try:
            cmap = pg.colormap.get("inferno")
        except Exception:
            cmap = pg.colormap.get("viridis")
        self.spec_view.setColorMap(cmap)

        # Allow non-square pixels so it fills the area
        self.spec_view.getView().setAspectLocked(False)

        # Initialize image
        self._update_image()

        layout.addWidget(self.spec_view, stretch=4)

    def _on_controls_changed(self) -> None:
        # Update stored settings from sliders
        self._spec_dynamic_range_db = float(self._contrast_slider.value())
        self._spec_floor_db = float(self._floor_slider.value())

        # Update little value labels
        self._contrast_value.setText(str(int(self._spec_dynamic_range_db)))
        self._floor_value.setText(str(int(self._spec_floor_db)))

        # Redraw immediately using current buffer
        self._update_image()


 
    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------

    def _on_timer(self) -> None:
        """
        Called ~20 times per second.
        Grabs the latest samples for this channel from MainWindow and
        updates all three plots.
        """
        
        mw = self._main_window

        samples_dict = getattr(mw, "_channel_last_samples", {}) or {}
        times = getattr(mw, "_last_times", None)

        # Always show sample rate / Nyquist, even if we have no samples yet
        sr = float(getattr(mw, "_current_sample_rate", 0.0) or 0.0)
        if sr <= 0 and self._sample_rate > 0:
            sr = float(self._sample_rate)

        if sr > 0:
            self._sr_label.setText(f"Fs: {sr:,.0f} Hz    Nyquist: {sr/2:,.0f} Hz")
        else:
            self._sr_label.setText("Fs: — Hz    Nyquist: — Hz")

        # If no data, say so (instead of silently returning)
        if not isinstance(samples_dict, dict) or not samples_dict:
            self._sr_label.setText(self._sr_label.text() + "   (waiting for data — start streaming)")
            return

        # If wrong channel_id, say what channels we actually have
        if self._channel_id not in samples_dict:
            have = sorted(list(samples_dict.keys()))
            self._sr_label.setText(self._sr_label.text() + f"   (channel {self._channel_id} not live; have {have})")
            return

        y = np.asarray(samples_dict[self._channel_id], dtype=np.float32)
        if y.size < 4:
            self._sr_label.setText(self._sr_label.text() + "   (not enough samples yet)")
            return


        # Time axis from MainWindow if available
        if times is not None:
            t = np.asarray(times, dtype=np.float32)
            if t.size == y.size and t.size > 1:
                dt = float(t[-1] - t[0]) / max(t.size - 1, 1)
                if dt > 0:
                    self._sample_rate = 1.0 / dt
                t_rel = t - t[-1]  # newest point at 0s
            else:
                t_rel = None
        else:
            t_rel = None

        # Prefer MainWindow's current sample rate if available
        sr = float(getattr(mw, "_current_sample_rate", 0.0) or 0.0)
        if sr <= 0 and self._sample_rate > 0:
            sr = float(self._sample_rate)
        if sr <= 0:
            sr = 10_000.0  # last-resort default

        nyquist = sr / 2.0
        self._sr_label.setText(f"Fs: {sr:,.0f} Hz    Nyquist: {nyquist:,.0f} Hz")


        # Use MainWindow's window if present
        window_sec = float(getattr(mw, "_current_window_sec", 1.0) or 1.0)

        # --- Middle plot: time trace ---
        if t_rel is None:
            # fabricate relative time axis if we don't have one
            dt = window_sec / max(y.size, 1)
            t_rel = np.linspace(-window_sec, 0.0, y.size, dtype=np.float32)

        self._time_curve.setData(t_rel, y)
        self.time_plot.getPlotItem().setXRange(-window_sec, 0.0, padding=0)

        # --- Top plot: FFT of most recent window ---
        if y.size >= self._fft_size:
            segment = y[-self._fft_size :]
        else:
            segment = np.zeros(self._fft_size, dtype=np.float32)
            segment[-y.size :] = y

        windowed = segment * self._window
        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

        freqs = np.fft.rfftfreq(self._fft_size, d=1.0 / sr)
        self._freqs = freqs
        self._fft_curve.setData(freqs, mag_db)
        fmin, fmax = self._current_freq_limits(sr)
        self.fft_plot.getPlotItem().setXRange(fmin, fmax, padding=0)


        # --- Bottom plot: append this FFT slice into the spectrogram image ---
        if freqs is None or freqs.size == 0:
            return

        needed_bins = freqs.shape[0]
        if self._spec_data.shape[0] != needed_bins:
            self._spec_data = np.full(
                (needed_bins, self._spec_columns),
                self._spec_floor_db,
                dtype=np.float32,
            )

        # --- KEY FIX: baseline subtraction so changes are obvious ---
        if self._use_baseline_subtraction:
            if self._baseline_db is None or self._baseline_db.shape != mag_db.shape:
                self._baseline_db = mag_db.copy()
            else:
                a = float(self._baseline_alpha)
                self._baseline_db = (1.0 - a) * self._baseline_db + a * mag_db

            # Show "excess power" above baseline (clearer than per-frame normalization)
            disp_db = mag_db - self._baseline_db
        else:
            disp_db = mag_db

        # Clip for display
        mag_db_clipped = np.clip(disp_db, self._spec_floor_db, self._db_ceil)

        # Roll and insert newest column on the right
        self._spec_data = np.roll(self._spec_data, -1, axis=1)
        self._spec_data[:, -1] = mag_db_clipped.astype(np.float32, copy=False)

        # Time axis for spectrogram columns
        self._time_axis = np.linspace(-window_sec, 0.0, self._spec_columns, dtype=np.float32)

        self._update_image()


        # Keep spectrogram view pinned to chosen frequency range
        fmin, fmax = self._current_freq_limits(sr)
        self.spec_view.getView().setYRange(fmin, fmax, padding=0)


    def _current_freq_limits(self, sample_rate: float) -> tuple[float, float]:
        """
        Returns (fmin, fmax) for display, based on dropdown and Nyquist.
        """
        nyq = max(sample_rate * 0.5, 1.0)

        mode, val = self._range_combo.currentData()
        if mode == "nyquist":
            fmax = nyq
        else:
            # fixed Hz (like 5000) but never above Nyquist
            fmax = min(float(val), nyq)
        fmin = 0.0
        return fmin, fmax


    # ------------------------------------------------------------------
    # Image helper
    # ------------------------------------------------------------------

    def _update_image(self) -> None:
        """Push the spectrogram buffer into the ImageView (time/freq mapped correctly)."""
        if self._spec_data.size == 0:
            return

        # self._spec_data is (freq_bins, time_cols)
        img = self._spec_data.astype(np.float32, copy=False)

        # Robust contrast: use percentiles so a few spikes don't wash out the image
        if getattr(self, "_auto_levels", True):
            lo = float(np.percentile(img, 5))
            hi = float(np.percentile(img, 99))

            if hi - lo < 1e-6:
                hi = lo + 1.0

            a = float(getattr(self, "_level_smooth_alpha", 0.15))
            level_low = getattr(self, "_level_low", None)
            level_high = getattr(self, "_level_high", None)

            if level_low is None or level_high is None:

                self._level_low, self._level_high = lo, hi
            else:
                self._level_low = (1.0 - a) * self._level_low + a * lo
                self._level_high = (1.0 - a) * self._level_high + a * hi

            levels = (float(self._level_low), float(self._level_high))
        else:
            # Manual / fixed window mode
            low = max(self._spec_floor_db, self._db_ceil - float(self._spec_dynamic_range_db))
            levels = (float(low), float(self._db_ceil))

        # Feed (time, freq) to ImageView: transpose -> (time_cols, freq_bins)
        self.spec_view.setImage(
            img.T,
            autoLevels=False,
            levels=levels,
            autoRange=False,          # <-- important
            autoHistogramRange=False, # <-- important
        )



        # Map pixels to real units using a rect: x=time (s), y=freq (Hz)
        if self._freqs is None or self._freqs.size < 2:
            return

        t0 = float(self._time_axis[0])
        t1 = float(self._time_axis[-1])
        f0 = float(self._freqs[0])
        f1 = float(self._freqs[-1])

        rect = QtCore.QRectF(min(t0, t1), min(f0, f1), abs(t1 - t0), abs(f1 - f0))
        self.spec_view.getImageItem().setRect(rect)

        # Fit the view to the image every update (keeps it filling the frame)
        v = self.spec_view.getView()
        v.setAspectLocked(False)

        v = self.spec_view.getView()
        v.setAspectLocked(False)

        # Keep X axis matched to our time window EVERY frame (prevents "tiny sliver")
        v.setXRange(t0, t1, padding=0.0)

        # Keep Y pinned to current freq range (optional; you already do this in _on_timer)
        # v.setYRange(fmin, fmax, padding=0.0)



