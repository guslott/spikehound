"""RecordingControlWidget - Manages recording controls for WAV file capture.

Extracted from MainWindow to provide a focused component for:
- File path selection with browse button
- Auto-increment filename checkbox
- Recording toggle with duration timer
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets


class RecordingControlWidget(QtWidgets.QGroupBox):
    """Widget encapsulating recording controls and state.
    
    Signals:
        recordingStarted: Emitted with (path, rollover) when recording begins
        recordingStopped: Emitted when recording ends
    """

    recordingStarted = QtCore.Signal(str, bool)  # path, rollover
    recordingStopped = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Recording", parent)
        
        self._recording_start_time: Optional[float] = None
        self._recording_timer: Optional[QtCore.QTimer] = None
        self._controller = None  # Will be set by parent for duration queries
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # File path row
        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setMaximumWidth(220)
        self.path_edit.setPlaceholderText("Select output file...")
        path_row.addWidget(self.path_edit, 1)
        
        self.browse_btn = QtWidgets.QPushButton("Browse…")
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(self.browse_btn)
        layout.addLayout(path_row)

        # Note: Auto-increment and Pro options moved to Settings tab

        # Record toggle button
        self.toggle_btn = QtWidgets.QPushButton("Start Recording")
        self.toggle_btn.setCheckable(True)
        self._apply_button_style(False)
        self.toggle_btn.setEnabled(False)  # Disabled until filename is set
        self.toggle_btn.clicked.connect(self._on_toggle)
        self.path_edit.textChanged.connect(self._update_button_enabled)
        layout.addWidget(self.toggle_btn)

        layout.addStretch(1)

    def set_controller(self, controller) -> None:
        """Set the pipeline controller for querying recording duration."""
        self._controller = controller

    def set_enabled_for_recording(self, enabled: bool) -> None:
        """Enable/disable controls (except toggle button during recording)."""
        self.path_edit.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)


    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select Recording File", "", "WAV Files (*.wav);;HDF5 (*.h5);;All Files (*)"
        )
        if path:
            self.path_edit.setText(path)

    def _update_button_enabled(self, text: str = "") -> None:
        """Enable/disable the record button based on whether a filename is set."""
        has_path = bool(self.path_edit.text().strip())
        self.toggle_btn.setEnabled(has_path)

    def _on_toggle(self, checked: bool) -> None:
        if checked:
            path = self.path_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.information(
                    self, "Recording", "Please choose a file path before recording."
                )
                self.toggle_btn.setChecked(False)
                return
            
            rollover = True
            if self._controller:
                rollover = self._controller.app_settings.recording_auto_increment
            
            # Auto-increment filename if enabled
            if rollover:
                path = self._get_next_filename(path)
            
            self._start_timer()
            self._apply_button_style(True)
            self.recordingStarted.emit(path, rollover)
        else:
            self._stop_timer()
            self._apply_button_style(False)
            self.recordingStopped.emit()

    # -------------------------------------------------------------------------
    # Recording timer
    # -------------------------------------------------------------------------

    def _start_timer(self) -> None:
        """Start the recording duration timer."""
        self._recording_start_time = time.perf_counter()
        if self._recording_timer is None:
            self._recording_timer = QtCore.QTimer(self)
            self._recording_timer.timeout.connect(self._update_duration)
        self._recording_timer.start(1000)  # Update every second

    def _stop_timer(self) -> None:
        """Stop the recording duration timer."""
        if self._recording_timer is not None:
            self._recording_timer.stop()
        self._recording_start_time = None

    def _update_duration(self) -> None:
        """Update the record button with elapsed time based on actual data logged."""
        if self._controller is None:
            return
        
        # Get actual duration from controller (based on frames written)
        elapsed = self._controller.recording_duration_seconds
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.toggle_btn.setText(f"Stop Recording ({minutes:02d}:{seconds:02d})")

    def _apply_button_style(self, recording: bool) -> None:
        if recording:
            self.toggle_btn.setText("Stop Recording (00:00)")
            self.toggle_btn.setStyleSheet(
                "background-color: rgb(46,204,113); color: rgb(0,0,0); font-weight: bold;"
            )
        else:
            self.toggle_btn.setText("Start Recording")
            self.toggle_btn.setStyleSheet(
                "background-color: rgb(220, 20, 60); color: rgb(0,0,0); font-weight: bold;"
            )

    # -------------------------------------------------------------------------
    # Filename utilities
    # -------------------------------------------------------------------------

    def _get_next_filename(self, base_path: str) -> str:
        """
        Find the next available filename with auto-increment.
        
        Given 'path/to/file.wav', checks for:
        - file.wav (returns this if doesn't exist)
        - file1.wav
        - file2.wav
        - etc.
        """
        directory = os.path.dirname(base_path) or "."
        basename = os.path.basename(base_path)
        name, ext = os.path.splitext(basename)
        
        # If the base file doesn't exist, use it
        if not os.path.exists(base_path):
            return base_path
        
        # Find all existing files matching the pattern
        pattern = re.compile(rf"^{re.escape(name)}(\d*)\.wav$", re.IGNORECASE)
        max_num = 0
        
        try:
            for filename in os.listdir(directory):
                match = pattern.match(filename)
                if match:
                    num_str = match.group(1)
                    if num_str:
                        max_num = max(max_num, int(num_str))
                    else:
                        # Base file exists, we need at least 1
                        max_num = max(max_num, 0)
        except OSError:
            pass
        
        # Next number is max + 1 (but at least 1 if base file exists)
        next_num = max_num + 1
        new_path = os.path.join(directory, f"{name}{next_num}{ext}")
        return new_path


__all__ = ["RecordingControlWidget"]
