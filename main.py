import sys
from PySide6.QtWidgets import QApplication
from gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SpikeHound")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


#TODO Notes:
# - Add spectrogram view in analysis mode, also, simply fft amplitudes view
# - Button to export of events list to CSV
# - Buttons to pop up windows for screen shotting for reports
# - DAQ devices need to provide discrete sample rate settings rather than arbitrary ones
# - fix the scope from auto-rescaling when I add a channel
# - Fix vertical range and offset per channel in scope view
# - Complete trigger options in scope view or remove them
# - Allow for custom channel names
# !!- Add option to save/load layouts of the GUI

# - Feature for cross channel correlation
#   - option to pop out the traces into a separate figure

# - Neural network based spike detection and sorting.
#   - resolve overlaps of spikes

