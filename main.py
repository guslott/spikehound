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
# - Add spectrogram view in analysis mode
# - Button to export of events list to CSV
# - Buttons to pop up windows for screen shotting
# - DAQ devices need to provide discrete sample rate settings rather than arbitrary ones

# - Features supporting individual spike classes.
#   - Option to view waveforms of individual all overlayed together in a plot aligned to their peaks
#       - Should shrink the analysis scatter to left and create an axis that shows the spikes together
#       - Support exporting these spikes in a class separately to a CSV file

# - Feature for cross channel correlation
#   - Spike triggered averaging of one channel based on spikes detected in another channel