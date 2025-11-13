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
# - Add a debug/settings tab that optionally pops up to describe the health of the 
#       application as well as the settings like output source, etc.
# - Button to export of events list to CSV
# - Buttons to pop up windows for screen shotting
# - DAQ devices need to provide discrete sample rate settings rather than arbitrary ones