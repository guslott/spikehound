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
#    - Add spectrogram button next to analyze and have it pop up a spectrogram tab for a channel

# - Button to export of class events properties list to CSV (class ID - zero for non-classified events))
# - Allow for custom channel names

# - Fix how sample rates are handled
# - DAQ devices need to provide discrete sample rate settings rather than arbitrary ones

# - Add PCA features (top two components) to analysis tab
#   - PCA needs to be debugged in the case that the user clears all the events.
#   - PCA is a pain, remove it for now


# - Test on Windows
#   - Get school laptop and install NIDAQmx drivers
#   - simulate devices and write a windows only DAQ interface using pyDAQmx

