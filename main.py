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
# - Integrate in taylor's logging code (for .wav raw files).
# - Implement a DAQ for wav file playback 

# - Add analog output to DAQ devices to support input and output/stimulation

# - Button to export of class events properties list to CSV (class ID - zero for non-classified events))
# - Allow for custom channel names

# - Test on Windows w/ a school laptop
#   - (Low Priority) install NIDAQmx drivers
#   - (Low Priority) simulate devices and write a windows only DAQ interface using pyDAQmx
# - Add LabJack Hardware Support (low priority)
