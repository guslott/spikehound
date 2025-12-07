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

# - Add analog output to DAQ devices to support input and output/stimulation

#verify channel add after recording/filter settings issue