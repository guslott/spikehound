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
# - (Taylor) Add spectrogram view in separate tab, also, simply fft amplitudes view
# - Add analog output to DAQ devices to support input and output/stimulation

# - when disconnecting from file device, play controls remain visible