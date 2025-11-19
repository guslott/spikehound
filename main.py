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
# - Button to export raw trace data for waveforms to CSV
# - Buttons to pop up windows for screen shotting for reports
# - DAQ devices need to provide discrete sample rate settings rather than arbitrary ones
# - Allow for custom channel names

# - Add PCA features (top two components) to analysis tab
#   - PCA needs to be debugged in the case that the user clears all the events.

# - Feature for cross channel correlation
#   - option to pop out the traces into a separate figure

# - Automated spike sorting (advanced analysis tab)
#   - Could involve human support for identifying class types
#   - clustering algorithms (k-means, hierarchical, etc)
#   - Overlap detection and deconvolution
#   - Build templates and align spikes to templates for defining event start times
#   - Will want to build a model of noise for each channel as well

# - Neural network based spike detection and sorting.
#   - resolve overlaps of spikes

# - Test on Windows
#   - Get school laptop and install NIDAQmx drivers
#   - simulate devices and write a windows only DAQ interface using pyDAQmx

