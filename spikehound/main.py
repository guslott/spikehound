import gc
import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


# Enable OpenGL hardware acceleration for real-time plotting performance.
# This must be set before any PyQtGraph widgets are created.
pg.setConfigOptions(useOpenGL=True, enableExperimental=True)

# Tune garbage collection for real-time performance.
# Increase gen0 threshold to reduce frequency of small collections during streaming.
# Default is (700, 10, 10) - we raise gen0 to reduce per-frame GC pauses.
gc.set_threshold(1500, 15, 15)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SpikeHound")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
