import sys
import gc
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
pg.setConfigOptions(useOpenGL=True, enableExperimental=True)
gc.set_threshold(1500, 15, 15)

def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SpikeHound")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

