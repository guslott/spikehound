import sys
from PySide6 import QtWidgets
from gui.scope_window import ScopeMainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = ScopeMainWindow()
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
