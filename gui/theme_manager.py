from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets


class ThemeManager:
    """Centralizes application-wide styling, palettes, and PyQtGraph configurations."""

    @staticmethod
    def apply_theme(widget: QtWidgets.QWidget) -> None:
        """Apply the global palette and stylesheet to the given widget."""
        # 1. Apply Palette
        palette = widget.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(200, 200, 200))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(223, 223, 223))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(200, 200, 200))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 220))
        palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(223, 223, 223))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(30, 144, 255))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        widget.setPalette(palette)

        # 2. Apply Application Stylesheet
        widget.setStyleSheet(ThemeManager.get_stylesheet())

        # 3. Apply Plot Styling (global static state in pg)
        ThemeManager.style_plot()

    @staticmethod
    def style_plot() -> None:
        """Configure PyQtGraph global options for high-performance rendering."""
        pg.setConfigOption("foreground", (0, 0, 139))
        # OpenGL provides hardware acceleration; antialias is efficient with OpenGL.
        pg.setConfigOptions(useOpenGL=True, enableExperimental=True, antialias=True)

    @staticmethod
    def get_stylesheet() -> str:
        """Return the global application stylesheet."""
        return """
            QMainWindow { background-color: rgb(200,200,200); }
            QWidget { color: rgb(0,0,0); }
            QGroupBox {
                background-color: rgb(223,223,223);
                border: 1px solid rgb(120, 120, 120);
                border-radius: 4px;
                margin-top: 12px;
                padding: 6px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0px 4px 0px 4px;
                color: rgb(128, 0, 0);
            }
            QLabel { color: rgb(0,0,0); }
            QPushButton {
                background-color: rgb(223,223,223);
                color: rgb(0,0,0);
                border: 1px solid rgb(120,120,120);
                padding: 4px 8px;
            }
            QPushButton:checked {
                background-color: rgb(200,200,200);
            }
            QLineEdit,
            QPlainTextEdit,
            QTextEdit,
            QAbstractSpinBox,
            QComboBox {
                color: rgb(0,0,0);
                background-color: rgb(245,245,245);
                selection-background-color: rgb(30,144,255);
                selection-color: rgb(255,255,255);
                border: 1px solid rgb(120,120,120);
                padding: 2px 4px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid rgb(120,120,120);
                background-color: rgb(223,223,223);
            }
            QComboBox QAbstractItemView {
                color: rgb(0,0,0);
                background-color: rgb(245,245,245);
                selection-background-color: rgb(30,144,255);
                selection-color: rgb(255,255,255);
            }
            QListView,
            QListWidget {
                color: rgb(0,0,0);
                background-color: rgb(245,245,245);
                selection-background-color: rgb(30,144,255);
                selection-color: rgb(255,255,255);
                border: 1px solid rgb(120,120,120);
            }
            QCheckBox,
            QRadioButton {
                color: rgb(0,0,0);
            }
            QSlider::groove:horizontal {
                border: 1px solid rgb(120,120,120);
                height: 6px;
                background: rgb(200,200,200);
                margin: 0px;
            }
            QSlider::handle:horizontal {
                background: rgb(30,144,255);
                border: 1px solid rgb(0,0,0);
                width: 14px;
                margin: -4px 0;
            }
            QStatusBar { background-color: rgb(192,192,192); }
        """
