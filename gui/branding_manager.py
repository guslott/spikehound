from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


class BrandingManager(QtCore.QObject):
    """Manages application branding elements, including the splash screen."""

    def __init__(self, splash_label: QtWidgets.QLabel, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._splash_label = splash_label
        self._splash_pixmap: Optional[QtGui.QPixmap] = None
        self._splash_aspect_ratio: float = 1.0
        
        self._init_splash()

    def _init_splash(self) -> None:
        """Initialize the splash pixmap and initial label state."""
        self._splash_pixmap = self._load_splash_pixmap()
        if self._splash_pixmap is None:
            self._splash_label.setText("Manlius Pebble Hill School\nCornell University")
            self._splash_label.setWordWrap(True)
            self._splash_label.setStyleSheet(
                "#splashLabel { border: 2px solid rgb(0,0,0); background-color: rgb(0,0,0); color: rgb(240,240,240); padding: 6px; font-weight: bold; }"
            )
        else:
            # First scale occurs in MainWindow's resizeEvent via the singleShot timer
            pass

    def _load_splash_pixmap(self) -> Optional[QtGui.QPixmap]:
        """Load the splash image from the media directory."""
        # Assume media is at ../media relative to gui module
        splash_path = Path(__file__).resolve().parent.parent / "media" / "mph_cornell_splash.png"
        if not splash_path.exists():
            logger.debug("Splash image not found at %s", splash_path)
            return None
            
        pixmap = QtGui.QPixmap(str(splash_path))
        if pixmap.isNull():
            logger.debug("Failed to load splash pixmap from %s", splash_path)
            return None
            
        if pixmap.height() > 0:
            self._splash_aspect_ratio = pixmap.width() / float(pixmap.height())
        return pixmap

    def update_splash_pixmap(self) -> None:
        """Rescale the splash image to match the current label dimensions."""
        if self._splash_label is None or self._splash_pixmap is None or self._splash_pixmap.isNull():
            return

        label_rect = self._splash_label.contentsRect()
        available_width = label_rect.width()
        
        # Fallbacks for zero-width rectangles (e.g. before initial show)
        if available_width <= 0:
            available_width = self._splash_label.width()
        if available_width <= 0 and self._splash_label.parentWidget() is not None:
            available_width = self._splash_label.parentWidget().width()
        if available_width <= 0:
            available_width = 200

        border_px = 4  # Matches 2px border on each side in stylesheet
        available_width = int(max(50, available_width - border_px))

        # Use loaded aspect ratio or calculate on the fly
        aspect = self._splash_aspect_ratio
        if aspect <= 0:
            aspect = self._splash_pixmap.width() / max(1, self._splash_pixmap.height())
            
        target_height = max(1, int(round(available_width / aspect)))

        scaled = self._splash_pixmap.scaled(
            available_width,
            target_height,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        
        if scaled.isNull():
            return

        total_height = scaled.height() + border_px
        
        # We need to block recursive resizeEvents if we were modifying the parent directly,
        # but setting min/max height on the label is usually safe.
        self._splash_label.setMinimumHeight(total_height)
        self._splash_label.setMaximumHeight(total_height)
        self._splash_label.setPixmap(scaled)
        self._splash_label.updateGeometry()
