from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets


class BYBDebugTab(QtWidgets.QWidget):
    TAB_TITLE = "BYB Debug"

    def __init__(self, runtime, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._fields: dict[str, QtWidgets.QLabel] = {}
        self._messages = QtWidgets.QPlainTextEdit(self)
        self._messages.setReadOnly(True)
        self._messages.setMaximumBlockCount(64)
        self._build_ui()
        self._timer.start()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        for title, key in [
            ("Profile", "profile"),
            ("Transport", "transport"),
            ("Hardware", "hardware_type"),
            ("Firmware", "firmware_version"),
            ("Board", "board_type"),
            ("Reported Rate", "reported_sample_rate"),
            ("Reported Channels", "reported_channel_count"),
            ("Configured Rate", "sample_rate"),
            ("Locked Width", "stream_channels"),
            ("Frame Mode", "decoder_frame_mode"),
            ("Measured Byte Rate", "measured_byte_rate"),
            ("Measured Frame Rate", "measured_frame_rate"),
            ("Capture Path", "capture_path"),
        ]:
            label = QtWidgets.QLabel("–")
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self._fields[key] = label
            form.addRow(f"{title}:", label)

        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("Recent Protocol Messages:"))
        layout.addWidget(self._messages, 1)

    def _clear(self) -> None:
        for label in self._fields.values():
            label.setText("–")
        self._messages.clear()

    @staticmethod
    def _fmt_rate(value, *, unit: str) -> str:
        if value is None:
            return "–"
        numeric = float(value)
        if unit == "Hz":
            return f"{numeric:,.1f} Hz"
        return f"{numeric:,.1f} B/s"

    def refresh(self) -> None:
        driver = getattr(self.runtime, "daq_source", None)
        if driver is None:
            self._clear()
            return
        try:
            stats = driver.stats()
        except Exception:
            self._clear()
            return
        if stats.get("profile") is None:
            self._clear()
            return

        for key in ("profile", "transport", "hardware_type", "firmware_version", "capture_path"):
            self._fields[key].setText(str(stats.get(key) or "–"))
        for key in ("board_type", "reported_sample_rate", "reported_channel_count", "stream_channels"):
            value = stats.get(key)
            self._fields[key].setText("–" if value is None else str(value))
        configured_rate = stats.get("sample_rate")
        self._fields["sample_rate"].setText("–" if configured_rate is None else f"{float(configured_rate):,.1f} Hz")
        self._fields["decoder_frame_mode"].setText(str(stats.get("decoder_frame_mode") or "–"))
        self._fields["measured_byte_rate"].setText(self._fmt_rate(stats.get("measured_byte_rate"), unit="B/s"))
        self._fields["measured_frame_rate"].setText(self._fmt_rate(stats.get("measured_frame_rate"), unit="Hz"))

        messages = stats.get("last_messages") or []
        text = "\n".join(str(message) for message in messages[-24:])
        if text != self._messages.toPlainText():
            self._messages.setPlainText(text)
