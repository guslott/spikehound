from __future__ import annotations

from typing import Dict, Optional

from PySide6 import QtCore, QtWidgets


class AnalysisDock(QtWidgets.QDockWidget):
    """Dockable workspace containing the scope tab plus ad-hoc analysis tabs."""

    def __init__(self, title: str = "Workspace", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(title, parent)
        self.setObjectName("AnalysisDock")
        self.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea
            | QtCore.Qt.RightDockWidgetArea
            | QtCore.Qt.TopDockWidgetArea
            | QtCore.Qt.BottomDockWidgetArea
        )
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self.setWidget(self._tabs)

        self._scope_widget: Optional[QtWidgets.QWidget] = None
        self._tab_info: Dict[QtWidgets.QWidget, str] = {}
        self._analysis_count: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Scope management
    # ------------------------------------------------------------------

    def set_scope_widget(self, widget: QtWidgets.QWidget, title: str = "Scope") -> None:
        if self._scope_widget is not None:
            idx = self._tabs.indexOf(self._scope_widget)
            if idx >= 0:
                self._tabs.removeTab(idx)
        self._scope_widget = widget
        widget.setParent(self._tabs)
        self._tabs.insertTab(0, widget, title)
        self._tabs.setCurrentIndex(0)
        tab_bar = self._tabs.tabBar()
        tab_bar.setTabButton(0, QtWidgets.QTabBar.RightSide, None)

    def select_scope(self) -> None:
        if self._scope_widget is None:
            return
        idx = self._tabs.indexOf(self._scope_widget)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Analysis tab management
    # ------------------------------------------------------------------

    def open_analysis(self, channel_name: str) -> QtWidgets.QWidget:
        count = self._analysis_count.get(channel_name, 0) + 1
        self._analysis_count[channel_name] = count
        title = f"Analysis - {channel_name}" if count == 1 else f"Analysis - {channel_name} #{count}"

        widget = QtWidgets.QWidget(self._tabs)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addStretch(1)

        insert_index = 1 if self._scope_widget is not None else self._tabs.count()
        self._tabs.insertTab(insert_index, widget, title)
        self._tabs.setCurrentWidget(widget)
        self._tab_info[widget] = channel_name
        return widget

    def close_tab(self, channel_name: str) -> None:
        removed = False
        for widget, mapped_name in list(self._tab_info.items()):
            if mapped_name != channel_name:
                continue
            index = self._tabs.indexOf(widget)
            if index >= 0:
                self._tabs.removeTab(index)
            widget.deleteLater()
            del self._tab_info[widget]
            current = self._analysis_count.get(channel_name, 1) - 1
            if current > 0:
                self._analysis_count[channel_name] = current
            else:
                self._analysis_count.pop(channel_name, None)
            removed = True
        if removed and not self._tab_info:
            self.select_scope()
        elif not removed:
            self.select_scope()

    def _on_tab_close_requested(self, index: int) -> None:
        widget = self._tabs.widget(index)
        if widget is None:
            return
        if widget is self._scope_widget:
            self._tabs.setCurrentIndex(index)
            return
        self._tabs.removeTab(index)
        name = self._tab_info.pop(widget, None)
        if isinstance(name, str):
            current = self._analysis_count.get(name, 1) - 1
            if current > 0:
                self._analysis_count[name] = current
            else:
                self._analysis_count.pop(name, None)
        widget.deleteLater()
        if not self._tab_info:
            self.select_scope()
