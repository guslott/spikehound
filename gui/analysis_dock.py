from __future__ import annotations

from typing import Dict, Optional

from PySide6 import QtCore, QtWidgets

from .analysis_tab import AnalysisTab
from analysis.analysis_worker import AnalysisWorker


class AnalysisDock(QtWidgets.QDockWidget):
    """Dockable workspace containing the scope tab plus ad-hoc analysis tabs."""

    def __init__(self, title: str = "Workspace", parent: Optional[QtWidgets.QWidget] = None, *, controller=None) -> None:
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
        self._tab_info: Dict[QtWidgets.QWidget, Dict[str, object]] = {}
        self._analysis_count: Dict[str, int] = {}
        self._controller = controller
        self._last_sample_rate: float = 0.0

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

    def open_analysis(self, channel_name: str, sample_rate: float) -> AnalysisTab:
        worker: Optional[AnalysisWorker] = None
        if self._controller is not None:
            worker = AnalysisWorker(self._controller, channel_name, sample_rate)
            worker.start()
        count = self._analysis_count.get(channel_name, 0) + 1
        self._analysis_count[channel_name] = count
        title = f"Analysis - {channel_name}" if count == 1 else f"Analysis - {channel_name} #{count}"

        widget = AnalysisTab(channel_name, sample_rate, self._tabs, controller=self._controller)
        if worker is not None:
            widget.set_analysis_queue(worker.output_queue)
            widget.set_worker(worker)
        insert_index = 1 if self._scope_widget is not None else self._tabs.count()
        self._tabs.insertTab(insert_index, widget, title)
        self._tabs.setCurrentWidget(widget)
        self._tab_info[widget] = {"channel": channel_name, "worker": worker}
        if sample_rate > 0:
            self._last_sample_rate = float(sample_rate)
        return widget

    def update_sample_rate(self, sample_rate: float) -> None:
        if sample_rate <= 0:
            return
        if abs(sample_rate - self._last_sample_rate) < 1e-3:
            return
        self._last_sample_rate = float(sample_rate)
        for widget, info in list(self._tab_info.items()):
            if isinstance(widget, AnalysisTab):
                widget.set_channel_info(widget.channel_name, sample_rate)
            worker = info.get("worker") if isinstance(info, dict) else None
            if isinstance(worker, AnalysisWorker):
                worker.update_sample_rate(sample_rate)

    def close_tab(self, channel_name: str) -> None:
        removed = False
        for widget, info in list(self._tab_info.items()):
            mapped_name = info.get("channel") if isinstance(info, dict) else None
            if mapped_name != channel_name:
                continue
            index = self._tabs.indexOf(widget)
            if index >= 0:
                self._tabs.removeTab(index)
            widget.deleteLater()
            worker = info.get("worker") if isinstance(info, dict) else None
            if isinstance(worker, AnalysisWorker):
                worker.stop()
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
        info = self._tab_info.pop(widget, None)
        worker = None
        mapped_name = None
        if isinstance(info, dict):
            mapped_name = info.get("channel")
            worker = info.get("worker")
        if isinstance(worker, AnalysisWorker):
            worker.stop()
        if isinstance(mapped_name, str):
            current = self._analysis_count.get(mapped_name, 1) - 1
            if current > 0:
                self._analysis_count[mapped_name] = current
            else:
                self._analysis_count.pop(mapped_name, None)
        widget.deleteLater()
        if not self._tab_info:
            self.select_scope()

    def shutdown(self) -> None:
        for widget, info in list(self._tab_info.items()):
            if widget is self._scope_widget:
                continue
            worker = info.get("worker") if isinstance(info, dict) else None
            if isinstance(worker, AnalysisWorker):
                worker.stop()
            if self._tabs.indexOf(widget) >= 0:
                self._tabs.removeTab(self._tabs.indexOf(widget))
            widget.deleteLater()
        self._tab_info.clear()
        self._analysis_count.clear()
        self.select_scope()
