from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg

# --- DAQ base (updated interface) ---
from daq.base_source import BaseSource, Chunk, DeviceInfo, ChannelInfo  # type: ignore

# --- Local plot widget (selection + drag) ---
from gui.scope_plot import ScopePlot


# ------------------------ utils / discovery ------------------------

@dataclass
class SourceEntry:
    cls: type
    module_name: str

@dataclass
class DeviceEntry:
    id: object
    name: str


def discover_sources() -> List[SourceEntry]:
    """
    Find DAQ source classes in daq/*.py that subclass BaseSource.
    Only modules ending with '_source.py' are considered (excluding base_source).
    """
    import daq as daq_pkg  # the package directory you already have
    results: List[SourceEntry] = []
    for m in pkgutil.iter_modules(daq_pkg.__path__, prefix="daq."):
        name = m.name
        if not name.endswith("_source"):
            continue
        if name.endswith("base_source"):
            continue
        mod = importlib.import_module(name)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            try:
                if issubclass(obj, BaseSource) and obj is not BaseSource:
                    results.append(SourceEntry(cls=obj, module_name=name))
            except Exception:
                pass
    return results


def discover_devices_for(source_cls: type) -> List[DeviceEntry]:
    """
    Ask an instance of the class for devices using BaseSource API.
    """
    try:
        inst: BaseSource = source_cls()  # type: ignore
        devs = inst.list_available_devices()
        if devs:
            return [DeviceEntry(id=d.id, name=d.name) for d in devs]
    except Exception:
        pass
    return [DeviceEntry(id=None, name=f"{source_cls.__name__} (default)")]


# ------------------------ connect dialog ------------------------

class ConnectDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Device")
        self.setModal(True)

        self.sources = discover_sources()

        layout = QtWidgets.QFormLayout(self)

        self.source_combo = QtWidgets.QComboBox()
        for s in self.sources:
            self.source_combo.addItem(s.cls.__name__, userData=s)

        self.device_combo = QtWidgets.QComboBox()
        self.sample_rate_spin = QtWidgets.QSpinBox()
        self.sample_rate_spin.setRange(1000, 384000)
        self.sample_rate_spin.setValue(44100)
        self.sample_rate_spin.setSingleStep(1000)

        self.chunk_spin = QtWidgets.QSpinBox()
        self.chunk_spin.setRange(32, 8192)
        self.chunk_spin.setValue(512)
        self.chunk_spin.setSingleStep(32)

        layout.addRow("Source:", self.source_combo)
        layout.addRow("Device:", self.device_combo)
        layout.addRow("Sample Rate (Hz):", self.sample_rate_spin)
        layout.addRow("Chunk Size:", self.chunk_spin)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addRow(btn_box)

        # Signals
        self.source_combo.currentIndexChanged.connect(self._refresh_devices)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        self._refresh_devices()

    def _refresh_devices(self):
        self.device_combo.clear()
        idx = self.source_combo.currentIndex()
        entry: SourceEntry = self.source_combo.itemData(idx)
        devices = discover_devices_for(entry.cls)
        for d in devices:
            self.device_combo.addItem(d.name, userData=d)

    # -- results --
    def selected(self) -> Tuple[type, DeviceEntry, int, int]:
        s_entry: SourceEntry = self.source_combo.currentData()
        d_entry: DeviceEntry = self.device_combo.currentData()
        return s_entry.cls, d_entry, int(self.sample_rate_spin.value()), int(self.chunk_spin.value())


# ------------------------ channel widgets (below the plot) ------------------------

class ColorButton(QtWidgets.QPushButton):
    colorChanged = QtCore.Signal(QtGui.QColor)

    def __init__(self, color: QtGui.QColor, parent=None):
        super().__init__(parent)
        self._color = QtGui.QColor(color)
        self.setText("Color")
        self.setFixedWidth(70)
        self._update_style()
        self.clicked.connect(self._choose)

    def color(self) -> QtGui.QColor:
        return QtGui.QColor(self._color)

    def setColor(self, c: QtGui.QColor):
        self._color = QtGui.QColor(c)
        self._update_style()
        self.colorChanged.emit(self._color)

    def _choose(self):
        c = QtWidgets.QColorDialog.getColor(self._color, self, "Choose Channel Color")
        if c.isValid():
            self.setColor(c)

    def _update_style(self):
        self.setStyleSheet(f"background-color: {self._color.name()};")


class AvailableChannelsPanel(QtWidgets.QGroupBox):
    addRequested = QtCore.Signal(str, str, QtGui.QColor)  # (channel_id, name, color)

    def __init__(self, parent=None):
        super().__init__("Available Channels", parent)
        lay = QtWidgets.QGridLayout(self)
        self.device_label = QtWidgets.QLabel("Device: —")
        self.chan_combo = QtWidgets.QComboBox()
        self.name_edit = QtWidgets.QLineEdit()
        self.color_btn = ColorButton(QtGui.QColor("#1d59d1"))  # deep-ish blue like g-PRIME
        self.add_btn = QtWidgets.QPushButton("Add Channel")

        lay.addWidget(self.device_label, 0, 0, 1, 3)
        lay.addWidget(QtWidgets.QLabel("Chan:"), 1, 0)
        lay.addWidget(self.chan_combo, 1, 1)
        lay.addWidget(self.color_btn, 1, 2)
        lay.addWidget(QtWidgets.QLabel("Name:"), 2, 0)
        lay.addWidget(self.name_edit, 2, 1, 1, 2)
        lay.addWidget(self.add_btn, 3, 0, 1, 3)

        self.chan_combo.currentIndexChanged.connect(self._seed_name)
        self.add_btn.clicked.connect(self._emit_add)

    def setDeviceName(self, name: str):
        self.device_label.setText(f"Device: {name}")

    def setChannels(self, ids_and_names: List[Tuple[str, str]]):
        self.chan_combo.clear()
        for cid, nm in ids_and_names:
            self.chan_combo.addItem(f"{nm} [{cid}]", userData=(cid, nm))
        self._seed_name()

    def _seed_name(self):
        data = self.chan_combo.currentData()
        if not data:
            self.name_edit.setText("")
            return
        cid, nm = data
        self.name_edit.setText(nm)

    def _emit_add(self):
        data = self.chan_combo.currentData()
        if not data:
            return
        cid, default_nm = data
        nm = self.name_edit.text().strip() or default_nm
        self.addRequested.emit(cid, nm, self.color_btn.color())


class ActiveChannelsPanel(QtWidgets.QGroupBox):
    removeRequested = QtCore.Signal(str)  # channel_id
    selectedChanged = QtCore.Signal(str)  # channel_id

    def __init__(self, parent=None):
        super().__init__("Active Channels", parent)
        lay = QtWidgets.QVBoxLayout(self)
        self.listw = QtWidgets.QListWidget()
        self.remove_btn = QtWidgets.QPushButton("Remove Channel")
        self.remove_btn.setEnabled(False)
        lay.addWidget(self.listw)
        lay.addWidget(self.remove_btn)

        self.listw.currentItemChanged.connect(self._on_sel)
        self.remove_btn.clicked.connect(self._on_remove)

    def addItem(self, channel_id: str, name: str, color: QtGui.QColor):
        item = QtWidgets.QListWidgetItem(name)
        item.setData(QtCore.Qt.UserRole, channel_id)
        # color swatch
        icon = QtGui.QPixmap(16, 16)
        icon.fill(color)
        item.setIcon(QtGui.QIcon(icon))
        self.listw.addItem(item)
        self.listw.setCurrentItem(item)

    def removeItem(self, channel_id: str):
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            if it.data(QtCore.Qt.UserRole) == channel_id:
                self.listw.takeItem(i)
                break
        self.remove_btn.setEnabled(self.listw.currentItem() is not None)

    def _on_sel(self, cur, prev):
        cid = cur.data(QtCore.Qt.UserRole) if cur else ""
        self.selectedChanged.emit(cid)
        self.remove_btn.setEnabled(bool(cur))

    def _on_remove(self):
        cur = self.listw.currentItem()
        if not cur:
            return
        cid = cur.data(QtCore.Qt.UserRole)
        self.removeRequested.emit(cid)


# ------------------------ main window ------------------------

class ScopeMainWindow(QtWidgets.QMainWindow):
    PLOT_SECONDS = 5.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("g‑PRIME Scope (PySide6 – minimal slice)")
        self._source: Optional[BaseSource] = None
        self._active_ids: List[str] = []     # string ids matching ChannelInfo.id
        self._id_to_curve_index: Dict[str, int] = {}  # mapping to plot rows
        self._device_id: Optional[str] = None
        self._sample_rate: int = 44100
        self._chunk_size: int = 512

        self._build_ui()
        self._apply_theme()

        # drain timer
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(15)  # ~60–70 Hz update cadence
        self._timer.timeout.connect(self._drain)
        # build initial device menu
        # (menu is also available for quick selection without the dialog)
        # populated after UI is built

    # --- UI setup ---
    def _build_ui(self):
        # Menu
        file_menu = self.menuBar().addMenu("&File")
        act_quit = QtGui.QAction("E&xit", self, triggered=self.close)
        file_menu.addAction(act_quit)

        self.dev_menu = self.menuBar().addMenu("&Device")
        act_connect = QtGui.QAction("&Connect…", self, triggered=self._connect_dialog)
        act_disconnect = QtGui.QAction("&Disconnect", self, triggered=self._disconnect)
        self.dev_menu.addAction(act_connect)
        self.dev_menu.addAction(act_disconnect)
        self.dev_menu.addSeparator()
        self.device_select_menu = self.dev_menu.addMenu("Select &Device")
        act_refresh = QtGui.QAction("&Refresh Devices", self, triggered=self._refresh_device_menu)
        self.dev_menu.addAction(act_refresh)

        # Central layout: 2x2 grid where only upper-left + bottom-left are used now
        central = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(central)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.setCentralWidget(central)

        # Plot area (upper-left)
        self.scope = ScopePlot()
        grid.addWidget(self.scope, 0, 0, 1, 1)

        # Future panels placeholder (upper-right + bottom-right)
        self.right_placeholder_top = QtWidgets.QFrame()
        self.right_placeholder_top.setFrameShape(QtWidgets.QFrame.StyledPanel)
        grid.addWidget(self.right_placeholder_top, 0, 1)
        self.right_placeholder_bottom = QtWidgets.QFrame()
        self.right_placeholder_bottom.setFrameShape(QtWidgets.QFrame.StyledPanel)
        grid.addWidget(self.right_placeholder_bottom, 1, 1)

        # Channel panels under the plot (bottom-left)
        chan_container = QtWidgets.QWidget()
        chan_layout = QtWidgets.QHBoxLayout(chan_container)
        chan_layout.setContentsMargins(0, 0, 0, 0)

        self.avail_panel = AvailableChannelsPanel()
        self.active_panel = ActiveChannelsPanel()
        chan_layout.addWidget(self.avail_panel, 1)
        chan_layout.addWidget(self.active_panel, 1)

        grid.addWidget(chan_container, 1, 0)

        # Behavior wiring
        self.avail_panel.addRequested.connect(self._add_channel)
        self.active_panel.removeRequested.connect(self._remove_channel)
        self.active_panel.selectedChanged.connect(self._select_channel)

        # Column stretches: left wider than right (plot-centric)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setRowStretch(0, 4)
        grid.setRowStretch(1, 1)
        # Fill the device selection menu with discovered drivers/devices
        self._refresh_device_menu()

    def _apply_theme(self):
        # g‑PRIME‑ish palette: soft green window, blue traces, light grid
        base = QtGui.QColor("#E6F0E0")  # soft green
        p = self.palette()
        p.setColor(QtGui.QPalette.Window, base)
        p.setColor(QtGui.QPalette.Base, base)
        self.setPalette(p)

        self.scope.apply_scope_theme()

    # --- Device connection ---
    def _connect_dialog(self):
        dlg = ConnectDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        cls, device_entry, sr, chunk = dlg.selected()

        # tear down prior
        self._disconnect()

        # instantiate driver
        self._source = cls()  # type: ignore
        self._device_id = str(device_entry.id) if device_entry.id is not None else "sim0"
        self._sample_rate = sr
        self._chunk_size = chunk

        # Open and query channels
        self._source.open(self._device_id)
        chans: List[ChannelInfo] = self._source.list_available_channels(self._device_id)
        device_name = device_entry.name or cls.__name__
        self.avail_panel.setDeviceName(device_name)
        # pairs of (id_string, display_name)
        pairs = [(str(c.id), c.name) for c in chans]
        self.avail_panel.setChannels(pairs)

        # Configure with no channels selected yet -> configure will default to all,
        # so we delay configuration until first add. Set timebase for plot now.
        self.scope.set_time_base(sample_rate=sr, seconds=self.PLOT_SECONDS)

        self._active_ids = []
        self._id_to_curve_index.clear()
        self.scope.clear_all()
        self._timer.start()

    def _disconnect(self):
        self._timer.stop()
        if self._source:
            try:
                if getattr(self._source, "running", False):
                    self._source.stop()
                self._source.close()
            except Exception:
                pass
        self._source = None
        self._active_ids.clear()
        self._id_to_curve_index.clear()
        self.scope.clear_all()
        self.avail_panel.setDeviceName("—")
        self.avail_panel.setChannels([])

    # --- config/reconfig helper ---
    def _apply_channel_selection(self):
        if not self._source or self._device_id is None:
            return
        # Reconfigure driver with selected channel ids (as ints)
        chan_ids = [int(cid) for cid in self._active_ids if str(cid).isdigit()]
        if not chan_ids:
            # if none selected, stop streaming
            try:
                if getattr(self._source, "running", False):
                    self._source.stop()
            except Exception:
                pass
            return
        try:
            if getattr(self._source, "running", False):
                self._source.stop()
            self._source.configure(sample_rate=self._sample_rate, channels=chan_ids, chunk_size=self._chunk_size, num_units=6)
            self._source.start()
        except Exception as e:
            print(f"Configure/start failed: {e}")

    # --- Channel operations ---
    @QtCore.Slot(str, str, QtGui.QColor)
    def _add_channel(self, channel_id: str, name: str, color: QtGui.QColor):
        if not self._source:
            return
        if channel_id in self._active_ids:
            return  # already added
        self._active_ids.append(channel_id)
        curve_index = self.scope.add_curve(channel_id, name, color)
        self._id_to_curve_index[channel_id] = curve_index
        self.active_panel.addItem(channel_id, name, color)
        self._apply_channel_selection()

    @QtCore.Slot(str)
    def _remove_channel(self, channel_id: str):
        if not self._source:
            return

        if channel_id in self._active_ids:
            self._active_ids.remove(channel_id)
        idx = self._id_to_curve_index.pop(channel_id, None)
        if idx is not None:
            self.scope.remove_curve(channel_id)
        self.active_panel.removeItem(channel_id)
        self._apply_channel_selection()

    @QtCore.Slot(str)
    def _select_channel(self, channel_id: str):
        self.scope.select_curve(channel_id)

    # --- Data flow ---
    def _drain(self):
        if not self._source:
            return
        got = False
        while True:
            try:
                ch: Chunk = self._source.data_queue.get_nowait()
            except Exception:
                break
            data = np.asarray(ch.data)  # (chunk, n_active)
            # Dispatch each column to its curve, in add order
            ncols = data.shape[1] if data.ndim == 2 else 1
            if ncols == 1 and len(self._active_ids) == 1:
                cid = self._active_ids[0]
                self.scope.append_samples(cid, data.reshape(-1))
            else:
                for i, cid in enumerate(self._active_ids[:ncols]):
                    self.scope.append_samples(cid, data[:, i])
            got = True
        if got:
            self.scope.refresh()

    # --- Qt events ---
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._disconnect()
        finally:
            return super().closeEvent(event)

    # --- Device menu population and selection ---
    def _refresh_device_menu(self):
        self.device_select_menu.clear()
        for s in discover_sources():
            submenu = QtWidgets.QMenu(s.cls.__name__, self)
            try:
                inst: BaseSource = s.cls()  # type: ignore
                devs = inst.list_available_devices()
            except Exception:
                devs = []
            if not devs:
                act = QtGui.QAction("(No devices)", self)
                act.setEnabled(False)
                submenu.addAction(act)
            else:
                for d in devs:
                    act = QtGui.QAction(d.name, self)
                    act.triggered.connect(lambda checked=False, cls=s.cls, dev=d: self._open_from_menu(cls, dev))
                    submenu.addAction(act)
            self.device_select_menu.addMenu(submenu)

    def _open_from_menu(self, cls: type, dev_entry) -> None:
        self._disconnect()
        try:
            self._source = cls()  # type: ignore
            self._device_id = str(getattr(dev_entry, 'id', '')) or 'sim0'
            self._source.open(self._device_id)
            chans: List[ChannelInfo] = self._source.list_available_channels(self._device_id)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Error", f"Failed to open device: {e}")
            self._disconnect()
            return
        self.avail_panel.setDeviceName(getattr(dev_entry, 'name', cls.__name__))
        pairs = [(str(c.id), c.name) for c in chans]
        self.avail_panel.setChannels(pairs)
        self.scope.set_time_base(sample_rate=self._sample_rate, seconds=self.PLOT_SECONDS)
        self._active_ids.clear()
        self._id_to_curve_index.clear()
        self.scope.clear_all()
        self._timer.start()
