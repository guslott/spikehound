from __future__ import annotations

from types import SimpleNamespace

from gui.main_window import MainWindow


class _FakeComboBox:
    def __init__(self) -> None:
        self._items: list[tuple[str, float]] = []
        self._current_index = -1

    def blockSignals(self, blocked: bool) -> None:
        return

    def clear(self) -> None:
        self._items.clear()
        self._current_index = -1

    def addItem(self, label: str, data: float) -> None:
        self._items.append((label, float(data)))
        if self._current_index < 0:
            self._current_index = 0

    def count(self) -> int:
        return len(self._items)

    def setCurrentIndex(self, index: int) -> None:
        self._current_index = int(index)

    def findData(self, value: float) -> int:
        wanted = float(value)
        for idx, (_, data) in enumerate(self._items):
            if data == wanted:
                return idx
        return -1

    def currentData(self):
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def itemData(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index][1]
        return None

    def setEnabled(self, enabled: bool) -> None:
        return


def test_connected_device_sample_rate_selector_refreshes_from_live_caps() -> None:
    combo = _FakeComboBox()
    entry = {
        "key": "backyard_brains::/dev/tty.mfi",
        "device_id": "/dev/tty.mfi",
        "capabilities": {"sample_rates": [10000]},
    }
    live_caps = SimpleNamespace(sample_rates=[10000, 20000])
    driver = SimpleNamespace(
        config=SimpleNamespace(sample_rate=20000),
        get_capabilities=lambda device_id: live_caps,
    )
    state = SimpleNamespace(
        _device_connected=True,
        runtime=SimpleNamespace(
            daq_source=driver,
            sample_rate=20000.0,
            active_device_key=lambda: entry["key"],
        ),
        device_control=SimpleNamespace(sample_rate_combo=combo),
        channel_controls=SimpleNamespace(active_combo=SimpleNamespace(count=lambda: 0)),
        _logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        _is_file_source_entry=lambda key, device_entry: False,
    )
    state._current_sample_rate_value = lambda: float(combo.currentData() or 0.0)
    state._set_sample_rate_value = lambda value: MainWindow._set_sample_rate_value(state, value)

    MainWindow._populate_sample_rate_options(state, entry)

    assert [combo.itemData(i) for i in range(combo.count())] == [10000.0, 20000.0]
    assert combo.currentData() == 20000.0
    assert entry["capabilities"] is live_caps


def test_byb_debug_tab_visibility_tracks_live_byb_connection() -> None:
    calls: list[tuple[str, str]] = []
    debug_tab = SimpleNamespace(refresh=lambda: calls.append(("refresh", "")))
    dock = SimpleNamespace(
        set_aux_widget=lambda key, widget, title, insert_index=2: calls.append(("set", key)),
        remove_aux_widget=lambda key: calls.append(("remove", key)),
    )
    state = SimpleNamespace(
        _device_connected=True,
        _byb_debug_tab=debug_tab,
        _analysis_dock=dock,
        runtime=SimpleNamespace(
            daq_source=SimpleNamespace(stats=lambda: {"profile": "neuron_pro_mfi"})
        ),
        _logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
    )

    MainWindow._update_byb_debug_tab(state)
    assert calls == [("set", "byb_debug"), ("refresh", "")]

    calls.clear()
    state._device_connected = False
    MainWindow._update_byb_debug_tab(state)
    assert calls == [("remove", "byb_debug")]
