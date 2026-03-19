from __future__ import annotations

from types import SimpleNamespace

from gui.main_window import MainWindow


class _FakeComboBox:
    def __init__(self) -> None:
        self._items: list[tuple[str, object]] = []
        self._current_index = -1

    def blockSignals(self, blocked: bool) -> None:
        return

    def clear(self) -> None:
        self._items.clear()
        self._current_index = -1

    def addItem(self, label: str, data) -> None:
        self._items.append((label, data))
        if self._current_index < 0:
            self._current_index = 0

    def count(self) -> int:
        return len(self._items)

    def setCurrentIndex(self, index: int) -> None:
        self._current_index = int(index)

    def findData(self, value: float) -> int:
        for idx, (_, data) in enumerate(self._items):
            if data == value:
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

    def setItemData(self, index: int, value, role=None) -> None:
        return

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


def test_disconnected_device_selector_drops_stale_unsupported_rate() -> None:
    combo = _FakeComboBox()
    entry = {
        "key": "soundcard::default",
        "device_id": "default",
        "capabilities": {"sample_rates": [44100, 48000, 88200, 96000]},
    }
    state = SimpleNamespace(
        _device_connected=False,
        runtime=SimpleNamespace(
            daq_source=None,
            sample_rate=0.0,
            active_device_key=lambda: None,
        ),
        device_control=SimpleNamespace(sample_rate_combo=combo),
        channel_controls=SimpleNamespace(active_combo=SimpleNamespace(count=lambda: 0)),
        _logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        _is_file_source_entry=lambda key, device_entry: False,
    )
    state._current_sample_rate_value = lambda: 20000.0
    state._set_sample_rate_value = lambda value: MainWindow._set_sample_rate_value(state, value)

    MainWindow._populate_sample_rate_options(state, entry)

    assert [combo.itemData(i) for i in range(combo.count())] == [44100.0, 48000.0, 88200.0, 96000.0]
    assert combo.currentData() == 44100.0


def test_device_selection_repopulates_only_rates_for_selected_device() -> None:
    device_combo = _FakeComboBox()
    sample_rate_combo = _FakeComboBox()
    state = SimpleNamespace(
        _device_map={},
        _device_connected=False,
        runtime=SimpleNamespace(active_device_key=lambda: None, daq_source=None, sample_rate=0.0),
        device_control=SimpleNamespace(
            device_combo=device_combo,
            sample_rate_combo=sample_rate_combo,
            set_file_source_mode=lambda enabled: None,
        ),
        channel_controls=SimpleNamespace(active_combo=SimpleNamespace(count=lambda: 0)),
        _logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        _is_file_source_entry=lambda key, entry: False,
        _apply_device_state=lambda enabled: None,
        _update_channel_buttons=lambda: None,
    )
    state._current_sample_rate_value = lambda: float(sample_rate_combo.currentData() or 0.0)
    state._set_sample_rate_value = lambda value: MainWindow._set_sample_rate_value(state, value)
    state._populate_sample_rate_options = lambda entry: MainWindow._populate_sample_rate_options(state, entry)
    state._update_sample_rate_enabled = lambda: None
    state._on_device_selected = lambda: MainWindow._on_device_selected(state)

    entries = [
        {
            "key": "soundcard::default",
            "name": "Sound Card - System Default",
            "device_id": "default",
            "capabilities": {"sample_rates": [44100, 48000, 88200, 96000]},
        },
        {
            "key": "simulated::dev0",
            "name": "Simulated Physiology",
            "device_id": "dev0",
            "capabilities": {"sample_rates": [10000, 20000]},
        },
    ]

    MainWindow._on_devices_changed(state, entries)
    assert [sample_rate_combo.itemData(i) for i in range(sample_rate_combo.count())] == [44100.0, 48000.0, 88200.0, 96000.0]
    assert sample_rate_combo.currentData() == 44100.0

    device_combo.setCurrentIndex(1)
    MainWindow._on_device_selected(state)
    assert [sample_rate_combo.itemData(i) for i in range(sample_rate_combo.count())] == [10000.0, 20000.0]
    assert sample_rate_combo.currentData() == 10000.0


def test_device_list_load_populates_sound_card_rates_on_startup() -> None:
    device_combo = _FakeComboBox()
    sample_rate_combo = _FakeComboBox()
    state = SimpleNamespace(
        _device_map={},
        _device_connected=False,
        runtime=SimpleNamespace(active_device_key=lambda: None, daq_source=None, sample_rate=0.0),
        device_control=SimpleNamespace(
            device_combo=device_combo,
            sample_rate_combo=sample_rate_combo,
            set_file_source_mode=lambda enabled: None,
        ),
        channel_controls=SimpleNamespace(active_combo=SimpleNamespace(count=lambda: 0)),
        _logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        _is_file_source_entry=lambda key, entry: False,
        _apply_device_state=lambda enabled: None,
        _update_channel_buttons=lambda: None,
    )
    state._current_sample_rate_value = lambda: 20000.0
    state._set_sample_rate_value = lambda value: MainWindow._set_sample_rate_value(state, value)
    state._populate_sample_rate_options = lambda entry: MainWindow._populate_sample_rate_options(state, entry)
    state._update_sample_rate_enabled = lambda: None
    state._on_device_selected = lambda: MainWindow._on_device_selected(state)

    MainWindow._on_devices_changed(
        state,
        [
            {
                "key": "soundcard::default",
                "name": "Sound Card - System Default",
                "device_id": "default",
                "capabilities": {"sample_rates": [44100, 48000, 88200, 96000]},
            }
        ],
    )

    assert device_combo.currentData() == "soundcard::default"
    assert [sample_rate_combo.itemData(i) for i in range(sample_rate_combo.count())] == [44100.0, 48000.0, 88200.0, 96000.0]
    assert sample_rate_combo.currentData() == 44100.0


def test_file_source_selector_stays_empty_until_file_is_loaded() -> None:
    combo = _FakeComboBox()
    entry = {
        "key": "daq.file_source.FileSource::file",
        "device_id": "file",
        "capabilities": {"sample_rates": None},
    }
    state = SimpleNamespace(
        _device_connected=False,
        runtime=SimpleNamespace(
            daq_source=None,
            sample_rate=0.0,
            active_device_key=lambda: None,
        ),
        device_control=SimpleNamespace(sample_rate_combo=combo),
        channel_controls=SimpleNamespace(active_combo=SimpleNamespace(count=lambda: 0)),
        _logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        _is_file_source_entry=lambda key, device_entry: True,
    )
    state._current_sample_rate_value = lambda: 20000.0
    state._set_sample_rate_value = lambda value: MainWindow._set_sample_rate_value(state, value)

    MainWindow._populate_sample_rate_options(state, entry)

    assert combo.count() == 0
    assert combo.currentData() is None


def test_file_source_connect_forces_native_rate_resolution() -> None:
    calls: list[tuple[str, float, dict]] = []

    def _connect_device(device_key: str, sample_rate: float, **driver_kwargs) -> None:
        calls.append((device_key, sample_rate, dict(driver_kwargs)))

    state = SimpleNamespace(
        _device_connected=False,
        _device_map={
            "daq.file_source.FileSource::file": {
                "key": "daq.file_source.FileSource::file",
                "device_id": "file",
            }
        },
        runtime=SimpleNamespace(connect_device=_connect_device),
        device_control=SimpleNamespace(set_connected=lambda connected: None),
        _browse_file_source_path=lambda: "/tmp/example.wav",
        _is_file_source_entry=lambda key, entry: True,
    )

    MainWindow._on_device_connect_requested(state, "daq.file_source.FileSource::file", 20000.0)

    assert calls == [
        (
            "daq.file_source.FileSource::file",
            0.0,
            {"device_id_override": "/tmp/example.wav"},
        )
    ]


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
