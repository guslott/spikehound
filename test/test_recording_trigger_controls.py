from __future__ import annotations

from types import SimpleNamespace

from gui.main_window import MainWindow


class _TrackedControl:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def setEnabled(self, enabled: bool) -> None:
        self.calls.append(bool(enabled))


def test_recording_state_does_not_toggle_trigger_controls() -> None:
    trigger_control = SimpleNamespace(
        window_combo=_TrackedControl(),
        trigger_mode_single=_TrackedControl(),
        trigger_mode_repeated=_TrackedControl(),
        threshold_spin=_TrackedControl(),
        trigger_channel_combo=_TrackedControl(),
        trigger_single_button=_TrackedControl(),
    )
    record_group = SimpleNamespace(set_enabled_for_recording=lambda enabled: None)
    device_control = SimpleNamespace(
        device_combo=_TrackedControl(),
        sample_rate_combo=_TrackedControl(),
        available_combo=_TrackedControl(),
        add_channel_btn=_TrackedControl(),
    )
    channel_controls = SimpleNamespace(setEnabled=lambda enabled: None)
    state = SimpleNamespace(
        trigger_control=trigger_control,
        record_group=record_group,
        device_control=device_control,
        channel_controls=channel_controls,
        _device_connected=True,
        _update_channel_buttons=lambda: None,
    )

    MainWindow._set_panels_enabled(state, False)

    assert trigger_control.window_combo.calls == []
    assert trigger_control.trigger_mode_single.calls == []
    assert trigger_control.trigger_mode_repeated.calls == []
    assert trigger_control.threshold_spin.calls == []
    assert trigger_control.trigger_channel_combo.calls == []
    assert trigger_control.trigger_single_button.calls == []
