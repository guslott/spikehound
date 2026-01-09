from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import MainWindow

logger = logging.getLogger(__name__)


class SignalBridge:
    """
    Centralizes the wiring of signal-slot connections between UI modules.
    
    This class acts as the 'nervous system' of the application, isolating 
    the complex logic of how different widgets talk to each other 
    from the layout and lifecycle code in MainWindow.
    """

    def __init__(self, main_window: MainWindow) -> None:
        self.mw = main_window

    def wire_ui_internal(self) -> None:
        """Connect signals between UI widgets themselves."""
        mw = self.mw
        
        # Plot Manager
        mw._plot_manager.sampleRateChanged.connect(mw._maybe_update_analysis_sample_rate)

        # Recording Widget
        mw.record_group.recordingStarted.connect(mw._on_recording_started)
        mw.record_group.recordingStopped.connect(mw._on_recording_stopped)

        # Trigger Widget
        mw._trigger_controller.configChanged.connect(mw._on_trigger_config_changed)
        
        # Channel Manager
        mw._channel_manager.channelConfigChanged.connect(mw._on_channel_manager_config_changed)
        mw._channel_manager.activeChannelChanged.connect(mw._on_channel_manager_active_changed)
        mw._channel_manager.channelsUpdated.connect(mw._on_channel_manager_channels_updated)
        mw._channel_manager.listenChannelRequested.connect(mw._audio_listen_manager.handle_listen_change)
        mw._channel_manager.analysisRequested.connect(mw._open_analysis_for_channel)
        mw._channel_manager.filterSettingsChanged.connect(mw._sync_filter_settings)

        # Device Control Widget
        mw.device_control.deviceSelected.connect(mw._on_device_selected)
        mw.device_control.deviceConnectRequested.connect(mw._on_device_connect_requested)
        mw.device_control.deviceDisconnectRequested.connect(mw._on_device_disconnect_requested)
        mw.device_control.channelAddRequested.connect(mw._on_channel_add_requested)
        mw.device_control.playPauseToggled.connect(mw._on_playback_toggled)
        mw.device_control.seekRequested.connect(mw._on_seek_requested)

        # Settings Tab
        if mw._settings_tab:
            mw._settings_tab.saveConfigRequested.connect(mw._on_save_scope_config)
            mw._settings_tab.loadConfigRequested.connect(mw._on_load_scope_config)

    def wire_runtime_signals(self) -> None:
        """Connect signals from the runtime/device manager."""
        mw = self.mw
        dm = mw.runtime.device_manager
        
        dm.devicesChanged.connect(mw._on_devices_changed)
        dm.deviceConnected.connect(mw._on_device_connected)
        dm.deviceDisconnected.connect(mw._on_device_disconnected)
        dm.availableChannelsChanged.connect(mw._on_available_channels)

    def wire_controller(self, controller) -> None:
        """Connect signals requiring a PipelineController (called during attach)."""
        mw = self.mw
        if controller is None:
            return

        mw.startRecording.connect(controller.start_recording)
        mw.stopRecording.connect(controller.stop_recording)
        mw.triggerConfigChanged.connect(controller.update_trigger_config)
        
        # Wire controller to recording widget for duration queries
        mw.record_group.set_controller(controller)
