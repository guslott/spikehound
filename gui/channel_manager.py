"""ChannelManager - Manages channel configurations, active channel tracking, and panel lifecycle.

Provides a focused component for channel state management, including channel
configs, panels, color cycling, and active channel selection.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from .types import ChannelConfig
from .channel_controls_widget import ChannelControlsWidget, ChannelDetailPanel

if TYPE_CHECKING:
    from .device_control_widget import DeviceControlWidget

logger = logging.getLogger(__name__)


class ChannelManager(QtCore.QObject):
    """Manages channel configurations, active channel tracking, and panel lifecycle.
    
    Responsibilities:
    - Channel configuration storage and lifecycle
    - Channel panel creation and removal
    - Active channel tracking and selection
    - Color cycling for new channels
    - Filter settings synchronization triggers
    """
    
    # Signals
    channelConfigChanged = QtCore.Signal(int, ChannelConfig)  # channel_id, config
    activeChannelChanged = QtCore.Signal(object)  # Optional[int]
    channelsUpdated = QtCore.Signal(list, list)  # List[int] channel_ids, List[str] channel_names
    listenChannelRequested = QtCore.Signal(int, bool)  # channel_id, enabled
    analysisRequested = QtCore.Signal(int)  # channel_id
    filterSettingsChanged = QtCore.Signal()  # Signal to trigger filter sync
    
    def __init__(
        self,
        channel_controls: ChannelControlsWidget,
        device_control: "DeviceControlWidget",
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(__name__)
        
        # Widget references
        self._channel_controls = channel_controls
        self._device_control = device_control
        
        # Channel state
        self._channel_configs: Dict[int, ChannelConfig] = {}
        self._channel_panels: Dict[int, ChannelDetailPanel] = {}
        self._channel_ids_current: List[int] = []
        self._channel_names: List[str] = []
        self._active_channel_infos: List[object] = []
        self._active_channel_id: Optional[int] = None
        
        # Color cycling
        self._channel_color_cycle: List[QtGui.QColor] = [
            QtGui.QColor(0, 0, 139),
            QtGui.QColor(178, 34, 34),
            QtGui.QColor(0, 105, 148),
            QtGui.QColor(34, 139, 34),
            QtGui.QColor(128, 0, 128),
            QtGui.QColor(255, 140, 0),
        ]
        self._next_color_index = 0
        
        # Connect widget signals
        self._channel_controls.activeChannelSelected.connect(self._on_active_channel_selected)
    
    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------
    
    @property
    def channel_configs(self) -> Dict[int, ChannelConfig]:
        """Current channel configurations."""
        return self._channel_configs
    
    @property
    def channel_panels(self) -> Dict[int, ChannelDetailPanel]:
        """Current channel panels."""
        return self._channel_panels
    
    @property
    def channel_ids_current(self) -> List[int]:
        """List of currently active channel IDs."""
        return self._channel_ids_current
    
    @property
    def channel_names(self) -> List[str]:
        """List of currently active channel names."""
        return self._channel_names
    
    @property
    def active_channel_id(self) -> Optional[int]:
        """Currently focused/active channel ID."""
        return self._active_channel_id
    
    @property
    def active_channel_infos(self) -> List[object]:
        """List of active channel info objects."""
        return self._active_channel_infos
    
    # -------------------------------------------------------------------------
    # Color Management
    # -------------------------------------------------------------------------
    
    def reset_color_cycle(self) -> None:
        """Reset channel color selection to the initial palette order."""
        self._next_color_index = 0
    
    def next_channel_color(self) -> QtGui.QColor:
        """Get the next color in the channel color cycle."""
        color = self._channel_color_cycle[self._next_color_index % len(self._channel_color_cycle)]
        self._next_color_index += 1
        return QtGui.QColor(color)
    
    # -------------------------------------------------------------------------
    # Channel Configuration
    # -------------------------------------------------------------------------
    
    def initial_screen_offset(self) -> float:
        """Pick a starting offset; default all new channels to center."""
        return 0.5
    
    def ensure_channel_config(self, channel_id: int, channel_name: str) -> ChannelConfig:
        """Ensure a channel config exists, creating one if necessary."""
        config = self._channel_configs.get(channel_id)
        if config is None:
            color = self.next_channel_color()
            config = ChannelConfig(
                color=color,
                channel_name=channel_name,
                screen_offset=self.initial_screen_offset(),
            )
            self._channel_configs[channel_id] = config
        else:
            config.channel_name = channel_name
        return config
    
    def get_config(self, channel_id: int) -> Optional[ChannelConfig]:
        """Get the configuration for a specific channel."""
        return self._channel_configs.get(channel_id)
    
    def update_config(self, channel_id: int, config: ChannelConfig) -> None:
        """Update the configuration for a specific channel."""
        self._channel_configs[channel_id] = config
        panel = self._channel_panels.get(channel_id)
        if panel is not None:
            panel.set_config(config)
        self.channelConfigChanged.emit(channel_id, config)
    
    # -------------------------------------------------------------------------
    # Panel Management
    # -------------------------------------------------------------------------
    
    def sync_channel_panels(
        self,
        channel_ids: Sequence[int],
        channel_names: Sequence[str],
        *,
        analysis_dock: Optional[object] = None,
    ) -> None:
        """Synchronize channel panels with the current active channel list."""
        desired = {cid: name for cid, name in zip(channel_ids, channel_names)}
        
        # Remove panels for channels no longer active
        for cid, panel in list(self._channel_panels.items()):
            if cid not in desired:
                config = self._channel_configs.get(cid)
                if config is not None and analysis_dock is not None:
                    channel_name = config.channel_name or f"Channel {cid}"
                    try:
                        analysis_dock.close_tab(channel_name)
                    except Exception:
                        pass
                self._channel_controls.remove_panel(cid)
                del self._channel_panels[cid]
                self._channel_configs.pop(cid, None)
        
        if not channel_ids:
            self._show_channel_panel(None)
            return
        
        # Add/update panels for active channels
        for cid, name in desired.items():
            config = self.ensure_channel_config(cid, name)
            panel = self._channel_panels.get(cid)
            if panel is None:
                panel = ChannelDetailPanel(cid, name, self._channel_controls.stack)
                panel.configChanged.connect(lambda cfg, cid=cid: self._on_channel_config_changed(cid, cfg))
                panel.analysisRequested.connect(lambda cid=cid: self.analysisRequested.emit(cid))
                self._channel_controls.add_panel(cid, panel)
                self._channel_panels[cid] = panel
            panel.set_config(config)
        
        # Show the appropriate panel
        idx = self._channel_controls.active_combo.currentIndex()
        if idx >= 0:
            info = self._channel_controls.active_combo.itemData(idx)
            cid = info.id if info is not None else None
            self._show_channel_panel(cid)
        else:
            first_id = channel_ids[0] if channel_ids else None
            self._show_channel_panel(first_id)
            if first_id is not None:
                self.set_active_channel_focus(first_id)
    
    def clear_channel_panels(self) -> None:
        """Clear all channel panels."""
        self._channel_controls.clear_panels()
        self._channel_panels.clear()
        # Don't clear _channel_configs - configs should persist so channels maintain
        # their color/settings when re-added. Configs are cleared on device disconnect.
        self._show_channel_panel(None)
    
    def clear_all_state(self) -> None:
        """Clear all channel state including configs (called on device disconnect)."""
        self.clear_channel_panels()
        self._channel_configs.clear()
        self._channel_ids_current.clear()
        self._channel_names.clear()
        self._active_channel_infos.clear()
        self._active_channel_id = None
        self.reset_color_cycle()
    
    def _show_channel_panel(self, channel_id: Optional[int]) -> None:
        """Show the panel for the specified channel."""
        self._channel_controls.show_panel(channel_id)
    
    # -------------------------------------------------------------------------
    # Active Channel Management
    # -------------------------------------------------------------------------
    
    def ensure_active_channel_focus(self) -> None:
        """Ensure active channel is valid."""
        if self._channel_ids_current:
            if self._active_channel_id not in self._channel_ids_current:
                self._active_channel_id = self._channel_ids_current[0]
        else:
            self._active_channel_id = None
        self.activeChannelChanged.emit(self._active_channel_id)
    
    def set_active_channel_focus(self, channel_id: Optional[int]) -> None:
        """Set the active channel focus."""
        if channel_id is not None and channel_id not in self._channel_ids_current:
            return
        if self._active_channel_id == channel_id:
            return
        self._active_channel_id = channel_id
        self.activeChannelChanged.emit(channel_id)
    
    def select_active_channel_by_id(self, channel_id: int) -> None:
        """Select a channel by ID in the active combo."""
        for i in range(self._channel_controls.active_combo.count()):
            info = self._channel_controls.active_combo.itemData(i)
            if info is not None and info.id == channel_id:
                self._channel_controls.active_combo.setCurrentIndex(i)
                self.set_active_channel_focus(channel_id)
                return
    
    # -------------------------------------------------------------------------
    # Channel Add/Remove
    # -------------------------------------------------------------------------
    
    def on_add_channel(self) -> Optional[int]:
        """Handle add channel action. Returns the added channel ID or None."""
        idx = self._device_control.available_combo.currentIndex()
        if idx < 0:
            return None
        
        info = self._device_control.available_combo.itemData(idx)
        text = self._device_control.available_combo.currentText()
        
        self._channel_controls.active_combo.addItem(text, info)
        self._device_control.available_combo.removeItem(idx)
        
        if self._device_control.available_combo.count():
            self._device_control.available_combo.setCurrentIndex(
                min(idx, self._device_control.available_combo.count() - 1)
            )
        
        # Activate and focus the newly added channel without extra signal chatter
        self._channel_controls.active_combo.blockSignals(True)
        self._channel_controls.active_combo.setCurrentIndex(
            self._channel_controls.active_combo.count() - 1
        )
        self._channel_controls.active_combo.blockSignals(False)
        
        channel_id = info.id if info is not None else None
        return channel_id
    
    def publish_active_channels(self) -> tuple[List[int], List[str]]:
        """
        Collect and publish the current active channels.
        
        Returns:
            Tuple of (channel_ids, channel_names)
        """
        infos = []
        for index in range(self._channel_controls.active_combo.count()):
            info = self._channel_controls.active_combo.itemData(index)
            if info is not None:
                infos.append(info)
        
        self._active_channel_infos = infos
        
        ids = [info.id for info in infos if info is not None]
        ids = [cid for cid in ids if cid is not None]
        names = [info.name if info is not None else str(info) for info in infos]
        
        # Track if channels changed
        channels_changed = list(ids) != self._channel_ids_current
        
        # Update current state
        self._channel_ids_current = list(ids)
        self._channel_names = list(names)
        
        # Emit channels updated signal
        self.channelsUpdated.emit(ids, names)
        
        return ids, names
    
    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------
    
    def _on_active_channel_selected(self, index: int) -> None:
        """Handle active channel selection change."""
        if index < 0:
            self._show_channel_panel(None)
            return
        
        info = self._channel_controls.active_combo.itemData(index)
        if info is None:
            self._show_channel_panel(None)
            return
        
        channel_id = info.id if info is not None else None
        name = info.name if info is not None else str(info)
        
        if channel_id is None:
            self._show_channel_panel(None)
            return
        
        config = self.ensure_channel_config(channel_id, name)
        panel = self._channel_panels.get(channel_id)
        if panel is not None:
            panel.set_config(config)
        
        self._show_channel_panel(channel_id)
        self.set_active_channel_focus(channel_id)
    
    def _on_channel_config_changed(self, channel_id: int, config: ChannelConfig) -> None:
        """Handle channel configuration changes from panel."""
        previous = self._channel_configs.get(channel_id)
        
        # Update panel display
        panel = self._channel_panels.get(channel_id)
        if panel is not None:
            panel.set_config(config)
        
        # Store updated config
        self._channel_configs[channel_id] = config
        
        # Check if filters changed
        filters_changed = self._filters_changed(previous, config)
        if filters_changed:
            self.filterSettingsChanged.emit()
        
        # Handle listen toggle
        if previous is None or previous.listen_enabled != config.listen_enabled:
            self.listenChannelRequested.emit(channel_id, config.listen_enabled)
        
        # Emit config changed signal
        self.channelConfigChanged.emit(channel_id, config)
    
    def _filters_changed(self, previous: Optional[ChannelConfig], current: ChannelConfig) -> bool:
        """Check if filter settings changed between configs."""
        if previous is None:
            return True
        
        def _diff(a: float, b: float) -> bool:
            return abs(a - b) > 1e-6
        
        if previous.notch_enabled != current.notch_enabled:
            return True
        if current.notch_enabled and _diff(previous.notch_freq_hz, current.notch_freq_hz):
            return True
        if previous.highpass_enabled != current.highpass_enabled:
            return True
        if current.highpass_enabled and _diff(previous.highpass_hz, current.highpass_hz):
            return True
        if previous.lowpass_enabled != current.lowpass_enabled:
            return True
        if current.lowpass_enabled and _diff(previous.lowpass_hz, current.lowpass_hz):
            return True
        return False
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def get_nearest_channel_at_y(self, y: float) -> Optional[int]:
        """Return the channel whose configured offset is closest to the given y view coordinate."""
        if not self._channel_ids_current:
            return None
        best_id: Optional[int] = None
        best_dist = float("inf")
        for cid in self._channel_ids_current:
            cfg = self._channel_configs.get(cid)
            if cfg is None:
                continue
            dist = abs(cfg.screen_offset - y)
            if dist < best_dist:
                best_dist = dist
                best_id = cid
        return best_id
    
    def update_listen_state(self, channel_id: int, enabled: bool) -> None:
        """Update listen state for a channel, ensuring only one channel can listen at a time."""
        if enabled:
            # Disable listen on all other channels
            for cid, cfg in self._channel_configs.items():
                if cid != channel_id and cfg.listen_enabled:
                    cfg.listen_enabled = False
                    panel = self._channel_panels.get(cid)
                    if panel is not None:
                        panel.set_config(cfg)
            
            # Enable listen on the requested channel
            cfg = self._channel_configs.get(channel_id)
            if cfg is not None:
                cfg.listen_enabled = True
                panel = self._channel_panels.get(channel_id)
                if panel is not None:
                    panel.set_config(cfg)
        else:
            # Disable listen on the specified channel
            cfg = self._channel_configs.get(channel_id)
            if cfg is not None and cfg.listen_enabled:
                cfg.listen_enabled = False
                panel = self._channel_panels.get(channel_id)
                if panel is not None:
                    panel.set_config(cfg)
