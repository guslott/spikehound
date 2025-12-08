"""AudioListenManager - Manages audio monitoring state and routing.

Provides a focused component for audio listen functionality, including
channel selection, output device routing, and UI synchronization.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from core import PipelineController
    from .types import ChannelConfig
    from .channel_controls_widget import ChannelDetailPanel

logger = logging.getLogger(__name__)


class AudioListenManager(QtCore.QObject):
    """Manages audio monitoring state and routing between GUI and core AudioManager.
    
    Responsibilities:
    - Listen channel selection (single channel at a time)
    - Audio output device routing
    - UI synchronization for listen toggles
    """
    
    # Signals
    listenChannelChanged = QtCore.Signal(object)  # Optional[int]
    listenDeviceChanged = QtCore.Signal(object)   # Optional[str]
    
    def __init__(
        self,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(__name__)
        
        # State
        self._listen_channel_id: Optional[int] = None
        self._listen_device_key: Optional[str] = None
        
        # External references (set via setters)
        self._controller: Optional["PipelineController"] = None
        self._get_channel_configs: Optional[Callable[[], Dict[int, "ChannelConfig"]]] = None
        self._get_channel_panels: Optional[Callable[[], Dict[int, "ChannelDetailPanel"]]] = None
        self._get_sample_rate: Optional[Callable[[], float]] = None
        self._get_active_channel_ids: Optional[Callable[[], list]] = None
        self._show_message: Optional[Callable[[str, str], None]] = None
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    def set_controller(self, controller: Optional["PipelineController"]) -> None:
        """Set the pipeline controller for audio operations."""
        self._controller = controller
    
    def set_callbacks(
        self,
        *,
        get_channel_configs: Optional[Callable[[], Dict[int, "ChannelConfig"]]] = None,
        get_channel_panels: Optional[Callable[[], Dict[int, "ChannelDetailPanel"]]] = None,
        get_sample_rate: Optional[Callable[[], float]] = None,
        get_active_channel_ids: Optional[Callable[[], list]] = None,
        show_message: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Set callback functions for accessing application state."""
        if get_channel_configs is not None:
            self._get_channel_configs = get_channel_configs
        if get_channel_panels is not None:
            self._get_channel_panels = get_channel_panels
        if get_sample_rate is not None:
            self._get_sample_rate = get_sample_rate
        if get_active_channel_ids is not None:
            self._get_active_channel_ids = get_active_channel_ids
        if show_message is not None:
            self._show_message = show_message
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def listen_channel_id(self) -> Optional[int]:
        """Currently monitored channel ID."""
        return self._listen_channel_id
    
    @property
    def listen_device_key(self) -> Optional[str]:
        """Currently selected audio output device key."""
        return self._listen_device_key
    
    # -------------------------------------------------------------------------
    # Listen Channel Management
    # -------------------------------------------------------------------------
    
    def handle_listen_change(self, channel_id: int, enabled: bool) -> None:
        """Apply listen toggle semantics (single selection) and spin up/tear down audio plumbing."""
        if enabled:
            self.set_listen_channel(channel_id)
        else:
            if self._listen_channel_id == channel_id:
                self.clear_listen_channel(channel_id)
    
    def set_listen_channel(self, channel_id: int) -> None:
        """Activate audio monitoring for the requested channel, disabling other listen toggles."""
        channel_configs = self._get_channel_configs() if self._get_channel_configs else {}
        channel_panels = self._get_channel_panels() if self._get_channel_panels else {}
        sample_rate = self._get_sample_rate() if self._get_sample_rate else 0.0
        
        cfg = channel_configs.get(channel_id)
        if cfg is None:
            return
        
        # Check if streaming is active
        if sample_rate <= 0:
            if self._show_message:
                self._show_message(
                    "Audio Output",
                    "Audio output becomes available after streaming starts.",
                )
            cfg.listen_enabled = False
            panel = channel_panels.get(channel_id)
            if panel is not None:
                panel.set_config(cfg)
            return
        
        # Disable listen on all other channels
        for cid, other_cfg in channel_configs.items():
            if cid == channel_id:
                continue
            if other_cfg.listen_enabled:
                other_cfg.listen_enabled = False
                panel = channel_panels.get(cid)
                if panel is not None:
                    panel.set_config(other_cfg)
        
        # Enable listen on the requested channel
        cfg.listen_enabled = True
        panel = channel_panels.get(channel_id)
        if panel is not None:
            panel.set_config(cfg)
        
        # Track for UI state
        self._listen_channel_id = channel_id
        
        # Use controller API to start audio monitoring
        if self._controller:
            self._controller.set_audio_monitoring(channel_id)
            # Update active channels so AudioManager knows about them
            if hasattr(self._controller, '_audio_manager') and self._controller._audio_manager:
                active_ids = self._get_active_channel_ids() if self._get_active_channel_ids else []
                self._controller._audio_manager.update_active_channels(active_ids)
        
        self.listenChannelChanged.emit(channel_id)
    
    def clear_listen_channel(self, channel_id: Optional[int] = None) -> None:
        """Disable audio monitoring, optionally for a specific channel."""
        target = channel_id if channel_id is not None else self._listen_channel_id
        if target is None:
            return
        
        channel_configs = self._get_channel_configs() if self._get_channel_configs else {}
        channel_panels = self._get_channel_panels() if self._get_channel_panels else {}
        
        # Update UI state
        cfg = channel_configs.get(target)
        if cfg is not None and cfg.listen_enabled:
            cfg.listen_enabled = False
            panel = channel_panels.get(target)
            if panel is not None:
                panel.set_config(cfg)
        
        # Clear UI state tracking
        if self._listen_channel_id == target:
            self._listen_channel_id = None
        
        # Use controller API to stop audio monitoring
        if self._controller:
            self._controller.set_audio_monitoring(None)
        
        self.listenChannelChanged.emit(None)
    
    def clear_if_channel_removed(self, active_channel_ids: list) -> None:
        """Clear listen channel if it's no longer in the active channel list."""
        if self._listen_channel_id is not None and self._listen_channel_id not in active_channel_ids:
            self.clear_listen_channel()
    
    # -------------------------------------------------------------------------
    # Output Device Management
    # -------------------------------------------------------------------------
    
    def set_listen_device(self, device_key: Optional[str]) -> None:
        """Set the audio output device for monitoring."""
        normalized = None if device_key in (None, "") else str(device_key)
        self._listen_device_key = normalized
        self.listenDeviceChanged.emit(normalized)
