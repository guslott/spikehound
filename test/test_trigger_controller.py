
import pytest
import numpy as np
from gui.trigger_controller import TriggerController

class TestTriggerController:
    @pytest.fixture
    def controller(self):
        ctrl = TriggerController()
        # Setup basic config
        ctrl.configure(
            mode="repeated",
            threshold=0.5,
            pre_seconds=0.002,     # 2ms pre
            window_sec=0.010,      # 10ms total
            channel_id=0
        )
        ctrl.update_sample_rate(10000.0) # 10kHz -> 1 sample = 0.1ms
        return ctrl

    def test_basic_threshold_crossing(self, controller):
        """Verify standard rising edge trigger still works."""
        # Create a signal: 0 -> 1 crossing
        # 20ms of data (200 samples) to ensure we have enough for alignment padding
        t = np.linspace(0, 0.02, 200)
        y = np.zeros_like(t)
        # Crossing at index 20 (2ms)
        y[20:] = 1.0 
        
        controller.push_samples(y, 10000.0, 0.01)
        
        # Check crossing detection
        idx = controller.detect_crossing(y)
        assert idx == 20
        
        # Trigger manually (since push_samples doesn't auto-trigger logic, usually loop does)
        # But we can simulate the loop logic
        assert controller.should_arm(1.0)
        controller.start_capture(0, idx)
        
        # Need to push enough history to finalize
        # We pushed 100 samples. 
        # Window is 10ms -> 100 samples.
        # Pre is 2ms -> 20 samples.
        # Start capture at 20.
        # Window needs data from 20-20=0 to 20-20+100=100.
        # This fits exactly in the buffer we pushed.
        
        success = controller.advance_capture()
        assert success
        
        display = controller.display_data
        assert display is not None
        assert display.shape[0] == 100
        # With threshold alignment, index 0 of display should be (trigger_idx - pre_samples)
        # Trigger at 20. Pre=20. So index 0 of display should be original index 0.
        # display[0] should be 0.0
        # display[20] should be 1.0 (the crossing)
        
        assert display[20] >= 0.5
        assert display[19] < 0.5

    def _make_capture(self, controller):
        """Helper: create a deterministic captured display."""
        y = np.zeros(200, dtype=np.float32)
        y[20:] = 1.0
        controller.push_samples(y, 10000.0, 0.01)
        trigger_idx = controller.detect_crossing(y)
        assert trigger_idx == 20
        controller.start_capture(0, trigger_idx)
        assert controller.advance_capture()
        assert controller.display_data is not None

    def test_configure_preserve_display_on_reset(self, controller):
        """UI config changes should preserve the visible capture for visual tuning."""
        self._make_capture(controller)
        before = np.array(controller.display_data, copy=True)

        controller.configure(
            mode="repeated",
            threshold=0.6,
            pre_seconds=0.003,
            window_sec=0.02,
            channel_id=0,
            reset_state=True,
            preserve_display_on_reset=True,
        )

        after = controller.display_data
        assert after is not None
        np.testing.assert_allclose(after, before)

    def test_reset_state_can_keep_display(self, controller):
        """Resetting trigger internals should optionally keep displayed waveform."""
        self._make_capture(controller)
        before = np.array(controller.display_data, copy=True)

        controller.reset_state(clear_display=False)

        after = controller.display_data
        assert after is not None
        np.testing.assert_allclose(after, before)

    def test_legacy_continuous_mode_is_rejected(self):
        """Only canonical trigger mode names should be accepted."""
        ctrl = TriggerController()
        with pytest.raises(ValueError, match="mode must be one of"):
            ctrl.configure(
                mode="continuous",
                threshold=0.5,
                pre_seconds=0.01,
                window_sec=0.2,
                channel_id=0,
            )
