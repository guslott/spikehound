
import pytest
import numpy as np
from gui.trigger_controller import TriggerController

class TestTriggerController:
    @pytest.fixture
    def controller(self):
        ctrl = TriggerController()
        # Setup basic config
        ctrl.configure(
            mode="continuous",
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
        
        success = controller.finalize_capture()
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

    def test_peak_alignment_shift(self, controller):
        """
        Verify that peak alignment shifts the window to center the peak.
        
        Signal:
          - Rises past 0.5 at t=0 samples (relative to start)
          - Hits peak of 1.0 at t=10 samples
          - We want alignment to the PEAK.
          - So the peak (index 10) should appear at 'pre_samples' index in the output.
        """
        # Enable peak alignment (property will be added)
        if hasattr(controller, "alignment_mode"):
            controller.alignment_mode = "peak"
        else:
            pytest.skip("Peak alignment not implemented yet")

        # 10kHz sample rate. 
        # Pre-trigger = 2ms = 20 samples.
        # Total window = 100 samples.
        
        # Create synthetic spike
        # Rise at 100, Peak at 110.
        # Total buffer 300 samples
        y = np.zeros(300, dtype=np.float32)
        
        # Linear rise 0 -> 1 over 10 samples starting at 100
        y[100:111] = np.linspace(0.5, 1.0, 11) 
        y[100] = 0.51 # Ensure instant crossing if needed, or ramp
        # Let's make a gaussian-ish bump
        # Peak at 110, width 5
        x = np.arange(300)
        y = np.exp(-0.5 * ((x - 110)/5.0)**2)
        
        # Threshold is 0.5.
        # Gaussian e^(-0.5 * z^2) = 0.5 => -0.5*z^2 = ln(0.5) = -0.693 => z^2 = 1.386 => z = 1.17
        # Width 5 -> dx = 1.17 * 5 = 5.85 samples.
        # So crossing is roughly at 110 - 6 = 104.
        
        # Let's find exact crossing
        crossing_idx = np.argmax(y > 0.5)
        assert 100 < crossing_idx < 110
        
        # Push all data
        controller.push_samples(y, 10000.0, 0.01)
        
        # Detect
        # We need to manually call detection as it relies on state
        trigger_idx = controller.detect_crossing(y)
        assert trigger_idx == crossing_idx
        
        controller.start_capture(0, trigger_idx)
        success = controller.finalize_capture()
        assert success
        
        display = controller.display_data
        
        # Expected behavior:
        # The PEAK (original index 110) should be at display[pre_samples] (index 20).
        # So display[20] should be the max.
        
        peak_idx_in_display = np.argmax(display)
        assert peak_idx_in_display == 20
        assert np.isclose(display[20], 1.0)
