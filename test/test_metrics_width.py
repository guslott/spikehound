
import numpy as np
import pytest
from analysis.metrics import event_width

def test_event_width_clean_pulse():
    sr = 1000.0
    pre_samples = 100
    
    # Create a clean square pulse
    # Noise floor 0.1
    # Pulse height 2.0
    # Pulse width 10ms (10 samples)
    
    data = np.random.normal(0, 0.1, 300)
    start_idx = pre_samples + 20
    width_samples = 10
    end_idx = start_idx + width_samples
    data[start_idx:end_idx] = 2.0
    
    # Peak is in the middle of pulse
    peak_idx = start_idx + 5
    
    # Manually pass threshold. 6 sigma of 0.1 noise ~ 0.6. 
    # Let's say threshold is 1.0 for simplicity.
    threshold = 1.0
    
    # Note: our event_width function expects the user to pass a threshold value if sigma is not used 
    # OR we can assume sigma=6.0. 
    # Let's verify the signature from implementation plan:
    # event_width(samples, sr, threshold=None, sigma=None, min_run=3, off_count=3, off_window=4)
    
    width_ms = event_width(data, sr, threshold=threshold, min_run=3, off_count=3, off_window=4)
    
    # Expected: The clean pulse crosses 1.0 exactly at start_idx and end_idx.
    # Start scan back from peak (start_idx+5) -> should find start_idx.
    # End scan fwd from peak -> should find end_idx.
    # Width = 10 samples = 10ms.
    
    assert abs(width_ms - 10.0) < 0.1

def test_event_width_noisy_dip():
    # Test "3 of 4" logic
    sr = 1000.0
    threshold = 1.0
    
    # Create a pulse that dips below threshold for 2 samples then comes back
    data = np.zeros(200)
    # Start: 50
    # End: 80 (30 samples)
    data[50:80] = 2.0
    
    # Dip at 60,61 (2 samples to 0.5)
    data[60:62] = 0.5
    
    # Logic: off_count=3, off_window=4.
    # A 2-sample dip should NOT trigger end.
    
    width_ms = event_width(data, sr, threshold=threshold, min_run=3, off_count=3, off_window=4)
    
    assert abs(width_ms - 30.0) < 0.1

def test_event_width_noisy_termination():
    # Test termination when 3 of 4 are below
    sr = 1000.0
    threshold = 1.0
    data = np.zeros(200)
    
    # Pulse starts at 50, ends at 80
    data[50:80] = 2.0
    
    # At 80, 81, 82 it is 0.0 (below th) -> count=3 in window 4 -> SHOULD END
    data[80:85] = 0.0
    
    width_ms = event_width(data, sr, threshold=threshold, min_run=3, off_count=3, off_window=4)
    assert abs(width_ms - 30.0) < 0.1

def test_event_width_not_meeting_min_run():
    # If the peak is an isolated spike < min_run (3), it might have width 0 or start==end
    # Actually if peak is > th but neighbors are not, run < 3.
    # Start scan: requires run of 3 > th. If not found, it might default to peak or something.
    sr = 1000.0
    threshold = 1.0
    data = np.zeros(100)
    data[50] = 2.0 # single spike

    width_ms = event_width(data, sr, threshold=threshold, min_run=3, off_count=3, off_window=4)

    # Should probably be small or zero? The definition says "Start is first sample of a run of min_run".
    # If no run of 3 exists attached to peak, maybe it returns 0?
    assert width_ms == 0.0


@pytest.mark.parametrize(
    "block_len, expected_ms",
    [
        (2, 0.0),   # below min_run -> rejected by the final width gate
        (3, 3.0),   # exactly min_run -> accepted (width_samples == min_run is NOT < min_run)
        (5, 5.0),   # above min_run -> accepted
    ],
)
def test_event_width_min_run_gate_boundary(block_len, expected_ms):
    """Characterize the min_run boundary (finding 5.1).

    The backward start-scan used to contain an ``if/else`` whose two branches were
    identical, so ``min_run`` had no effect there; the *only* place min_run gates
    the result is the final ``width_samples < min_run`` check. This test pins that
    behavior at the boundary (block_len == min_run is the inclusive cutoff) so the
    dead-branch removal is provably behavior-preserving and a future editor doesn't
    reintroduce a redundant backward gate.
    """
    sr = 1000.0  # 1 sample == 1 ms, so width_samples maps 1:1 to ms
    threshold = 1.0
    data = np.zeros(100)
    data[50 : 50 + block_len] = 2.0  # contiguous above-threshold block

    width_ms = event_width(
        data, sr, threshold=threshold, min_run=3, off_count=3, off_window=4
    )

    assert abs(width_ms - expected_ms) < 1e-6

