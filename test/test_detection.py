import queue
import numpy as np
import pytest
from core import Dispatcher, FilterSettings, Chunk, EndOfStream
from core.detection import AmpThresholdDetector, DETECTOR_REGISTRY
from shared.models import Event

def _make_chunk(samples: np.ndarray, *, start_time: float, dt: float, seq: int) -> Chunk:
    return Chunk(
        samples=samples,
        start_time=start_time,
        dt=dt,
        seq=seq,
        channel_names=tuple(f"ch{i}" for i in range(samples.shape[0])),
        units="V",
        meta={"start_sample": seq * samples.shape[1]},
    )

def test_amp_threshold_detector_logic():
    fs = 1000.0
    dt = 1.0 / fs
    duration = 1.0
    t = np.arange(int(duration * fs)) * dt
    
    # Create signal: noise + spike
    # Noise: sigma=0.1
    # Spike: -1.0 at t=0.5
    noise = np.random.normal(0, 0.1, size=len(t))
    signal = noise.copy()
    spike_idx = int(0.5 * fs)
    signal[spike_idx] = -1.0 # Negative spike
    
    detector = AmpThresholdDetector()
    detector.configure(factor=5.0, sign=-1, window_ms=5.0)
    detector.reset(fs, 1)
    
    chunk = _make_chunk(signal.reshape(1, -1), start_time=0.0, dt=dt, seq=0)
    events = detector.process_chunk(chunk)
    
    assert len(events) >= 1
    # Check if we detected the spike near 0.5s
    found = False
    for e in events:
        if abs(e.t - 0.5) < 0.01:
            found = True
            assert e.chan == 0
            # Check window size: 5ms at 1000Hz = 5 samples
            assert len(e.window) == 5
            break
    assert found

    detector.reset(fs, 1)
    detector.configure(factor=5.0, sign=-1, window_ms=5.0)
    events = detector.process_chunk(chunk)
    
    found = False
    for e in events:
        if abs(e.t - 0.5) < 0.02:
            found = True
            # Check window size: 5ms at 1000Hz = 5 samples
            assert len(e.window) == 5
            # Check pre_samples
            assert "pre_samples" in e.params
            # Window 5ms, 1000Hz -> 5 samples. pre = 5//3 = 1.
            assert e.params["pre_samples"] == 1 
            break
    assert found

def test_amp_threshold_detector_refractory():
    fs = 1000.0
    dt = 1.0 / fs
    # Create two spikes close to each other
    # Window 5ms. Refractory 1ms (default).
    # But we enforce refractory >= window. So refractory should be 5ms.
    signal = np.zeros(1000)
    signal[100] = 10.0 # Spike 1
    signal[103] = 10.0 # Spike 2 (3ms later, within 5ms window)
    
    detector = AmpThresholdDetector()
    detector.configure(factor=5.0, sign=1, window_ms=5.0, refractory_ms=1.0)
    detector.reset(fs, 1)
    
    chunk = _make_chunk(signal.reshape(1, -1), start_time=0.0, dt=dt, seq=0)
    events = detector.process_chunk(chunk)
    
    # Should only detect the first spike because the second is within the window/refractory period
    assert len(events) == 1
    assert abs(events[0].t - 0.1) < dt/2
    
    # Now test with spike outside refractory
    signal[106] = 10.0 # Spike 3 (6ms later, outside 5ms window)
    detector.reset(fs, 1)
    chunk = _make_chunk(signal.reshape(1, -1), start_time=0.0, dt=dt, seq=0)
    events = detector.process_chunk(chunk)
    # Should detect spike 1 and spike 3 (spike 2 is still suppressed by spike 1)
    assert len(events) == 2
    assert abs(events[0].t - 0.1) < dt/2
    assert abs(events[1].t - 0.106) < dt/2

def test_amp_threshold_detector_bidirectional():
    fs = 1000.0
    dt = 1.0 / fs
    signal = np.zeros(1000)
    # Positive spike
    signal[200] = 2.0
    # Negative spike
    signal[600] = -2.0
    
    detector = AmpThresholdDetector()
    # Threshold = 1.0 (since noise is 0, we need to force threshold or add noise)
    # With 0 noise, MAD is 0. Threshold is 0.
    # Let's add some noise
    noise = np.random.normal(0, 0.1, size=1000)
    signal += noise
    
    detector.configure(factor=5.0, sign=0) # Bidirectional
    detector.reset(fs, 1)
    
    chunk = _make_chunk(signal.reshape(1, -1), start_time=0.0, dt=dt, seq=0)
    events = detector.process_chunk(chunk)
    
    # Should find both spikes
    found_pos = False
    found_neg = False
    for e in events:
        if abs(e.t - 0.2) < 0.02:
            found_pos = True
        if abs(e.t - 0.6) < 0.02:
            found_neg = True
            
    assert found_pos
    assert found_neg

def test_dispatcher_integration_with_detection():
    raw_queue = queue.Queue()
    viz_queue = queue.Queue()
    audio_queue = queue.Queue()
    log_queue = queue.Queue()
    event_queue = queue.Queue()
    
    dispatcher = Dispatcher(
        raw_queue,
        viz_queue,
        audio_queue,
        log_queue,
        event_queue,
        filter_settings=FilterSettings()
    )
    
    # Configure detector
    dispatcher.configure_detectors(["amp_threshold"])
    
    dispatcher.start()
    
    # Feed data
    fs = 1000.0
    dt = 1.0 / fs
    signal = np.random.normal(0, 0.1, size=1000)
    signal[500] = -1.0 # Spike
    
    chunk = _make_chunk(signal.reshape(1, -1).astype(np.float32), start_time=0.0, dt=dt, seq=0)
    
    # We need to set source buffer for dispatcher to work, or at least set sample rate
    # Dispatcher needs a buffer to read from if we pass ChunkPointer, 
    # OR we can hack it? No, Dispatcher reads from buffer.
    # So we need a SharedRingBuffer.
    
    from shared.ring_buffer import SharedRingBuffer
    buffer = SharedRingBuffer((1, 2000), dtype=np.float32)
    buffer.write(signal.reshape(1, -1).astype(np.float32))
    
    dispatcher.set_source_buffer(buffer, sample_rate=fs)
    
    # Create pointer
    from shared.models import ChunkPointer
    ptr = ChunkPointer(start_index=0, length=1000, render_time=0.0)
    
    raw_queue.put(ptr)
    raw_queue.put(EndOfStream)
    
    dispatcher.join(timeout=2.0)
    
    # Check events
    events = []
    while not event_queue.empty():
        item = event_queue.get()
        if item is EndOfStream:
            continue
        events.append(item)
        
    assert len(events) >= 1
    found = False
    for e in events:
        if abs(e.t - 0.5) < 0.01:
            found = True
            break
    assert found

def test_amp_threshold_detector_cross_chunk():
    fs = 1000.0
    dt = 1.0 / fs
    # Create signal with spike at the boundary
    # Chunk 1: 0.0 to 0.5s (500 samples)
    # Chunk 2: 0.5 to 1.0s (500 samples)
    # Spike at 0.499s (index 499)
    # Window 5ms (5 samples). Pre=1, Post=3.
    # Spike at 499. Window: 498, 499, 500, 501, 502.
    # 498, 499 are in Chunk 1.
    # 500, 501, 502 are in Chunk 2.
    
    signal = np.zeros(1000)
    spike_idx = 499
    # Make a distinct shape to verify window integrity
    signal[spike_idx-1] = 0.5
    signal[spike_idx] = 1.0
    signal[spike_idx+1] = -0.5
    signal[spike_idx+2] = -0.2
    signal[spike_idx+3] = 0.0
    
    detector = AmpThresholdDetector()
    detector.configure(factor=5.0, sign=1, window_ms=5.0) # Threshold approx 0 if noise is 0
    # Add tiny noise to avoid div/0 in MAD or force threshold
    signal += np.random.normal(0, 0.001, size=1000)
    detector.reset(fs, 1)
    
    # Split into two chunks
    chunk1_data = signal[:500].reshape(1, -1)
    chunk2_data = signal[500:].reshape(1, -1)
    
    chunk1 = _make_chunk(chunk1_data, start_time=0.0, dt=dt, seq=0)
    chunk2 = _make_chunk(chunk2_data, start_time=0.5, dt=dt, seq=1)
    
    # Process chunk 1
    events1 = detector.process_chunk(chunk1)
    # Should NOT detect the spike yet because it's at the end
    # Valid end index for chunk 1 (len 500) with post=3 is 497.
    # Spike is at 499. So it should be deferred.
    assert len(events1) == 0
    
    # Process chunk 2
    events2 = detector.process_chunk(chunk2)
    # Should detect the spike now
    assert len(events2) == 1
    e = events2[0]
    
    # Check timestamp
    expected_t = 0.499
    assert abs(e.t - expected_t) < dt/2
    
    # Check window content
    # Should match signal[498:503]
    expected_window = signal[498:503]
    # Allow small diff due to float precision or noise
    assert np.allclose(e.window, expected_window, atol=1e-5)
    
    # Check pre_samples
    assert e.params["pre_samples"] == 1
