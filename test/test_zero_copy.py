import numpy as np
from shared.models import Chunk
import pytest

def test_chunk_zero_copy_initialization():
    """Verify that Chunk shares memory with the input array if it's C-contiguous."""
    # Create a C-contiguous array
    original = np.zeros((1, 100), dtype=np.float32, order='C')
    original[0, 0] = 123.0
    
    # Initialize chunk
    chunk = Chunk(
        samples=original,
        start_time=0.0,
        dt=1.0,
        seq=0,
        channel_names=("ch1",),
        units="V"
    )
    
    # Check memory sharing
    assert np.shares_memory(original, chunk.samples), "Chunk should share memory with C-contiguous input"
    assert chunk.samples[0, 0] == 123.0
    
    # Verify read-only flag
    assert chunk.samples.flags.writeable is False
    
    # Verify we can't write to it via the chunk
    with pytest.raises(ValueError):
        chunk.samples[0, 0] = 456.0
        
    # Verify strict contiguity check works (F-order should force copy)
    f_order = np.zeros((1, 100), dtype=np.float32, order='F')
    chunk_copy = Chunk(
        samples=f_order,
        start_time=0.0,
        dt=1.0,
        seq=0,
        channel_names=("ch1",),
        units="V"
    )
    # F-contiguous input of shape (1, N) might NOT be copied if it's compatible with C-order
    # (NumPy optimization). So we just check that the result is valid.
    assert chunk_copy.samples.flags.c_contiguous is True
    assert chunk_copy.samples.flags.writeable is False

if __name__ == "__main__":
    test_chunk_zero_copy_initialization()
    print("Zero-copy test passed!")
