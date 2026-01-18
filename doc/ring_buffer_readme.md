# SharedRingBuffer

`shared.ring_buffer.SharedRingBuffer` is a thread-safe circular buffer designed for high-performance audio/signal data exchange between threads.

## Design

- **Backing Store**: A pre-allocated `numpy.ndarray`.
- **Thread Safety**: Uses `threading.RLock` to protect read/write operations.
- **Data Layout**: `(channels, capacity)` or `(..., capacity)`. The last dimension is always the time axis (capacity).

## Contract

### Writers
- **Single Writer**: The class is designed for a single writer. While the lock protects internal state, multiple writers would race for the write position without external coordination.
- **Overwriting**: Writes that exceed remaining capacity will wrap around and overwrite the oldest data. The buffer does *not* block on write; it always accepts data.

### Readers
- **Multiple Readers**: Safe for multiple concurrent readers.
- **Views vs. Copies**: 
  - If the requested read range is contiguous in memory, `read()` returns a **view** (zero-copy).
  - If the range wraps around the buffer end, `read()` returns a **copy** (concatenated).

## Usage

```python
# Initialize
rb = SharedRingBuffer(shape=(channels, capacity), dtype=np.float32)

# Write (Writer Thread)
# Returns the start index of the written chunk in the ring coordinate space
start_idx = rb.write(new_data)

# Read (Reader Thread)
# Read 'length' samples starting at 'start_idx'
data = rb.read(start_idx, length)
```
