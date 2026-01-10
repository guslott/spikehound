# Dispatcher

`core.dispatcher.Dispatcher` is the central router for signal data. It accepts raw chunks from the DAQ, conditions them (filtering), and fans them out to visualization, audio, and logging consumers.

## Architecture

- **Input**: `raw_queue` receives `ChunkPointer`s from the DAQ.
- **Processing**:
  - Reads raw data from the source `SharedRingBuffer`.
  - Applies signal conditioning (filters).
  - Writes processed data to an internal `viz_buffer` (another `SharedRingBuffer`).
- **Output**: Fans out data to multiple queues:
  - `visualization`: For GUI plotting.
  - `audio`: For audio monitoring.
  - `logging`: For disk recording.
  - `analysis`: For registered analysis workers.
  - `events`: For detection events (threshold crossings, spikes, etc.).

## Queue Semantics

The Dispatcher handles queues differently based on their criticality. The policy for each queue is defined in `QUEUE_POLICIES`:

| Queue | Policy | Behavior |
|-------|--------|----------|
| `visualization` | `drop-newest` | Drops incoming item if full |
| `audio` | `drop-newest` | Drops incoming item if full |
| `logging` | `lossless` | Blocks up to 10s, then raises error |
| `analysis` | `drop-oldest` | Evicts oldest item to make room |
| `events` | `drop-newest` | Drops incoming item if full |

### Policy Details

1. **Visualization, Audio & Events (drop-newest)**
   - **Method**: Non-blocking (`put_nowait`).
   - **Behavior**: If the queue is full, the item is **dropped**. This prevents slow UI or audio consumers from blocking the critical DAQ path.
   - **Stats**: Drops are recorded in `DispatcherStats.dropped`.

2. **Logging (lossless)**
   - **Method**: Blocking with timeout (`put(timeout=10.0)`).
   - **Behavior**: If the queue is full (e.g., slow disk I/O), the Dispatcher **blocks** up to 10 seconds.
   - **Failure**: If it remains blocked, it raises a `RuntimeError`, effectively halting the pipeline to prevent silent data loss.

3. **Analysis (drop-oldest)**
   - **Method**: Non-blocking with eviction (`_enqueue_drop_oldest`).
   - **Behavior**: If the queue is full, the **oldest** item is evicted to make room for the new item. This ensures analysis workers always see the most recent data, even if they can't keep up.
   - **EOS Delivery**: EndOfStream is always delivered using drop-oldest to ensure cleanup.
   - **Stats**: Evictions are recorded in `DispatcherStats.evicted`.

## Stats

The `Dispatcher` maintains `DispatcherStats` to track performance:
- `received`: Total chunks received from DAQ.
- `processed`: Chunks successfully processed.
- `forwarded`: Chunks successfully enqueued to consumers (per queue).
- `dropped`: Chunks dropped due to full queues (drop-newest queues).
- `evicted`: Chunks removed from full queues to make room (drop-oldest queues).

## Thread Safety

- **Internal State**: Protected by `threading.Lock`s (`_stats_lock`, `_ring_lock`, `_analysis_lock`).
- **Queues**: Uses standard thread-safe `queue.Queue`.
