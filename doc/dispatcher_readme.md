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

## Queue Semantics

The Dispatcher handles queues differently based on their criticality:

1. **Visualization & Audio (Lossy)**
   - **Method**: Non-blocking (`put_nowait`).
   - **Behavior**: If the queue is full, the item is **dropped**. This prevents slow UI or audio consumers from blocking the critical DAQ path.
   - **Stats**: Drops are recorded in `DispatcherStats.dropped`.

2. **Logging (Lossless)**
   - **Method**: Blocking with timeout (`put(timeout=10.0)`).
   - **Behavior**: If the queue is full (e.g., slow disk I/O), the Dispatcher **blocks** up to 10 seconds.
   - **Failure**: If it remains blocked, it raises a `RuntimeError`, effectively halting the pipeline to prevent silent data loss.

## Stats

The `Dispatcher` maintains `DispatcherStats` to track performance:
- `received`: Total chunks received from DAQ.
- `processed`: Chunks successfully processed.
- `forwarded`: Chunks successfully enqueued to consumers (per queue).
- `dropped`: Chunks dropped due to full queues (lossy queues only).
- `evicted`: Chunks removed from full queues to make room (analysis queues).

## Thread Safety

- **Internal State**: Protected by `threading.Lock`s (`_stats_lock`, `_ring_lock`, `_analysis_lock`).
- **Queues**: Uses standard thread-safe `queue.Queue`.
