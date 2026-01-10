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
  - `logging`: For disk recording (only when recording is enabled).
  - `analysis`: For registered analysis workers.
  - `events`: For detection events (threshold crossings, spikes, etc.).

## Queue Semantics

The Dispatcher handles queues differently based on their criticality. The policy for each queue is defined in `QUEUE_POLICIES`:

| Queue | Policy | Behavior |
|-------|--------|----------|
| `visualization` | `drop-oldest` | Evicts oldest item to make room |
| `audio` | `drop-oldest` | Evicts oldest item to make room |
| `logging` | `lossless` | Blocks up to 10s, then raises error |
| `analysis` | `drop-oldest` | Evicts oldest item to make room |
| `events` | `drop-oldest` | Evicts oldest item to make room |

### Policy Details

1. **Visualization, Audio & Events (drop-oldest)**
   - **Method**: Non-blocking with eviction.
   - **Behavior**: If the queue is full, the **oldest** item is evicted to make room. This ensures real-time consumers always see the most recent data, minimizing lag.
   - **Stats**: Evictions are recorded in `DispatcherStats.evicted`.

2. **Logging (lossless)**
   - **Method**: Blocking with timeout (`put(timeout=10.0)`).
   - **Behavior**: If the queue is full (e.g., slow disk I/O), the Dispatcher **blocks** up to 10 seconds.
   - **Failure**: If it remains blocked, it raises a `RuntimeError`, effectively halting the pipeline to prevent silent data loss.

4. **Analysis (drop-oldest)**
   - **Method**: Non-blocking with eviction.
   - **Behavior**: If the queue is full, the **oldest** item is evicted to make room for the new item. This ensures analysis workers always see the most recent data, even if they can't keep up.
   - **Stats**: Evictions are recorded in `DispatcherStats.evicted`.

### EOS Delivery Guarantee

EndOfStream (EOS) signals are **always** delivered to all queues, regardless of queue fullness:

- **drop-oldest queues** (viz, audio, events, analysis): EOS is delivered using the native drop-oldest policy (evicts oldest if needed).
- **lossless queues** (logging): EOS uses the blocking lossless policy to ensure no data loss before shutdown.
- **drop-newest queues** (if any exist): `enqueue_with_policy()` temporarily upgrades them to `drop-oldest` for EOS messages to guarantee delivery.

This guarantee is enforced in `enqueue_with_policy()` in `shared/models.py`.

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

## Fail-Fast and Gap Policies

The Dispatcher supports configurable policies for handling internal invariant violations and data gaps, useful for distinguishing between development/debugging ("fail fast") and production resilience ("reset and continue").

### strict_invariants
- **Default**: `False`
- **Behavior**:
  - `False`: Invariants (e.g., missing source buffer, invalid sample rate) log warnings and skip processing.
  - `True`: Violations raise `RuntimeError`, immediately halting the pipeline.

### gap_policy
Controls behavior when a discontinuity in sample indices is detected (e.g., dropped chunks from DAQ).

| Policy | Behavior | Use Case |
|--------|----------|----------|
| `crash` | Raises `RuntimeError` immediately. | strict dev/science mode (ensure data integrity) |
| `reset` | Logs warning, resets visualization buffer counters, continues. | production/long-running (recover from glitches) |
| `ignore` | Logs warning, continues appending (risk of artifacts). | debugging/loose constraints |
