# LoadGenServerClient High QPS Hang Analysis

## Problem Statement

When running the Server scenario with high target QPS (e.g., 200+ QPS), the `LoadGenServerClient` can hang or become unresponsive. This document analyzes the root cause and proposes a solution.

## Root Cause Analysis

### Current Implementation

The `LoadGenServerClient` uses the following architecture:

```python
def _process_queries_worker(self):
    """Worker thread to process queued queries."""
    while True:
        qitem = self.query_queue.get()  # Blocking get from queue
        
        # PROBLEM: Spawns a NEW thread for EACH query
        threading.Thread(
            target=self._async_process_query,
            args=(input_ids_tensor, qitem.id),
            daemon=True
        ).start()
```

### The Problem

1. **Unlimited Thread Creation**: Each query spawns a new thread
   - At 200 QPS, this creates 200 threads per second
   - At 1000 QPS, this creates 1000 threads per second
   - No limit on concurrent threads

2. **Thread Explosion**:
   - Each thread consumes ~8MB stack space
   - 1000 threads = ~8GB just for thread stacks
   - Context switching overhead becomes enormous
   - System can become unresponsive

3. **Queue Blocking**:
   - `query_queue.put()` can block if queue is full (though default is unbounded)
   - Worker threads may not keep up with query arrival rate
   - Backpressure not handled

4. **Resource Exhaustion**:
   - File descriptor limits
   - Memory limits
   - CPU thrashing from context switching

### Why It Hangs

At high QPS:
1. LoadGen sends queries rapidly to `issue_query()`
2. Queries are queued faster than workers can process
3. Each query spawns a new thread
4. System runs out of resources (memory, file descriptors, CPU)
5. Thread creation becomes slow or fails
6. Queue processing stalls
7. System appears hung

## Solution: ThreadPoolExecutor (Same as Offline)

### Proposed Implementation

Use `ThreadPoolExecutor` to limit concurrent threads, similar to the offline async implementation:

```python
def _process_queries_worker(self):
    """Worker thread to process queued queries using thread pool."""
    with ThreadPoolExecutor(max_workers=self.max_concurrent_queries) as executor:
        while True:
            qitem = self.query_queue.get()
            
            if qitem is None:
                break
            
            # Submit to thread pool instead of spawning new thread
            executor.submit(self._async_process_query, input_ids_tensor, qitem.id)
```

### Benefits

1. **Bounded Concurrency**: Limits number of concurrent requests
2. **Resource Control**: Prevents thread explosion
3. **Better Scalability**: Handles high QPS without hanging
4. **Consistent Architecture**: Same pattern as offline async mode

### Implementation Details

#### Configuration
- Add `server_max_concurrent_queries` parameter (default: 50)
- Command-line flag: `--server-max-concurrent-queries <N>`
- Configurable per server capacity

#### Architecture Change
- Replace unlimited thread spawning with ThreadPoolExecutor
- Each worker thread uses its own executor (or shared pool)
- Maintains streaming API support
- Preserves first token handling

## Scalability Analysis

### Current (Unlimited Threads)

| QPS | Threads Created/sec | Memory (8MB/thread) | Status |
|-----|---------------------|---------------------|--------|
| 10  | 10                  | 80 MB               | OK     |
| 50  | 50                  | 400 MB              | OK     |
| 100 | 100                 | 800 MB              | Slow   |
| 200 | 200                 | 1.6 GB              | Hangs  |
| 500 | 500                 | 4 GB                | Hangs  |
| 1000| 1000                | 8 GB                | Crashes|

### Proposed (Bounded Thread Pool)

| QPS | Concurrent Threads | Memory (8MB/thread) | Status |
|-----|---------------------|---------------------|--------|
| 10  | 10                  | 80 MB               | OK     |
| 50  | 50                  | 400 MB              | OK     |
| 100 | 50                  | 400 MB              | OK     |
| 200 | 50                  | 400 MB              | OK     |
| 500 | 50                  | 400 MB              | OK     |
| 1000| 50                  | 400 MB              | OK     |

**Key Insight**: With bounded concurrency, memory usage is constant regardless of QPS.

## Implementation Plan

### Phase 1: Add ThreadPoolExecutor to Server Client

1. Add `max_concurrent_queries` parameter to `LoadGenServerClient`
2. Replace thread spawning with ThreadPoolExecutor
3. Maintain backward compatibility (default behavior)

### Phase 2: Configuration

1. Add `--server-max-concurrent-queries` command-line flag
2. Add to config parsing
3. Document recommended values

### Phase 3: Testing

1. Test with various QPS values (10, 50, 100, 200, 500, 1000)
2. Verify no hangs occur
3. Measure throughput and latency
4. Compare with original implementation

## Recommended Default Values

| Server Capacity | Recommended max_concurrent_queries |
|----------------|-------------------------------------|
| Small (1 GPU)   | 20-30                               |
| Medium (2-4 GPU)| 50-100                             |
| Large (8+ GPU) | 100-200                             |

**Note**: Should be tuned based on:
- Server processing capacity
- Average request latency
- Target QPS
- Available memory

## Alternative Solutions Considered

### 1. Queue Size Limit
- **Issue**: Doesn't solve thread explosion, just delays it
- **Verdict**: Not sufficient

### 2. Semaphore-based Limiting
- **Issue**: More complex, doesn't reuse threads efficiently
- **Verdict**: ThreadPoolExecutor is better

### 3. Async I/O (aiohttp)
- **Issue**: Requires significant refactoring
- **Verdict**: Future enhancement, ThreadPoolExecutor is good intermediate solution

## Conclusion

The hang issue is caused by **unlimited thread creation** at high QPS. Using `ThreadPoolExecutor` with bounded concurrency (same approach as offline async) will:
- ✅ Fix the hang issue
- ✅ Improve scalability
- ✅ Maintain consistent architecture
- ✅ Be easy to implement

This is a **high priority fix** for production use at scale.
