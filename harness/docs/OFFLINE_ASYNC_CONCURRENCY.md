# Offline Async Concurrency Implementation

## Overview

The `offline-async-concurrency` feature enables asynchronous request processing for the Offline scenario when `--offline-back-to-back` is enabled. Instead of sending requests sequentially (one at a time, waiting for each response), this implementation sends multiple requests concurrently and processes responses as they arrive.

## Problem Statement

### Original Behavior (Sequential)
```
Request 1 → Wait for Response 1 → Request 2 → Wait for Response 2 → ...
```
- **Issue**: Low throughput - each request must complete before the next starts
- **Latency**: Total time = sum of all request latencies
- **Resource utilization**: Network and server idle between requests

### New Behavior (Async with Concurrency Control)
```
Request 1, Request 2, ..., Request N (concurrent) → Process responses as they arrive
```
- **Benefit**: Higher throughput - multiple requests in-flight simultaneously
- **Latency**: Total time ≈ max(latency of concurrent requests)
- **Resource utilization**: Better network and server utilization

## Implementation Details

### Architecture

The implementation uses Python's `concurrent.futures.ThreadPoolExecutor` to manage concurrent request processing:

```python
with ThreadPoolExecutor(max_workers=self.offline_async_concurrency) as executor:
    # Submit all requests
    future_to_sample = {
        executor.submit(process_single_async, q_sample): q_sample 
        for q_sample in query_samples
    }
    
    # Process responses as they complete
    for future in as_completed(future_to_sample):
        success, query_id = future.result()
```

### Key Components

#### 1. Thread Pool Executor
- **Purpose**: Manages a pool of worker threads for concurrent request processing
- **Configuration**: `max_workers` controls how many requests are in-flight at once
- **Default**: 10 concurrent requests
- **Configurable**: Via `--offline-async-concurrency <N>` command-line flag

#### 2. Async Processing Function
```python
def process_single_async(q_sample: 'lg.QuerySample') -> tuple:
    """Process a single query asynchronously. Returns (success, q_sample.id)."""
    try:
        # Blocking call, but runs in parallel with other requests
        self._process_api_single(q_sample, temperature, top_k, top_p)
        return (True, q_sample.id)
    except Exception as e:
        # Error handling
        return (False, q_sample.id)
```

- Each request still uses synchronous `requests.post()` internally
- Multiple requests run in parallel threads
- Thread-safe response handling for LoadGen

#### 3. Response Processing
- Uses `as_completed()` to process futures as they finish
- Responses sent to LoadGen immediately when ready
- Thread-safe progress tracking with locks

#### 4. Thread Safety
```python
progress_lock = threading.Lock()
with progress_lock:
    completed_samples += 1
```
- Shared counters protected by locks
- LoadGen response handling is thread-safe

### Flow Diagram

```
LoadGen calls issue_query() with N samples
    ↓
Create ThreadPoolExecutor (max_workers=10)
    ↓
Submit all N requests to thread pool
    ├─ Request 1 → Worker Thread 1 → HTTP Request → Response 1
    ├─ Request 2 → Worker Thread 2 → HTTP Request → Response 2
    ├─ ...
    └─ Request N → Worker Thread 10 → HTTP Request → Response N
    ↓
Process futures as they complete (as_completed)
    ├─ Response 1 arrives → Send to LoadGen
    ├─ Response 2 arrives → Send to LoadGen
    └─ ...
    ↓
All responses processed → Return to LoadGen
```

## Configuration

### Command-Line Arguments

```bash
# Enable async back-to-back mode with default concurrency (10)
python harness_main.py --offline-back-to-back --model <model> --dataset-path <path>

# Custom concurrency level
python harness_main.py --offline-back-to-back --offline-async-concurrency 20 --model <model> --dataset-path <path>
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--offline-back-to-back` | False | Enable async individual request mode |
| `--offline-async-concurrency` | 10 | Max concurrent requests in-flight |

### Choosing Concurrency Level

**Factors to consider:**
1. **Server capacity**: How many concurrent requests can the server handle?
2. **Network bandwidth**: More concurrency = more network usage
3. **Latency requirements**: Higher concurrency can increase tail latency
4. **Client resources**: More threads = more memory/CPU usage

**Recommendations:**
- **Start with default (10)**: Good balance for most scenarios
- **Increase (20-50)**: If server can handle more and you need higher throughput
- **Decrease (5)**: If experiencing timeouts or server overload
- **Monitor**: Watch server metrics and adjust based on performance

## Performance Characteristics

### Throughput
- **Sequential**: ~1 request per average_latency
- **Async (N concurrent)**: ~N requests per average_latency (up to server limit)

### Latency
- **Sequential**: Total time = N × average_latency
- **Async**: Total time ≈ max_latency + overhead (much faster for large N)

### Resource Usage
- **Memory**: Each thread uses ~8MB stack space
- **CPU**: Context switching overhead increases with more threads
- **Network**: More concurrent connections

## Thread Safety

### LoadGen Response Handling
- `lg.QuerySamplesComplete()` is called from multiple threads
- LoadGen C++ library handles thread safety internally
- Response arrays stored in `self.response_arrays` (protected by GIL)

### Progress Tracking
- Shared counters (`completed_samples`, `failed_samples`) protected by locks
- Logging is thread-safe (Python's logging module)

## Error Handling

1. **Request failures**: Caught in `process_single_async()`, error response sent to LoadGen
2. **Future exceptions**: Caught when calling `future.result()`
3. **Thread pool cleanup**: Automatic via context manager (`with ThreadPoolExecutor`)

## Limitations

1. **Still uses blocking HTTP**: Each request uses synchronous `requests.post()`
   - Could be improved with `aiohttp` for true async I/O (future enhancement)
2. **Thread pool overhead**: Context switching cost increases with more threads
3. **Memory usage**: Each thread has stack space overhead
4. **GIL limitations**: Python's GIL limits true parallelism (I/O bound operations still benefit)

## Comparison with Server Scenario

| Aspect | Offline Async | Server Scenario |
|--------|---------------|------------------|
| Architecture | ThreadPoolExecutor | Worker threads + Queue |
| Concurrency control | Fixed pool size | Unlimited threads per query |
| Scalability | Limited by pool size | Can create thread explosion |
| Use case | Batch processing | Real-time query handling |

## Future Enhancements

1. **True async I/O**: Use `aiohttp` instead of `requests` for better scalability
2. **Dynamic concurrency**: Adjust pool size based on server response times
3. **Backpressure**: Reduce concurrency if server is overloaded
4. **Metrics**: Track request latency, queue depth, thread utilization

## Code Location

- **Main implementation**: `harness/Client/loadgen_client.py` - `LoadGenOfflineClient.issue_query()`
- **Configuration**: `harness/harness/arg_parser.py` - `--offline-async-concurrency`
- **Initialization**: `harness/harness/base_harness.py` - `initialize_client()`
