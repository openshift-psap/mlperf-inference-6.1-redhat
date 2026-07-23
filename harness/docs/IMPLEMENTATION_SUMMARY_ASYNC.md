# Async Implementation Summary

## Overview

This document summarizes the async implementations for both Offline and Server scenarios to improve scalability and prevent hangs at high QPS.

## 1. Offline Async Concurrency

### Implementation
- **Location**: `harness/Client/loadgen_client.py` - `LoadGenOfflineClient.issue_query()`
- **Technology**: `ThreadPoolExecutor` from `concurrent.futures`
- **Purpose**: Send multiple requests concurrently instead of sequentially

### Configuration
- **Flag**: `--offline-back-to-back` (enables async mode)
- **Parameter**: `--offline-async-concurrency <N>` (default: 10)
- **Documentation**: `harness/docs/OFFLINE_ASYNC_CONCURRENCY.md`

### Benefits
- ✅ Higher throughput (N requests in parallel)
- ✅ Better resource utilization
- ✅ Prevents sequential bottleneck

## 2. Server Client Hang Fix

### Problem
- **Issue**: Thread explosion at high QPS causing hangs
- **Root Cause**: Unlimited thread creation (one thread per query)
- **Impact**: System becomes unresponsive at 200+ QPS

### Solution
- **Implementation**: `ThreadPoolExecutor` per worker thread
- **Location**: `harness/Client/loadgen_client.py` - `LoadGenServerClient._process_queries_worker()`
- **Technology**: Same as offline async (ThreadPoolExecutor)

### Configuration
- **Parameter**: `--server-max-concurrent-queries <N>` (default: 50)
- **Documentation**: `harness/docs/SERVER_CLIENT_HANG_ANALYSIS.md`

### Benefits
- ✅ Prevents thread explosion
- ✅ Bounded resource usage
- ✅ Handles high QPS without hanging
- ✅ Consistent architecture with offline async

## 3. Bash Script for vLLM Testing

### Script
- **Location**: `harness/scripts/run_harness_with_vllm.sh`
- **Purpose**: Automate vLLM server lifecycle and harness testing

### Features
- Spins up vLLM server
- Waits for server to be ready
- Runs harness with varying QPS values
- Cleans up server on exit
- Configurable via command-line or environment variables

### Usage
```bash
./harness/scripts/run_harness_with_vllm.sh \
    --dataset-path ./dataset.pkl \
    --qps-values "10 50 100 200"
```

## Architecture Comparison

### Before (Server Scenario)
```
LoadGen → issue_query() → query_queue.put()
                              ↓
                    Worker Thread
                              ↓
                    threading.Thread.start()  ← UNLIMITED THREADS!
                              ↓
                    _async_process_query()
```

### After (Server Scenario)
```
LoadGen → issue_query() → query_queue.put()
                              ↓
                    Worker Thread
                              ↓
                    ThreadPoolExecutor.submit()  ← BOUNDED CONCURRENCY
                              ↓
                    _async_process_query()
```

## Scalability Analysis

| Scenario | QPS | Before | After |
|----------|-----|--------|-------|
| Server   | 10  | OK     | OK     |
| Server   | 100 | Slow   | OK     |
| Server   | 200 | Hangs  | OK     |
| Server   | 500 | Hangs  | OK     |
| Server   | 1000| Crashes| OK     |

## Configuration Recommendations

### Offline Async
- **Default**: 10 concurrent requests
- **High throughput**: 20-50
- **Limited resources**: 5

### Server Max Concurrent Queries
- **Small server (1 GPU)**: 20-30 per worker
- **Medium server (2-4 GPU)**: 50-100 per worker
- **Large server (8+ GPU)**: 100-200 per worker
- **Total**: num_workers × max_concurrent_queries

## Code Locations

### Offline Async
- Implementation: `harness/Client/loadgen_client.py:896-975`
- Config parsing: `harness/harness/arg_parser.py:94-95, 253`
- Client init: `harness/Client/loadgen_client.py:192-206`

### Server Fix
- Implementation: `harness/Client/loadgen_client.py:1593-1632`
- Config parsing: `harness/harness/arg_parser.py:92-93, 247`
- Client init: `harness/Client/loadgen_client.py:235, 1567-1569`

## Testing

### Manual Testing
```bash
# Test offline async
python harness/harness_main.py \
    --offline-back-to-back \
    --offline-async-concurrency 20 \
    --model <model> \
    --dataset-path <path>

# Test server with high QPS
python harness/harness_main.py \
    --scenario Server \
    --server-target-qps 500 \
    --server-max-concurrent-queries 100 \
    --model <model> \
    --dataset-path <path>
```

### Automated Testing
```bash
# Use the bash script
./harness/scripts/run_harness_with_vllm.sh \
    --dataset-path <path> \
    --qps-values "10 50 100 200 500"
```

## Future Enhancements

1. **True async I/O**: Use `aiohttp` instead of `requests` for better scalability
2. **Dynamic concurrency**: Adjust based on server response times
3. **Backpressure**: Reduce concurrency if server is overloaded
4. **Metrics**: Track thread utilization, queue depth, latency

## Related Documentation

- `harness/docs/OFFLINE_ASYNC_CONCURRENCY.md` - Detailed offline async documentation
- `harness/docs/SERVER_CLIENT_HANG_ANALYSIS.md` - Server hang analysis and fix
- `harness/scripts/run_harness_with_vllm.sh` - Automated testing script
