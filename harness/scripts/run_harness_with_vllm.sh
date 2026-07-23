#!/bin/bash
# ============================================================================
# run_harness_with_vllm.sh
# -------------------------
# Script to spin up a vLLM server, run harness with varying target QPS,
# and clean up the server when done.
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Configuration
# ============================================================================

# Default values (can be overridden via command-line arguments)
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_WORK_DIR="${VLLM_WORK_DIR:-./vllm_server_output}"
HARNESS_OUTPUT_DIR="${HARNESS_OUTPUT_DIR:-./harness_output}"
DATASET_PATH="${DATASET_PATH:-}"
USER_CONF="${USER_CONF:-user.conf}"
LG_MODEL_NAME="${LG_MODEL_NAME:-llama3_1-8b}"

# QPS values to test (can be overridden)
QPS_VALUES="${QPS_VALUES:-10 20 50 100 200}"

# vLLM server arguments
VLLM_ARGS="${VLLM_ARGS:---tensor-parallel-size 1 --gpu-memory-utilization 0.8}"

# Harness arguments (additional to what's set by script)
HARNESS_EXTRA_ARGS="${HARNESS_EXTRA_ARGS:-}"

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

# Global variable to track current vLLM server PID
VLLM_PID=""

start_vllm_server() {
    local qps=$1
    local vllm_log_file=$2
    
    log "Starting vLLM server for QPS=$qps..."
    log "  Model: $MODEL"
    log "  Port: $VLLM_PORT"
    log "  Host: $VLLM_HOST"
    log "  Log file: $vllm_log_file"
    log "  Args: $VLLM_ARGS"
    
    # Start vLLM server in background, redirecting all output to log file
    python -m vllm.entrypoints.api_server \
        --model "$MODEL" \
        --port "$VLLM_PORT" \
        --host "$VLLM_HOST" \
        $VLLM_ARGS \
        > "$vllm_log_file" 2>&1 &
    
    VLLM_PID=$!
    log "vLLM server started with PID: $VLLM_PID"
    
    # Wait for server to be ready
    log "Waiting for vLLM server to be ready..."
    SERVER_URL="http://${VLLM_HOST}:${VLLM_PORT}"
    MAX_WAIT=300  # 5 minutes
    WAIT_INTERVAL=5
    ELAPSED=0
    
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        if curl -s -f "$SERVER_URL/health" > /dev/null 2>&1; then
            log "vLLM server is ready!"
            return 0
        fi
        
        # Check if process is still running
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            error "vLLM server process died. Check logs: $vllm_log_file"
        fi
        
        sleep $WAIT_INTERVAL
        ELAPSED=$((ELAPSED + WAIT_INTERVAL))
        log "  Still waiting... (${ELAPSED}s/${MAX_WAIT}s)"
    done
    
    error "vLLM server did not become ready within ${MAX_WAIT}s. Check logs: $vllm_log_file"
}

stop_vllm_server() {
    local qps=$1
    
    if [ -z "$VLLM_PID" ]; then
        return 0
    fi
    
    log "Stopping vLLM server for QPS=$qps (PID: $VLLM_PID)..."
    
    # Try graceful shutdown first
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        # Wait up to 30 seconds for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                log "vLLM server stopped gracefully"
                VLLM_PID=""
                return 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            log "Force killing vLLM server..."
            kill -9 "$VLLM_PID" 2>/dev/null || true
            wait "$VLLM_PID" 2>/dev/null || true
        fi
    fi
    
    VLLM_PID=""
    log "vLLM server stopped"
    
    # Brief pause to ensure port is released
    sleep 2
}

cleanup() {
    log "Cleaning up..."
    
    # Kill vLLM server if it's still running
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log "Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    
    # Also try to kill any remaining vLLM processes
    pkill -f "python.*vllm.entrypoints.api_server" 2>/dev/null || true
    
    log "Cleanup complete"
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# ============================================================================
# Parse Command-Line Arguments
# ============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --model MODEL                 Model name/path (default: $MODEL)
    --port PORT                   vLLM server port (default: $VLLM_PORT)
    --host HOST                   vLLM server host (default: $VLLM_HOST)
    --dataset-path PATH           Path to dataset file (REQUIRED)
    --qps-values "QPS1 QPS2 ..."  QPS values to test (default: "$QPS_VALUES")
    --user-conf PATH              LoadGen user config (default: $USER_CONF)
    --lg-model-name NAME          LoadGen model name (default: $LG_MODEL_NAME)
    --vllm-args "ARGS"            Additional vLLM server arguments
    --harness-args "ARGS"         Additional harness arguments
    --output-dir DIR              Harness output directory (default: $HARNESS_OUTPUT_DIR)
                                  Each QPS measurement will create a subdirectory: qps_<QPS>
    --vllm-work-dir DIR            vLLM server work directory (default: $VLLM_WORK_DIR)
                                  Each QPS measurement will create a log file: vllm_server_qps_<QPS>.log
    --help                        Show this help message

Note:
    The vLLM server will be started and stopped for EACH QPS measurement.
    This ensures a clean state for each test and prevents interference between runs.

Environment Variables:
    All options can also be set via environment variables (see script for names)

Examples:
    # Basic usage
    $0 --dataset-path ./dataset.pkl --qps-values "10 50 100"

    # With custom model and port
    $0 --model meta-llama/Llama-2-70B-Instruct --port 8001 \\
       --dataset-path ./dataset.pkl --qps-values "20 100"

    # With additional vLLM arguments
    $0 --dataset-path ./dataset.pkl \\
       --vllm-args "--tensor-parallel-size 2 --max-model-len 4096"

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --host)
            VLLM_HOST="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --qps-values)
            QPS_VALUES="$2"
            shift 2
            ;;
        --user-conf)
            USER_CONF="$2"
            shift 2
            ;;
        --lg-model-name)
            LG_MODEL_NAME="$2"
            shift 2
            ;;
        --vllm-args)
            VLLM_ARGS="$2"
            shift 2
            ;;
        --harness-args)
            HARNESS_EXTRA_ARGS="$2"
            shift 2
            ;;
        --output-dir)
            HARNESS_OUTPUT_DIR="$2"
            shift 2
            ;;
        --vllm-work-dir)
            VLLM_WORK_DIR="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1. Use --help for usage."
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATASET_PATH" ]; then
    error "--dataset-path is required. Use --help for usage."
fi

if [ ! -f "$DATASET_PATH" ]; then
    error "Dataset file not found: $DATASET_PATH"
fi

# ============================================================================
# Run Harness with Varying QPS
# ============================================================================

log "Starting harness runs with varying QPS values..."
log "QPS values to test: $QPS_VALUES"
log "Note: vLLM server will be started and stopped for each QPS measurement"

# Create base output directories
mkdir -p "$HARNESS_OUTPUT_DIR"
mkdir -p "$VLLM_WORK_DIR"

SERVER_URL="http://${VLLM_HOST}:${VLLM_PORT}"

# Run harness for each QPS value
for QPS in $QPS_VALUES; do
    log "========================================================================"
    log "QPS Measurement: $QPS"
    log "========================================================================"
    
    # Create per-QPS directories
    QPS_OUTPUT_DIR="${HARNESS_OUTPUT_DIR}/qps_${QPS}"
    QPS_VLLM_LOG="${VLLM_WORK_DIR}/vllm_server_qps_${QPS}.log"
    
    mkdir -p "$QPS_OUTPUT_DIR"
    
    # Start vLLM server for this QPS measurement
    start_vllm_server "$QPS" "$QPS_VLLM_LOG"
    
    # Run harness
    log "Running harness with target QPS: $QPS"
    log "Harness output directory: $QPS_OUTPUT_DIR"
    
    HARNESS_SUCCESS=false
    if python harness/harness_main.py \
        --model "$MODEL" \
        --dataset-path "$DATASET_PATH" \
        --scenario Server \
        --test-mode performance \
        --api-server-url "$SERVER_URL" \
        --server-target-qps "$QPS" \
        --user-conf "$USER_CONF" \
        --lg-model-name "$LG_MODEL_NAME" \
        --output-dir "$QPS_OUTPUT_DIR" \
        $HARNESS_EXTRA_ARGS; then
        HARNESS_SUCCESS=true
        log "✓ Harness run completed successfully for QPS=$QPS"
    else
        log "✗ Harness run failed for QPS=$QPS"
        log "  Check logs: $QPS_OUTPUT_DIR"
        log "  vLLM server logs: $QPS_VLLM_LOG"
    fi
    
    # Stop vLLM server for this QPS measurement
    stop_vllm_server "$QPS"
    
    if [ "$HARNESS_SUCCESS" = true ]; then
        log "Results saved to: $QPS_OUTPUT_DIR"
        log "vLLM server logs saved to: $QPS_VLLM_LOG"
    fi
    
    log ""
    
    # Brief pause between runs to ensure clean state
    sleep 3
done

log "========================================================================"
log "All harness runs completed!"
log "Results saved to: $HARNESS_OUTPUT_DIR"
log "vLLM server logs saved to: $VLLM_WORK_DIR"
log "========================================================================"
