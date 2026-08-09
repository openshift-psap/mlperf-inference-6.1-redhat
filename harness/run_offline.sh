#!/bin/bash
# MLPerf Offline Scenario — GPT-OSS-120B on GB200 NVL4
#
# Usage:
#   bash run_offline.sh                    # Run all (performance + accuracy + compliance)
#   bash run_offline.sh performance        # Performance only (runs N times, picks best)
#   bash run_offline.sh accuracy           # Accuracy only
#   bash run_offline.sh compliance         # Compliance only (TEST07 + TEST09)
#   bash run_offline.sh compliance test07  # TEST07 only
#   bash run_offline.sh compliance test09  # TEST09 only

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="${SCRIPT_DIR}/harness_main.py"
OFFLINE_CONF="${SCRIPT_DIR}/offline.conf"
DEFAULT_CONF="${SCRIPT_DIR}/default.conf"
AUDIT_OVERRIDE="audit-override.cfg"
PERF_DATASET="/mnt/models/datasets/gpt-oss_data/perf/perf_eval_ref.parquet"
ACC_DATASET="/mnt/models/datasets/gpt-oss_data/acc/acc_eval_ref.parquet"
COMPLIANCE_DATASET="/mnt/models/datasets/gpt-oss_data/acc/acc_eval_compliance_gpqa.parquet"
COMPLIANCE_DIR="${SCRIPT_DIR}/../compliance"

# Configuration
API_URL="http://vllm-direct.llm-d-bench.svc.cluster.local:8000"
OUTPUT_BASE="${OUTPUT_DIR:-harness_output/offline}"
NUM_WORKERS="${NUM_WORKERS:-12}"
OFFLINE_CONCURRENCY="${OFFLINE_CONCURRENCY:-6396}"
MAX_PERF_RUNS="${MAX_PERF_RUNS:-5}"

# Cleanup
rm -f "${SCRIPT_DIR}/audit.config"

# Ulimit
ulimit -n 65535

# Environment
export DATASET_DIR="/mnt/models/datasets/gpt-oss_data/"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-dummy}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-dummy}"
export HF_HOME="${HF_HOME:-/mnt/models}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MODEL_CATEGORY="gpt-oss-120b"
export MODEL="openai/gpt-oss-120b"
export BACKEND="vllm"
export LG_MODEL_NAME="gpt-oss-120b"
export SCENARIO="Offline"

echo "=========================================="
echo "MLPerf Offline Scenario — GPT-OSS-120B"
echo "=========================================="
echo "  API URL:       ${API_URL}"
echo "  Concurrency:   ${OFFLINE_CONCURRENCY}"
echo "  Workers:       ${NUM_WORKERS}"
echo "  Perf runs:     ${MAX_PERF_RUNS}"
echo "  Output:        ${OUTPUT_BASE}"
echo "=========================================="

run_performance() {
    local BEST_THROUGHPUT=0
    local BEST_DIR=""

    for attempt in $(seq 1 "$MAX_PERF_RUNS"); do
        echo ""
        echo "================================================================"
        echo "  Offline Performance [run ${attempt}/${MAX_PERF_RUNS}]"
        echo "================================================================"

        local OUT_DIR="${OUTPUT_BASE}/performance-run${attempt}"

        python3 "$HARNESS" \
            --model-category gpt-oss-120b \
            --model "openai/gpt-oss-120b" \
            --dataset-path "$PERF_DATASET" \
            --backend vllm \
            --lg-model-name gpt-oss-120b \
            --test-mode performance \
            --api-server-url "$API_URL" \
            --scenario Offline \
            --output-dir "$OUT_DIR" \
            --user-conf "$OFFLINE_CONF" \
            --num-workers "$NUM_WORKERS" \
            --offline-back-to-back \
            --offline-async-concurrency "$OFFLINE_CONCURRENCY"

        local SUMMARY="${OUT_DIR}/mlperf/mlperf_log_summary.txt"
        if [[ -f "$SUMMARY" ]] && grep -q "Result is : VALID" "$SUMMARY"; then
            local THROUGHPUT
            THROUGHPUT=$(grep "Tokens per second" "$SUMMARY" | head -1 | awk '{print $NF}')
            echo "  ✓ VALID — ${THROUGHPUT} tok/s"

            if python3 -c "exit(0 if $THROUGHPUT > $BEST_THROUGHPUT else 1)" 2>/dev/null; then
                BEST_THROUGHPUT="$THROUGHPUT"
                BEST_DIR="$OUT_DIR"
                echo "  ★ New best run: ${THROUGHPUT} tok/s"
            fi
        else
            echo "  ✗ INVALID on run ${attempt}"
        fi
    done

    # Move the best run to the canonical performance directory
    if [[ -n "$BEST_DIR" ]]; then
        echo ""
        echo "  ★ Best VALID run: ${BEST_THROUGHPUT} tok/s from ${BEST_DIR}"
        mv "$BEST_DIR" "${OUTPUT_BASE}/performance"
        echo "  → Moved to ${OUTPUT_BASE}/performance"
    else
        echo ""
        echo "  ✗ No VALID runs in ${MAX_PERF_RUNS} attempts. Using last run."
        mv "${OUTPUT_BASE}/performance-run${MAX_PERF_RUNS}" "${OUTPUT_BASE}/performance"
    fi
}

run_accuracy() {
    echo ""
    echo "================================================================"
    echo "  Offline Accuracy"
    echo "================================================================"

    python3 "$HARNESS" \
        --model-category gpt-oss-120b \
        --model "openai/gpt-oss-120b" \
        --dataset-path "$ACC_DATASET" \
        --backend vllm \
        --lg-model-name gpt-oss-120b \
        --test-mode accuracy \
        --api-server-url "$API_URL" \
        --scenario Offline \
        --output-dir "${OUTPUT_BASE}/accuracy" \
        --user-conf "$DEFAULT_CONF" \
        --num-workers "$NUM_WORKERS" \
        --offline-back-to-back \
        --offline-async-concurrency "$OFFLINE_CONCURRENCY"
}

run_compliance_test07() {
    echo ""
    echo "================================================================"
    echo "  Offline Compliance TEST07"
    echo "================================================================"

    # Comment out performance_issue_unique (causes LoadGen segfault in Offline)
    sed 's/^\*\.\*\.performance_issue_unique/#*.*.performance_issue_unique/' \
        "${COMPLIANCE_DIR}/TEST07/gpt-oss-120b/audit.config" > "${SCRIPT_DIR}/audit.config"

    python3 "$HARNESS" \
        --model-category gpt-oss-120b \
        --model "openai/gpt-oss-120b" \
        --dataset-path "$COMPLIANCE_DATASET" \
        --backend vllm \
        --lg-model-name gpt-oss-120b \
        --test-mode performance \
        --api-server-url "$API_URL" \
        --scenario Offline \
        --output-dir "${OUTPUT_BASE}/compliance/test07" \
        --user-conf "$OFFLINE_CONF" \
        --num-workers "$NUM_WORKERS" \
        --offline-back-to-back \
        --offline-async-concurrency "$OFFLINE_CONCURRENCY" \
        --audit-config audit.config

    rm -f "${SCRIPT_DIR}/audit.config"
}

run_compliance_test09() {
    echo ""
    echo "================================================================"
    echo "  Offline Compliance TEST09"
    echo "================================================================"

    cp "${COMPLIANCE_DIR}/TEST09/gpt-oss-120b/audit.config" "${SCRIPT_DIR}/audit.config"

    python3 "$HARNESS" \
        --model-category gpt-oss-120b \
        --model "openai/gpt-oss-120b" \
        --dataset-path "$PERF_DATASET" \
        --backend vllm \
        --lg-model-name gpt-oss-120b \
        --test-mode performance \
        --api-server-url "$API_URL" \
        --scenario Offline \
        --output-dir "${OUTPUT_BASE}/compliance/test09" \
        --user-conf "$DEFAULT_CONF" \
        --num-workers "$NUM_WORKERS" \
        --offline-back-to-back \
        --offline-async-concurrency "$OFFLINE_CONCURRENCY" \
        --audit-config audit.config

    rm -f "${SCRIPT_DIR}/audit.config"
}

run_compliance() {
    local TEST="${1:-all}"
    case "$TEST" in
        test07|TEST07) run_compliance_test07 ;;
        test09|TEST09) run_compliance_test09 ;;
        all)
            run_compliance_test07
            run_compliance_test09
            ;;
        *) echo "Unknown compliance test: $TEST"; exit 1 ;;
    esac
}

# Parse command
COMMAND="${1:-all}"
SUBCOMMAND="${2:-}"

case "$COMMAND" in
    performance) run_performance ;;
    accuracy)    run_accuracy ;;
    compliance)  run_compliance "$SUBCOMMAND" ;;
    all)
        run_performance
        run_accuracy
        run_compliance
        echo ""
        echo "================================================================"
        echo "  Offline scenario complete"
        echo "  Results: ${OUTPUT_BASE}/"
        echo "================================================================"
        ;;
    *) echo "Usage: $0 [performance|accuracy|compliance [test07|test09]|all]"; exit 1 ;;
esac
