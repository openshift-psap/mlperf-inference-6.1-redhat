#!/bin/bash
# MLPerf Server Scenario — GPT-OSS-120B on GB200 NVL4
#
# Usage:
#   bash run_server.sh                    # Run all (performance + accuracy + compliance)
#   bash run_server.sh performance        # Performance only (with retries)
#   bash run_server.sh accuracy           # Accuracy only
#   bash run_server.sh compliance         # Compliance only (TEST07 + TEST09)
#   bash run_server.sh compliance test07  # TEST07 only
#   bash run_server.sh compliance test09  # TEST09 only

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="${SCRIPT_DIR}/harness_main.py"
USER_CONF="${SCRIPT_DIR}/default.conf"
AUDIT_OVERRIDE="audit-override.cfg"
PERF_DATASET="/mnt/models/datasets/gpt-oss_data/perf/perf_eval_ref.parquet"
ACC_DATASET="/mnt/models/datasets/gpt-oss_data/acc/acc_eval_ref.parquet"
COMPLIANCE_DATASET="/mnt/models/datasets/gpt-oss_data/acc/acc_eval_compliance_gpqa.parquet"
COMPLIANCE_DIR="${SCRIPT_DIR}/../compliance"

# Configuration
API_URL="http://vllm-direct.llm-d-bench.svc.cluster.local:8000"
QPS="${SERVER_TARGET_QPS:-39.14}"
OUTPUT_BASE="${OUTPUT_DIR:-harness_output/server}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MAX_PERF_RETRIES="${MAX_PERF_RETRIES:-5}"

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
export SCENARIO="Server"

echo "=========================================="
echo "MLPerf Server Scenario — GPT-OSS-120B"
echo "=========================================="
echo "  API URL:     ${API_URL}"
echo "  QPS:         ${QPS}"
echo "  Workers:     ${NUM_WORKERS}"
echo "  Output:      ${OUTPUT_BASE}"
echo "=========================================="

run_performance() {
    for attempt in $(seq 1 "$MAX_PERF_RETRIES"); do
        echo ""
        echo "================================================================"
        echo "  Server Performance [attempt ${attempt}/${MAX_PERF_RETRIES}]"
        echo "================================================================"

        local OUT_DIR="${OUTPUT_BASE}/performance"
        if [[ "$attempt" -gt 1 ]]; then
            OUT_DIR="${OUTPUT_BASE}/performance-attempt${attempt}"
        fi

        python3 "$HARNESS" \
            --model-category gpt-oss-120b \
            --model "openai/gpt-oss-120b" \
            --dataset-path "$PERF_DATASET" \
            --backend vllm \
            --lg-model-name gpt-oss-120b \
            --test-mode performance \
            --api-server-url "$API_URL" \
            --scenario Server \
            --output-dir "$OUT_DIR" \
            --server-target-qps "$QPS" \
            --user-conf "$USER_CONF" \
            --num-workers "$NUM_WORKERS"

        local SUMMARY="${OUT_DIR}/mlperf/mlperf_log_summary.txt"
        if [[ -f "$SUMMARY" ]] && grep -q "Result is : VALID" "$SUMMARY"; then
            echo "  ✓ VALID on attempt ${attempt}"
            grep "Completed tokens per second" "$SUMMARY" | head -1
            break
        else
            echo "  ✗ INVALID on attempt ${attempt}"
            if [[ "$attempt" -ge "$MAX_PERF_RETRIES" ]]; then
                echo "  Max retries reached."
            fi
        fi
    done
}

run_accuracy() {
    echo ""
    echo "================================================================"
    echo "  Server Accuracy"
    echo "================================================================"

    python3 "$HARNESS" \
        --model-category gpt-oss-120b \
        --model "openai/gpt-oss-120b" \
        --dataset-path "$ACC_DATASET" \
        --backend vllm \
        --lg-model-name gpt-oss-120b \
        --test-mode accuracy \
        --api-server-url "$API_URL" \
        --scenario Server \
        --output-dir "${OUTPUT_BASE}/accuracy" \
        --server-target-qps "$QPS" \
        --user-conf "$USER_CONF" \
        --num-workers "$NUM_WORKERS"
}

run_compliance_test07() {
    echo ""
    echo "================================================================"
    echo "  Server Compliance TEST07"
    echo "================================================================"

    cp "${COMPLIANCE_DIR}/TEST07/gpt-oss-120b/audit.config" "${SCRIPT_DIR}/audit.config"

    python3 "$HARNESS" \
        --model-category gpt-oss-120b \
        --model "openai/gpt-oss-120b" \
        --dataset-path "$COMPLIANCE_DATASET" \
        --backend vllm \
        --lg-model-name gpt-oss-120b \
        --test-mode performance \
        --api-server-url "$API_URL" \
        --scenario Server \
        --output-dir "${OUTPUT_BASE}/compliance/test07" \
        --server-target-qps "$QPS" \
        --user-conf "$AUDIT_OVERRIDE" \
        --audit-config audit.config \
        --num-workers "$NUM_WORKERS"

    rm -f "${SCRIPT_DIR}/audit.config"
}

run_compliance_test09() {
    echo ""
    echo "================================================================"
    echo "  Server Compliance TEST09"
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
        --scenario Server \
        --output-dir "${OUTPUT_BASE}/compliance/test09" \
        --server-target-qps "$QPS" \
        --user-conf "$USER_CONF" \
        --audit-config audit.config \
        --num-workers "$NUM_WORKERS"

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
        echo "  Server scenario complete"
        echo "  Results: ${OUTPUT_BASE}/"
        echo "================================================================"
        ;;
    *) echo "Usage: $0 [performance|accuracy|compliance [test07|test09]|all]"; exit 1 ;;
esac
