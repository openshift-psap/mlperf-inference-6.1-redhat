#!/bin/bash
# QPS sweep test for k8s-service and llmd-standalone endpoints
# Runs 8 experiments: 4 QPS values x 2 service endpoints

set -euo pipefail

HARNESS="/mnt/models/test-mlperf/mlperf-inference-6.1-redhat/harness/harness_main.py"
USER_CONF="/mnt/models/test-mlperf/mlperf-inference-6.1-redhat/harness/default.conf"
DATASET="/mnt/models/datasets/gpt-oss_data/perf/perf_eval_ref.parquet"

# Cleanup audit.config if it exists
rm -f "/mnt/models/test-mlperf/mlperf-inference-6.1-redhat/harness/audit.config"

# Set ulimit
ulimit -n 65535

# Environment Variables
export DATASET_DIR="/mnt/models/datasets/gpt-oss_data/"
export PERF_DATASET="/mnt/models/datasets/gpt-oss_data/perf/perf_eval_ref.parquet"
export ACC_DATASET="/mnt/models/datasets/gpt-oss_data/acc/acc_eval_ref.parquet"
export COMPLIANCE_DATASET="/mnt/models/datasets/gpt-oss_data/acc/acc_eval_compliance_gpqa.parquet"
export OUTPUT_DIR="./harness_output"
export AWS_ACCESS_KEY_ID="dummy"
export AWS_SECRET_ACCESS_KEY="dummy"
export HF_HOME="/mnt/models"
export MODEL_CATEGORY="gpt-oss-120b"
export MODEL="openai/gpt-oss-120b"
export BACKEND="vllm"
export LG_MODEL_NAME="gpt-oss-120b"
export SCENARIO="Server"
export MLFLOW_TRACKING_URI="https://mlflow.apps.psap-automation.ibm.rhperfscale.org"
export MLFLOW_EXPERIMENT_NAME="mlperf-6.1-gb200-gpt-oss-120b"
export MLFLOW_TRACKING_USERNAME="npalaska"
export MLFLOW_TRACKING_PASSWORD="AW1DO_2HUneEXAziJ8TBfYzD"
export MLFLOW_TRACKING_INSECURE_TLS="true"

QPS_VALUES=(10 20 30 40)

run_test() {
    local WORKSPACE="$1"
    local QPS="$2"
    local API_URL="$3"
    local MLFLOW_TAG="$4"

    local OUTPUT="harness_output/server-run-${WORKSPACE}-qps${QPS}/performance"

    echo ""
    echo "============================================================"
    echo "  ${WORKSPACE} — QPS ${QPS}"
    echo "  API: ${API_URL}"
    echo "  Output: ${OUTPUT}"
    echo "============================================================"
    echo ""

    python3 "$HARNESS" \
        --model-category gpt-oss-120b \
        --model "openai/gpt-oss-120b" \
        --dataset-path "$DATASET" \
        --backend vllm \
        --lg-model-name gpt-oss-120b \
        --test-mode performance \
        --api-server-url "$API_URL" \
        --scenario Server \
        --output-dir "$OUTPUT" \
        --mlflow-experiment-name "mlperf-6.1-gb200-gpt-oss-120b" \
        --mlflow-tracking-uri "https://mlflow.apps.psap-automation.ibm.rhperfscale.org" \
        --mlflow-description "Server Performance QPS${QPS} ${WORKSPACE}" \
        --mlflow-tag "${MLFLOW_TAG}" \
        --server-target-qps "$QPS" \
        --user-conf "$USER_CONF" \
        --num-workers 12

    echo ""
    echo "  Completed: ${WORKSPACE} QPS ${QPS}"
    echo ""
}

# k8s-service tests
K8S_URL="http://vllm-direct.llm-d-bench.svc.cluster.local:8000"
for QPS in "${QPS_VALUES[@]}"; do
    run_test "k8s-service" "$QPS" "$K8S_URL" \
        "test_type:performance,scenario:Server,qps:${QPS},workspace:k8s-service,mnbt:8192,mccs:2048"
done

# llmd-standalone tests
LLMD_URL="http://optimized-baseline-epp.llm-d-bench.svc.cluster.local:80"
for QPS in "${QPS_VALUES[@]}"; do
    run_test "llmd-standalone-mlperf-60-scorer" "$QPS" "$LLMD_URL" \
        "test_type:performance,scenario:Server,qps:${QPS},workspace:llmd-standalone,queue-scorer:2,kv-cache-util:2,mnbt:8192,mccs:2048"
done

echo ""
echo "============================================================"
echo "  All 8 experiments complete"
echo "============================================================"
echo ""
echo "Results:"
for QPS in "${QPS_VALUES[@]}"; do
    echo "  k8s-service QPS ${QPS}:      harness_output/server-run-k8s-service-qps${QPS}/performance/"
    echo "  llmd-standalone QPS ${QPS}:   harness_output/server-run-llmd-standalone-qps${QPS}/performance/"
done
