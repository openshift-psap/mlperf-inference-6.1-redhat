#!/bin/bash
# Compliance check script for MLPerf inference tests
# Usage: check_complaince.sh <output_dir> <TEST07|TEST09>

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DIR=$1
TEST=$2

if [ -z "$DIR" ] || [ -z "$TEST" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <output_dir> <TEST07|TEST09>"
    exit 1
fi

# Normalize test name to uppercase
TEST=$(echo "$TEST" | tr '[:lower:]' '[:upper:]')

# Resolve paths relative to script directory
# Script is in harness/scripts/, so:
# - compliance is at ../../compliance/ (repo root level)
# - language is at ../language/ (harness level)
COMPLIANCE_DIR="${SCRIPT_DIR}/../../compliance"
LANGUAGE_DIR="${SCRIPT_DIR}/../../language"

# Check if TEST is TEST07 or TEST09
if [ "$TEST" = "TEST07" ]; then
    # TEST07 requires --accuracy-script
    python3 "${COMPLIANCE_DIR}/${TEST}/run_verification.py" -c "${DIR}/mlperf/" -o "${DIR}" \
        --audit-config "${COMPLIANCE_DIR}/${TEST}/gpt-oss-120b/audit.config" \
        --accuracy-script "python3 ${LANGUAGE_DIR}/gpt-oss-120b/eval_mlperf_accuracy.py \
            --mlperf-log ${DIR}/mlperf/mlperf_log_accuracy.json \
            --reference-data ${DATASET_DIR}/acc/acc_eval_compliance_gpqa.parquet \
            --tokenizer openai/gpt-oss-120b"
elif [ "$TEST" = "TEST09" ]; then
    # TEST09 does not require --accuracy-script
    python3 "${COMPLIANCE_DIR}/${TEST}/run_verification.py" -c "${DIR}/mlperf/" -o "${DIR}" \
        --audit-config "${COMPLIANCE_DIR}/${TEST}/gpt-oss-120b/audit.config"
else
    echo "Error: Invalid test name: $TEST"
    echo "       Must be TEST07 or TEST09"
    exit 1
fi
