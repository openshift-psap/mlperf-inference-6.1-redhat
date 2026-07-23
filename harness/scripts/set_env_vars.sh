#!/bin/bash
#
# Environment Variables Setup Script for MLPerf Inference Harness
# ===============================================================
# This script sets up the required environment variables for running
# MLPerf inference harness tests using run_submission.py
#
# Usage:
#   source set_env_vars.sh
#   # or
#   . set_env_vars.sh
#
# To customize values, edit this file or export variables before sourcing:
#   export DATASET_DIR=/path/to/datasets
#   source set_env_vars.sh

# Required Environment Variables
# ------------------------------

# Dataset configuration
export DATASET_DIR="${DATASET_DIR:-}"
export PERF_DATASET="${PERF_DATASET:-}"
export ACC_DATASET="${ACC_DATASET:-}"
export COMPLIANCE_DATASET="${COMPLIANCE_DATASET:-}"

# Output directory
export OUTPUT_DIR="${OUTPUT_DIR:-./harness_output}"

# API Server configuration
export API_SERVER_URL="${API_SERVER_URL:-}"

# AWS credentials (if needed)
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"

# MLflow configuration
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-}"
export MLFLOW_USER_TAG="${MLFLOW_USER_TAG:-}"

# HuggingFace configuration
export HF_HOME="${HF_HOME:-}"

# Model configuration
export MODEL_CATEGORY="${MODEL_CATEGORY:-gpt-oss-120b}"
export MODEL="${MODEL:-openai/gpt-oss-120b}"
export BACKEND="${BACKEND:-vllm}"
export LG_MODEL_NAME="${LG_MODEL_NAME:-gpt-oss-120b}"

# Scenario configuration
export SCENARIO="${SCENARIO:-Server}"
export SERVER_TARGET_QPS="${SERVER_TARGET_QPS:-3}"

# Compliance test configuration
export COMPLIANCE_TEST="${COMPLIANCE_TEST:-TEST07}"
export AUDIT_CONFIG_SRC="${AUDIT_CONFIG_SRC:-}"
export AUDIT_OVERRIDE_CONF="${AUDIT_OVERRIDE_CONF:-audit-override.cfg}"

# User configuration
export USER_CONF="${USER_CONF:-}"

# Helper function to validate required variables
validate_env_vars() {
    local missing_vars=()
    
    if [ -z "$DATASET_DIR" ]; then
        missing_vars+=("DATASET_DIR")
    fi
    
    if [ -z "$API_SERVER_URL" ]; then
        missing_vars+=("API_SERVER_URL")
    fi
    
    if [ -z "$AWS_ACCESS_KEY_ID" ]; then
        missing_vars+=("AWS_ACCESS_KEY_ID")
    fi
    
    if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        missing_vars+=("AWS_SECRET_ACCESS_KEY")
    fi
    
    if [ -z "$MLFLOW_TRACKING_URI" ]; then
        missing_vars+=("MLFLOW_TRACKING_URI")
    fi
    
    if [ -z "$MLFLOW_EXPERIMENT_NAME" ]; then
        missing_vars+=("MLFLOW_EXPERIMENT_NAME")
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "ERROR: The following required environment variables are not set:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Please set them before running the harness tests."
        return 1
    fi
    
    return 0
}

# Print current configuration
print_env_vars() {
    echo "=========================================="
    echo "Current Environment Variables:"
    echo "=========================================="
    echo "  DATASET_DIR: ${DATASET_DIR:-<not set>}"
    echo "  PERF_DATASET: ${PERF_DATASET:-<not set>}"
    echo "  ACC_DATASET: ${ACC_DATASET:-<not set>}"
    echo "  COMPLIANCE_DATASET: ${COMPLIANCE_DATASET:-<not set>}"
    echo "  OUTPUT_DIR: ${OUTPUT_DIR:-<not set>}"
    echo "  API_SERVER_URL: ${API_SERVER_URL:-<not set>}"
    echo "  AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:-<not set>}"
    echo "  AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:+<set>}"
    echo "  MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI:-<not set>}"
    echo "  MLFLOW_EXPERIMENT_NAME: ${MLFLOW_EXPERIMENT_NAME:-<not set>}"
    echo "  MLFLOW_USER_TAG: ${MLFLOW_USER_TAG:-<not set>}"
    echo "  HF_HOME: ${HF_HOME:-<not set>}"
    echo "  MODEL_CATEGORY: ${MODEL_CATEGORY:-<not set>}"
    echo "  MODEL: ${MODEL:-<not set>}"
    echo "  BACKEND: ${BACKEND:-<not set>}"
    echo "  LG_MODEL_NAME: ${LG_MODEL_NAME:-<not set>}"
    echo "  SCENARIO: ${SCENARIO:-<not set>}"
    echo "  SERVER_TARGET_QPS: ${SERVER_TARGET_QPS:-<not set>}"
    echo "  COMPLIANCE_TEST: ${COMPLIANCE_TEST:-<not set>}"
    echo "  AUDIT_CONFIG_SRC: ${AUDIT_CONFIG_SRC:-<not set>}"
    echo "  AUDIT_OVERRIDE_CONF: ${AUDIT_OVERRIDE_CONF:-<not set>}"
    echo "  USER_CONF: ${USER_CONF:-<not set>}"
    echo "=========================================="
}

# If script is sourced (not executed), just set the variables
# If script is executed, print the configuration
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # Script is being sourced
    echo "Environment variables have been set."
    echo "Use 'print_env_vars' to see current values."
    echo "Use 'validate_env_vars' to check if required variables are set."
else
    # Script is being executed
    print_env_vars
    echo ""
    if validate_env_vars; then
        echo "✓ All required environment variables are set."
    else
        echo "✗ Some required environment variables are missing."
        exit 1
    fi
fi
