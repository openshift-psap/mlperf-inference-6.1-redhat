#!/bin/bash
# Client setup for GPT-OSS-120B on GB200 NVL4
# Run this inside the mlperf-client pod after creation
#
# Prerequisites:
#   1. Client pod created: oc apply -f client-pod.yaml
#   2. SCC granted: oc adm policy add-scc-to-user privileged -z default -n llm-d-bench
#   3. PVC mounted with model weights at /mnt/models
#
# Usage:
#   oc exec -it mlperf-client -n llm-d-bench -- bash
#   bash client_setup.sh

set -e

echo "=========================================="
echo "MLPerf Client Setup — GPT-OSS-120B GB200"
echo "=========================================="

# Install system dependencies
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git build-essential curl > /dev/null 2>&1

# Install uv for fast Python package management
pip install -q uv

# Clone the 6.1 inference repo
echo "Cloning mlperf-inference-6.1-redhat..."
git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.1-redhat.git

# Create and activate virtual environment
uv venv -p 3.12 gptoss_harness
source gptoss_harness/bin/activate

# Install GPT-OSS-120B benchmark dependencies
cd mlperf-inference-6.1-redhat/language/gpt-oss-120b
uv pip install pip
./setup.sh

# Install harness dependencies
cd ../../harness
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  source /gptoss_harness/bin/activate"
echo ""
echo "Set environment variables before running tests:"
echo "  export API_SERVER_URL=http://llm-d-inference-gateway.llm-d-bench.svc.cluster.local:80"
echo "  export DATASET_DIR=/mnt/models/datasets/gpt-oss_data/"
echo "  export HF_HOME=/mnt/models"
echo "  export TIKTOKEN_ENCODINGS_BASE=/mnt/models/gpt-oss-encoding"
echo "  export TIKTOKEN_RS_CACHE_DIR=/mnt/models/gpt-oss-encoding"
echo "  ulimit -n 65536"
