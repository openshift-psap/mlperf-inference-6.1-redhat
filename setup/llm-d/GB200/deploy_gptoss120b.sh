#!/bin/bash

# Deploy GPT-OSS-120B on GB200 NVL4 with llm-d
#
# Hardware: NVIDIA GB200 NVL4
#   - 4x GB200 GPUs (189GB HBM3e each)
#   - 2x Grace CPUs (72 ARM cores each, 144 total)
#   - NVLink 5.0 (1.8TB/s GPU-GPU), NVLink-C2C (900GB/s CPU-GPU)
#
# Software:
#   - vLLM 0.24.0 with FlashInfer 0.6.12
#   - llm-d (standalone or Istio gateway mode)
#   - OpenShift with LVM operator for PVC storage
#
# Usage:
#   ./deploy_gptoss120b.sh server                      # Deploy gateway mode
#   ./deploy_gptoss120b.sh server --standalone         # Deploy standalone mode
#   ./deploy_gptoss120b.sh offline                     # Deploy offline config
#   ./deploy_gptoss120b.sh server --cleanup            # Cleanup deployment
#   ./deploy_gptoss120b.sh server --standalone --cleanup
#   ./deploy_gptoss120b.sh server --dry-run            # Dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-v0.24.0}"

MODE=""
STANDALONE=false
EXTRA_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        server|offline) MODE="$1"; shift;;
        --standalone)   STANDALONE=true; shift;;
        --cleanup|--dry-run) EXTRA_FLAG="$1"; shift;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Usage: $0 [server|offline] [--standalone] [--cleanup|--dry-run]"
    exit 1
fi

if $STANDALONE; then
    OVERRIDE="${SCRIPT_DIR}/override_gptoss120b_${MODE}_standalone.yaml"
else
    OVERRIDE="${SCRIPT_DIR}/override_gptoss120b_${MODE}.yaml"
fi
[[ -f "$OVERRIDE" ]] || { echo "ERROR: Override not found: $OVERRIDE"; exit 1; }

DEPLOY_MODE="gateway"
$STANDALONE && DEPLOY_MODE="standalone"

echo "=========================================="
echo "GPT-OSS-120B on GB200 NVL4 — ${MODE} (${DEPLOY_MODE})"
echo "=========================================="

if [[ "$EXTRA_FLAG" == "--cleanup" ]]; then
    "${SCRIPT_DIR}/install_llmd.sh" -o "$OVERRIDE" --cleanup
    exit 0
fi

if [[ "$EXTRA_FLAG" == "--dry-run" ]]; then
    "${SCRIPT_DIR}/install_llmd.sh" -o "$OVERRIDE" --dry-run
    exit 0
fi

# Step 1: Deploy llm-d (Istio + router + model servers)
"${SCRIPT_DIR}/install_llmd.sh" -o "$OVERRIDE"

# Step 2: Apply OpenShift fixes (SCC, vLLM image, env vars, ulimits)
"${SCRIPT_DIR}/apply_ocp_fixes.sh" -o "$OVERRIDE" --vllm-image "$VLLM_IMAGE_TAG"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next: deploy the client pod to run MLPerf benchmarks"
echo "  oc apply -f setup/client/GB200/client-pod.yaml"
echo "  oc exec -it mlperf-client -n llm-d-bench -- bash client_setup.sh"
echo ""
echo "API URL:      http://vllm-direct.llm-d-bench.svc.cluster.local:8000"
# ROUTER_MODE=$(yq -r '.router.mode // "standalone"' "$OVERRIDE")
# if [[ "$ROUTER_MODE" == "standalone" ]]; then
#     GUIDE=$(yq -r '.guide' "$OVERRIDE")
#     echo "EPP URL:      http://${GUIDE}-epp.llm-d-bench.svc.cluster.local:80"
# else
#     echo "Gateway URL:  http://llm-d-inference-gateway.llm-d-bench.svc.cluster.local:80"
# fi
if $STANDALONE; then
    echo "Cleanup:      $0 ${MODE} --standalone --cleanup"
else
    echo "Cleanup:      $0 ${MODE} --cleanup"
fi
