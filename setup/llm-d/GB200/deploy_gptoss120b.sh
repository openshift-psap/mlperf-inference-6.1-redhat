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
#   - llm-d with Istio gateway
#   - OpenShift with LVM operator for PVC storage
#
# Usage:
#   ./deploy_gptoss120b.sh server              # Deploy with server config
#   ./deploy_gptoss120b.sh offline             # Deploy with offline config
#   ./deploy_gptoss120b.sh server --cleanup    # Cleanup deployment
#   ./deploy_gptoss120b.sh server --dry-run    # Dry-run (print manifests)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-v0.24.0}"

MODE="${1:-}"
EXTRA_FLAG="${2:-}"

if [[ "$MODE" != "server" && "$MODE" != "offline" ]]; then
    echo "Usage: $0 [server|offline] [--cleanup|--dry-run]"
    exit 1
fi

OVERRIDE="${SCRIPT_DIR}/override_gptoss120b_${MODE}.yaml"
[[ -f "$OVERRIDE" ]] || { echo "ERROR: Override not found: $OVERRIDE"; exit 1; }

echo "=========================================="
echo "GPT-OSS-120B on GB200 NVL4 — ${MODE} mode"
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
echo "Gateway URL:  http://llm-d-inference-gateway.llm-d-bench.svc.cluster.local:80"
echo "Cleanup:      $0 ${MODE} --cleanup"
