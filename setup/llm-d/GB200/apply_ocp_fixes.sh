#!/usr/bin/env bash
# =============================================================================
# post-deploy.sh — Fixes and patches required after deploy.sh on OpenShift
# =============================================================================
#
# Run this AFTER deploy.sh completes to apply OpenShift-specific fixes:
#   1. SCC (Security Context Constraints) for all service accounts
#   2. SELinux relabeling for hostPath volumes
#   3. File descriptor limits on gateway processes
#   4. MLPerf harness pod deployment
#   5. HTTPRoute timeout increase for LLM inference
#
# Usage:
#   ./post-deploy.sh -o <override.yaml> [OPTIONS]
#   ./post-deploy.sh -o <override.yaml> --cleanup
#   ./post-deploy.sh -o <override.yaml> --harness-only  # just deploy harness pod
#   ./post-deploy.sh -o <override.yaml> --fix-ulimits   # just fix file descriptors
#
# Prerequisites:
#   - deploy.sh has been run successfully
#   - oc CLI configured with cluster access
#   - yq installed
# =============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m OK:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- Argument parsing ---
OVERRIDE_FILE=""
CLEANUP=false
FIX_ULIMITS=false
NODE_NAME=""
VLLM_IMAGE_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--override)      OVERRIDE_FILE="$2"; shift 2;;
    --cleanup)          CLEANUP=true; shift;;
    --fix-ulimits)      FIX_ULIMITS=true; shift;;
    --node)             NODE_NAME="$2"; shift 2;;
    --vllm-image)       VLLM_IMAGE_TAG="$2"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") -o <override.yaml> [OPTIONS]

Post-deployment fixes for llm-d on OpenShift.

Options:
  -o, --override FILE    Override YAML file (same one used with deploy.sh)
  --cleanup              Revert SCC bindings
  --fix-ulimits          Only fix file descriptor limits on gateway
  --node NAME            Node name (auto-detected if omitted)
  --vllm-image TAG       Override vLLM model server image tag (e.g., v0.24.0)
  -h, --help             Show this help
EOF
      exit 0;;
    *) die "Unknown argument: $1";;
  esac
done

[[ -n "$OVERRIDE_FILE" ]] || die "Override file required (-o)"
[[ -f "$OVERRIDE_FILE" ]] || die "Override file not found: $OVERRIDE_FILE"

# --- Read config from override ---
yqr() { yq -r "$1" "$OVERRIDE_FILE"; }
yqd() { yq -r "$1 // \"$2\"" "$OVERRIDE_FILE"; }

NAMESPACE=$(yqr '.namespace')
GUIDE=$(yqr '.guide')
MODEL_SOURCE=$(yqd '.model.source' 'huggingface')
HOST_PATH=$(yqd '.model.host_path' '')
PVC_NAME=$(yqd '.model.pvc_name' '')

[[ "$NAMESPACE" != "null" && -n "$NAMESPACE" ]] || die "namespace is required in override file"

# Auto-detect node name
if [[ -z "$NODE_NAME" ]]; then
  NODE_NAME=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
fi

log "Post-deploy configuration:"
log "  Namespace:  $NAMESPACE"
log "  Guide:      $GUIDE"
log "  Node:       $NODE_NAME"

# --- Cleanup ---
if $CLEANUP; then
  log "Cleaning up post-deploy resources (keeping namespace and harness pod)..."

  log "Removing SCC bindings..."
  oc adm policy remove-scc-from-user privileged -z "${GUIDE}-nvidia-gpu-vllm-sa" -n "$NAMESPACE" 2>/dev/null || true
  oc adm policy remove-scc-from-user anyuid -z llm-d-inference-gateway -n "$NAMESPACE" 2>/dev/null || true
  oc adm policy remove-scc-from-user privileged -z llm-d-inference-gateway -n "$NAMESPACE" 2>/dev/null || true

  ok "Cleanup complete (namespace '$NAMESPACE' and mlperf-harness pod preserved)"
  exit 0
fi

# =============================================================================
# Fix 1: SCC + vLLM Image Override
# =============================================================================
fix_scc_and_image() {
  log "Fixing Security Context Constraints..."

  # Model server SA needs privileged (for hostPath/PVC + runAsUser:0)
  local ms_sa="${GUIDE}-nvidia-gpu-vllm-sa"
  oc adm policy add-scc-to-user privileged -z "$ms_sa" -n "$NAMESPACE" 2>/dev/null
  ok "Granted privileged SCC to $ms_sa"

  # Gateway SA needs anyuid (runs as UID 10101)
  oc adm policy add-scc-to-user anyuid -z llm-d-inference-gateway -n "$NAMESPACE" 2>/dev/null
  ok "Granted anyuid SCC to llm-d-inference-gateway"

  # Scale down deployments
  local deploy_name="${GUIDE}-nvidia-gpu-vllm-decode"
  log "Scaling down deployments..."
  kubectl scale deployment "$deploy_name" -n "$NAMESPACE" --replicas=0 2>/dev/null
  kubectl scale deployment llm-d-inference-gateway -n "$NAMESPACE" --replicas=0 2>/dev/null
  sleep 5

  # Override vLLM image if requested
  if [[ -n "$VLLM_IMAGE_TAG" ]]; then
    log "Overriding vLLM image to vllm/vllm-openai:${VLLM_IMAGE_TAG}..."
    kubectl set image deployment/"$deploy_name" \
      "modelserver=vllm/vllm-openai:${VLLM_IMAGE_TAG}" \
      -n "$NAMESPACE" 2>/dev/null
    ok "vLLM image set to vllm/vllm-openai:${VLLM_IMAGE_TAG}"
  fi

  # Set TIKTOKEN env vars on model server
  local mount_path
  mount_path=$(yqd '.model.mount_path' '/mnt/models')
  local tiktoken_base="${mount_path}/gpt-oss-encoding"
  log "Setting model server env vars..."
  kubectl set env deployment/"$deploy_name" -n "$NAMESPACE" \
    "TIKTOKEN_ENCODINGS_BASE=${tiktoken_base}" \
    "TIKTOKEN_RS_CACHE_DIR=${tiktoken_base}" \
    "HF_HUB_OFFLINE=1" 2>/dev/null
  ok "TIKTOKEN and HF_HUB_OFFLINE env vars set"

  # Delete stale ReplicaSets before scaling up
  log "Deleting old ReplicaSets..."
  kubectl delete rs -n "$NAMESPACE" $(kubectl get rs -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" | grep "$deploy_name") --ignore-not-found 2>/dev/null || true

  # Scale back up
  local replicas
  replicas=$(yqd '.modelserver.replicas' '4')
  kubectl scale deployment "$deploy_name" -n "$NAMESPACE" --replicas="$replicas" 2>/dev/null
  kubectl scale deployment llm-d-inference-gateway -n "$NAMESPACE" --replicas=1 2>/dev/null
  ok "Deployments restarted with SCC applied"
}

# =============================================================================
# Fix 3: SELinux Relabeling for hostPath
# =============================================================================
fix_selinux() {
  if [[ "$MODEL_SOURCE" != "hostpath" || -z "$HOST_PATH" ]]; then
    log "Skipping SELinux fix (not using hostpath)"
    return
  fi

  log "Fixing SELinux labels on $HOST_PATH..."
  oc debug "node/$NODE_NAME" -- chroot /host chcon -R -t svirt_sandbox_file_t "$HOST_PATH" 2>/dev/null
  ok "SELinux labels set to svirt_sandbox_file_t"
}

# =============================================================================
# Fix 3: File Descriptor Limits on Gateway
# =============================================================================
fix_ulimits() {
  log "Fixing file descriptor limits on gateway processes..."

  # Wait for gateway pod to be running
  local attempts=0
  while [[ $attempts -lt 30 ]]; do
    local ready
    ready=$(kubectl get pods -n "$NAMESPACE" -l gateway.networking.k8s.io/gateway-name=llm-d-inference-gateway \
      -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null)
    if [[ "$ready" == "true" ]]; then
      break
    fi
    sleep 10
    ((attempts++))
  done

  if [[ $attempts -ge 30 ]]; then
    warn "Gateway pod not ready after 5 minutes. Skipping ulimit fix."
    warn "Run './post-deploy.sh -o $OVERRIDE_FILE --fix-ulimits' later."
    return
  fi

  # Apply prlimit to all agentgateway processes
  oc debug "node/$NODE_NAME" --  chroot /host bash -c "
    for PID in \$(pgrep -f agentgateway 2>/dev/null); do
      prlimit --pid \$PID --nofile=65536:65536 2>/dev/null
    done
    echo 'Fixed \$(pgrep -f agentgateway 2>/dev/null | wc -l) agentgateway processes'
  " 2>/dev/null
  ok "Gateway file descriptor limits set to 65536"
}

# =============================================================================
# Fix 4: HTTPRoute Timeout
# =============================================================================
fix_timeout() {
  log "Setting HTTPRoute timeout to 7200s (2 hours)..."

  kubectl patch httproute "$GUIDE" -n "$NAMESPACE" --type=merge -p '{
    "spec": {
      "rules": [{
        "backendRefs": [{"group":"inference.networking.k8s.io","kind":"InferencePool","name":"'"$GUIDE"'"}],
        "matches": [{"path":{"type":"PathPrefix","value":"/"}}],
        "timeouts": {"request":"7200s"}
      }]
    }
  }' 2>/dev/null
  ok "HTTPRoute timeout set to 7200s"
}

# =============================================================================
# Wait for vLLM pods
# =============================================================================
wait_for_vllm() {
  log "Waiting for vLLM model server pods to be ready..."

  local replicas
  replicas=$(yqd '.modelserver.replicas' '4')
  local timeout=900  # 15 min (model load + autotune)
  local elapsed=0

  while [[ $elapsed -lt $timeout ]]; do
    local ready
    ready=$(kubectl get deployment "${GUIDE}-nvidia-gpu-vllm-decode" -n "$NAMESPACE" \
      -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    ready=${ready:-0}

    if [[ "$ready" -ge "$replicas" ]]; then
      ok "All $replicas vLLM pods are ready"
      return 0
    fi

    if (( elapsed % 60 == 0 )); then
      log "  $ready/$replicas pods ready (${elapsed}s elapsed, model loading...)"
    fi
    sleep 15
    ((elapsed += 15))
  done

  warn "Only $ready/$replicas vLLM pods ready after ${timeout}s"
  warn "Continuing — some pods may still be starting"
}

# =============================================================================
# Verify end-to-end connectivity
# =============================================================================
verify() {
  log "Verifying end-to-end connectivity..."

  # Find any running pod in namespace to exec curl from
  local test_pod
  test_pod=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' 2>/dev/null | awk '{print $1}')

  if [[ -z "$test_pod" ]]; then
    warn "No running pod found to verify connectivity. Skipping."
    return
  fi

  local result
  result=$(kubectl exec "$test_pod" -n "$NAMESPACE" -- curl -s --connect-timeout 30 \
    "http://llm-d-inference-gateway.${NAMESPACE}.svc.cluster.local:80/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"openai/gpt-oss-120b","prompt":"Hello","max_tokens":3}' 2>/dev/null)

  if echo "$result" | grep -q '"choices"'; then
    ok "End-to-end inference working!"
  else
    warn "Inference test failed. Response: $result"
    warn "Check: oc logs -n $NAMESPACE -l llm-d.ai/model=gpt-oss-120b --tail=10"
  fi
}

# =============================================================================
# Main
# =============================================================================

if $FIX_ULIMITS; then
  fix_ulimits
  exit 0
fi

log "=========================================="
log "Running all post-deploy fixes"
log "=========================================="
echo ""

fix_selinux
echo ""

fix_scc_and_image
echo ""

log "Waiting for pods to start after SCC fix..."
sleep 30

fix_timeout
echo ""

wait_for_vllm
echo ""

fix_ulimits
echo ""

verify

echo ""
log "=========================================="
log "Post-deploy complete!"
log "=========================================="
echo ""
log "Summary of fixes applied:"
log "  1. SCC: privileged for model server SA, anyuid for gateway SA"
log "  2. vLLM image + TIKTOKEN/HF env vars"
log "  3. SELinux: svirt_sandbox_file_t on hostPath (if applicable)"
log "  4. HTTPRoute timeout: 7200s"
log "  5. File descriptors: 65536 on gateway processes (temporary)"
echo ""
log "NOTE: The file descriptor fix is temporary."
log "If the gateway pod restarts, re-run:"
log "  ./apply_ocp_fixes.sh -o $OVERRIDE_FILE --fix-ulimits"
echo ""
log "Next: deploy the client pod (see setup/client/GB200/)"
