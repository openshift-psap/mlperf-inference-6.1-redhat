#!/bin/bash
# Update EPP configuration on a running llm-d deployment
#
# Usage:
#   ./update_epp_config.sh <config-file> [--log-level N]
#
# Examples:
#   ./update_epp_config.sh epp-configs/mlperf-v60.yaml
#   ./update_epp_config.sh epp-configs/token-load.yaml --log-level 3
#   ./update_epp_config.sh epp-configs/active-request.yaml --log-level 7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-llm-d-bench}"
GUIDE="${GUIDE:-optimized-baseline}"
LOG_LEVEL=""

# Parse arguments
CONFIG_FILE="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-level) LOG_LEVEL="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: $0 <config-file> [--log-level N]"
    echo ""
    echo "Available configs:"
    ls -1 "${SCRIPT_DIR}/epp-configs/"
    exit 1
fi

# Resolve relative path
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_FILE}"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Updating EPP config: $(basename "$CONFIG_FILE")"

# Patch configmap with new config
CONFIG_CONTENT=$(cat "$CONFIG_FILE")
kubectl patch configmap "${GUIDE}-epp" -n "$NAMESPACE" --type=merge \
  -p "{\"data\":{\"custom-plugins.yaml\":$(echo "$CONFIG_CONTENT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}}"

# Update log level if specified
if [[ -n "$LOG_LEVEL" ]]; then
    echo "Setting EPP log level to ${LOG_LEVEL}"
    # Find the epp container index (not envoy-proxy)
    EPP_IDX=$(kubectl get deployment "${GUIDE}-epp" -n "$NAMESPACE" -o jsonpath='{range .spec.template.spec.containers[*]}{.name}{"\n"}{end}' | grep -n '^epp$' | cut -d: -f1)
    EPP_IDX=$((EPP_IDX - 1))  # 0-based

    ARGS=$(kubectl get deployment "${GUIDE}-epp" -n "$NAMESPACE" -o jsonpath="{.spec.template.spec.containers[${EPP_IDX}].args}")
    # Find the index of --v=N arg in epp container, replace if exists, add if not
    ARG_IDX=$(echo "$ARGS" | python3 -c "
import sys,json
args=json.load(sys.stdin)
matches=[i for i,a in enumerate(args) if a.startswith('--v=')]
print(matches[0] if matches else -1)
")
    if [[ "$ARG_IDX" != "-1" ]]; then
      kubectl patch deployment "${GUIDE}-epp" -n "$NAMESPACE" --type=json \
        -p="[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/${EPP_IDX}/args/${ARG_IDX}\",\"value\":\"--v=${LOG_LEVEL}\"}]"
    else
      kubectl patch deployment "${GUIDE}-epp" -n "$NAMESPACE" --type=json \
        -p="[{\"op\":\"add\",\"path\":\"/spec/template/spec/containers/${EPP_IDX}/args/-\",\"value\":\"--v=${LOG_LEVEL}\"}]"
    fi
fi

# Scale down, delete old RS, scale back up
kubectl scale deployment/"${GUIDE}-epp" -n "$NAMESPACE" --replicas=0
sleep 3
kubectl delete rs -n "$NAMESPACE" $(kubectl get rs -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" | grep "${GUIDE}-epp") --ignore-not-found 2>/dev/null || true
kubectl scale deployment/"${GUIDE}-epp" -n "$NAMESPACE" --replicas=1
kubectl rollout status deployment/"${GUIDE}-epp" -n "$NAMESPACE" --timeout=60s

# Fix envoy ulimits (reset after pod restart)
NODE_NAME=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [[ -n "$NODE_NAME" ]]; then
    echo "Fixing envoy file descriptor limits..."
    sleep 3  # Wait for envoy to fully start
    oc debug "node/$NODE_NAME" -- chroot /host bash -c "
      for PID in \$(pgrep -f envoy 2>/dev/null); do
        prlimit --pid \$PID --nofile=65536:65536 2>/dev/null
      done
    " 2>/dev/null
    echo "Envoy ulimits set to 65536"
fi

echo ""
echo "EPP config updated successfully"
echo "Active config: $(basename "$CONFIG_FILE")"
if [[ -n "$LOG_LEVEL" ]]; then
    echo "Log level: ${LOG_LEVEL}"
fi
