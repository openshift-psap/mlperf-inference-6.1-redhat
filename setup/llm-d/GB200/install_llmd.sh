#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") -o <override.yaml> [OPTIONS]

Deploy an llm-d guide with custom overrides.

Options:
  -o, --override FILE    Override YAML file (required)
  -d, --dry-run          Print manifests without deploying
  -c, --clone-dir DIR    Directory to clone llm-d into (default: /tmp/llm-d-<version>)
  --cleanup          Tear down the deployment instead of deploying
  -h, --help             Show this help
EOF
}

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

check_prereqs() {
  local missing=()
  for cmd in yq jq kubectl helm kustomize git curl; do
    command -v "$cmd" &>/dev/null || missing+=("$cmd")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    die "Missing required tools: ${missing[*]}"
  fi
}

# --- Argument parsing ---
OVERRIDE_FILE=""
DRY_RUN=false
CLONE_DIR=""
CLEANUP=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--override)  OVERRIDE_FILE="$2"; shift 2;;
    -d|--dry-run)   DRY_RUN=true; shift;;
    -c|--clone-dir) CLONE_DIR="$2"; shift 2;;
    --cleanup)      CLEANUP=true; shift;;
    -h|--help)      usage; exit 0;;
    *) die "Unknown argument: $1";;
  esac
done

[[ -n "$OVERRIDE_FILE" ]] || { usage; die "Override file is required (-o)"; }
[[ -f "$OVERRIDE_FILE" ]] || die "Override file not found: $OVERRIDE_FILE"

check_prereqs

# --- Read override values ---
yqr() { yq -r "$1" "$OVERRIDE_FILE"; }
yqd() { yq -r "$1 // \"$2\"" "$OVERRIDE_FILE"; }

GUIDE=$(yqr '.guide')
NAMESPACE=$(yqr '.namespace')
VERSION=$(yqd '.version' 'main')

MODEL_NAME=$(yqr '.model.name')
MODEL_SOURCE=$(yqd '.model.source' 'huggingface')
PVC_NAME=$(yqd '.model.pvc_name' '')
HOST_PATH=$(yqd '.model.host_path' '')
MODEL_MOUNT_PATH=$(yqd '.model.mount_path' '/models')

ACCELERATOR=$(yqd '.modelserver.accelerator' 'gpu')
ENGINE=$(yqd '.modelserver.engine' 'vllm')
INFRA_PROVIDER=$(yqd '.modelserver.infra_provider' 'base')
REPLICAS=$(yqd '.modelserver.replicas' '8')
TP=$(yqd '.modelserver.tensor_parallel_size' '2')
GPU_COUNT="$TP"
CPU_LIMIT=$(yqd '.modelserver.cpu_limit' '16')
CPU_REQUEST=$(yqd '.modelserver.cpu_request' '8')
MEMORY_LIMIT=$(yqd '.modelserver.memory_limit' '')
MEMORY_REQUEST=$(yqd '.modelserver.memory_request' '')
SHM_SIZE=$(yqd '.modelserver.shm_size' '20Gi')

EPP_CONFIG=$(yqd '.router.epp_config' '')
ROUTER_MODE=$(yqd '.router.mode' 'standalone')

# Read vllm_args as a bash array
VLLM_ARGS=()
while IFS= read -r arg; do
  [[ -n "$arg" && "$arg" != "null" ]] && VLLM_ARGS+=("$arg")
done < <(yq -r '.modelserver.vllm_args // [] | .[]' "$OVERRIDE_FILE" 2>/dev/null || true)

[[ "$GUIDE" != "null" && -n "$GUIDE" ]]     || die "guide is required in override file"
[[ "$NAMESPACE" != "null" && -n "$NAMESPACE" ]] || die "namespace is required in override file"
[[ "$MODEL_NAME" != "null" && -n "$MODEL_NAME" ]] || die "model.name is required in override file"

case "$ROUTER_MODE" in
  standalone|gateway) ;;
  *) die "router.mode must be standalone or gateway (got: ${ROUTER_MODE})" ;;
esac

# Gateway mode is fully self-contained. Istio is the supported default
# inference Gateway and the upstream recipe fixes this name.
GATEWAY_PROVIDER="istio"
GATEWAY_NAME="llm-d-inference-gateway"

MODEL_SHORT="${MODEL_NAME##*/}"

# Resolve EPP config path relative to the override file's directory if not absolute
if [[ -n "$EPP_CONFIG" && "$EPP_CONFIG" != "null" ]]; then
  if [[ "$EPP_CONFIG" != /* ]]; then
    EPP_CONFIG="$(cd "$(dirname "$OVERRIDE_FILE")" && realpath "$EPP_CONFIG" 2>/dev/null || echo "${SCRIPT_DIR}/${EPP_CONFIG}")"
  fi
  [[ -f "$EPP_CONFIG" ]] || die "EPP config file not found: $EPP_CONFIG"
fi

log "Configuration:"
log "  Guide:      $GUIDE"
log "  Namespace:  $NAMESPACE"
log "  Version:    $VERSION"
log "  Model:      $MODEL_NAME ($MODEL_SOURCE)"
log "  Replicas:   $REPLICAS  TP: $TP  Total GPUs: $((REPLICAS * TP))"
log "  Router:     $ROUTER_MODE"
if [[ "$ROUTER_MODE" == "gateway" ]]; then
  log "  Gateway:    $GATEWAY_NAME (automatically provisioned with $GATEWAY_PROVIDER)"
fi
if [[ -n "$EPP_CONFIG" && "$EPP_CONFIG" != "null" ]]; then
  log "  EPP config: $EPP_CONFIG"
fi
if $DRY_RUN; then
  log "  Mode:       DRY-RUN"
fi

# --- Clone repo ---
CLONE_DIR="${CLONE_DIR:-/tmp/llm-d-${VERSION}}"

if [[ ! -d "$CLONE_DIR/guides" ]]; then
  log "Cloning llm-d (${VERSION}) into ${CLONE_DIR}..."
  git clone --branch "$VERSION" --depth 1 https://github.com/llm-d/llm-d.git "$CLONE_DIR"
else
  log "Using existing clone at ${CLONE_DIR}"
fi

# Source env.sh from the cloned repo
REPO_ROOT="$CLONE_DIR"
export REPO_ROOT
# shellcheck source=/dev/null
source "${REPO_ROOT}/guides/env.sh"

GATEWAY_RECIPE_DIR="${REPO_ROOT}/guides/recipes/gateway/istio"

if [[ "$ROUTER_MODE" == "standalone" ]]; then
  ROUTER_CHART="$ROUTER_STANDALONE_CHART"
  ROUTER_CHART_NAME="llm-d-router-standalone"
else
  ROUTER_CHART="$ROUTER_GATEWAY_CHART"
  ROUTER_CHART_NAME="llm-d-router-gateway"
fi

# --- Locate guide modelserver path ---
MODELSERVER_PATH="${REPO_ROOT}/guides/${GUIDE}/modelserver/${ACCELERATOR}/${ENGINE}/${INFRA_PROVIDER}"
if [[ ! -d "$MODELSERVER_PATH" ]]; then
  MODELSERVER_PATH="${REPO_ROOT}/guides/${GUIDE}/modelserver/${ACCELERATOR}/${ENGINE}"
  [[ -d "$MODELSERVER_PATH" ]] || die "Modelserver path not found for guide=${GUIDE} accelerator=${ACCELERATOR} engine=${ENGINE} provider=${INFRA_PROVIDER}"
fi
[[ -f "${MODELSERVER_PATH}/kustomization.yaml" ]] || die "No kustomization.yaml in ${MODELSERVER_PATH}"

# --- Locate guide router values ---
ROUTER_BASE_VALUES="${REPO_ROOT}/guides/recipes/router/base.values.yaml"
GUIDE_ROUTER_DIR="${REPO_ROOT}/guides/${GUIDE}/router"

GUIDE_VALUES=""
if [[ -d "$GUIDE_ROUTER_DIR" ]]; then
  for candidate in "${GUIDE_ROUTER_DIR}/${GUIDE}.values.yaml" "${GUIDE_ROUTER_DIR}"/*.values.yaml; do
    if [[ -f "$candidate" ]]; then
      GUIDE_VALUES="$candidate"
      break
    fi
  done
fi

# --- GPU resource key ---
gpu_resource_key() {
  case "$ACCELERATOR" in
    gpu)      echo "nvidia.com/gpu";;
    amd)      echo "amd.com/gpu";;
    xpu)      echo "intel.com/gpu";;
    tpu/*)    echo "google.com/tpu";;
    *)        echo "";;
  esac
}

GPU_KEY=$(gpu_resource_key)

# --- Merge args: guide base + overrides (model, TP, served-model-name, vllm_args) ---
build_merged_args() {
  local rendered="$1"
  local args_file="$2"

  local model_arg="$MODEL_NAME"
  # For both pvc and hostpath, keep HF model name — vLLM resolves via HF_HOME

  # Collect override flags: TP first, then user vllm_args
  local override_args=("--tensor-parallel-size=${TP}")
  for arg in "${VLLM_ARGS[@]+"${VLLM_ARGS[@]}"}"; do
    override_args+=("$arg")
  done

  # Extract guide args from rendered Deployment
  local guide_args_file="${WORK_DIR}/guide-args.json"
  yq 'select(.kind == "Deployment") | .spec.template.spec.containers[0].args' "$rendered" > "$guide_args_file"

  # Build override args as JSON
  local override_file="${WORK_DIR}/override-args.json"
  jq -n --args '$ARGS.positional' -- "${override_args[@]}" > "$override_file"

  # Merge: replace model (first arg), then override matching flags by name
  jq -n --slurpfile guide "$guide_args_file" \
        --slurpfile overrides "$override_file" \
        --arg model "$model_arg" '
    def flag_key: if startswith("--") then split("=")[0] else . end;
    $guide[0] |
    .[0] = $model |
    reduce ($overrides[0][]) as $oarg (.;
      ($oarg | flag_key) as $okey |
      if any(.[]; flag_key == $okey) then
        map(if flag_key == $okey then $oarg else . end)
      else
        . + [$oarg]
      end
    )
  ' > "$args_file"
}

# --- Transform model server manifests ---
transform_modelserver() {
  local rendered="$1"
  local output="$2"
  local args_file="$3"
  local gpu_key="$GPU_KEY"

  local jq_expr=""
  jq_expr+="if .kind == \"Deployment\" then"
  jq_expr+="  .spec.replicas = ${REPLICAS}"
  jq_expr+=" | .spec.template.spec.containers[0].args = \$newargs[0]"

  local res=".spec.template.spec.containers[0].resources"
  if [[ -n "$gpu_key" ]]; then
    if [[ -n "$GPU_COUNT" ]]; then
      jq_expr+=" | ${res}.limits.\"${gpu_key}\" = ${GPU_COUNT}"
      jq_expr+=" | ${res}.requests.\"${gpu_key}\" = ${GPU_COUNT}"
    else
      jq_expr+=" | del(${res}.limits.\"${gpu_key}\")"
      jq_expr+=" | del(${res}.requests.\"${gpu_key}\")"
    fi
  fi
  if [[ -n "$CPU_LIMIT" ]]; then
    jq_expr+=" | ${res}.limits.cpu = \"${CPU_LIMIT}\""
  else
    jq_expr+=" | del(${res}.limits.cpu)"
  fi
  if [[ -n "$CPU_REQUEST" ]]; then
    jq_expr+=" | ${res}.requests.cpu = \"${CPU_REQUEST}\""
  else
    jq_expr+=" | del(${res}.requests.cpu)"
  fi
  if [[ -n "$MEMORY_LIMIT" ]]; then
    jq_expr+=" | ${res}.limits.memory = \"${MEMORY_LIMIT}\""
  else
    jq_expr+=" | del(${res}.limits.memory)"
  fi
  if [[ -n "$MEMORY_REQUEST" ]]; then
    jq_expr+=" | ${res}.requests.memory = \"${MEMORY_REQUEST}\""
  else
    jq_expr+=" | del(${res}.requests.memory)"
  fi

  jq_expr+=' | (.spec.template.spec.volumes[] | select(.name == "shm") | .emptyDir.sizeLimit) = "'"${SHM_SIZE}"'"'

  jq_expr+=' | .metadata.labels."llm-d.ai/model" = "'"${MODEL_SHORT}"'"'
  jq_expr+=' | .spec.selector.matchLabels."llm-d.ai/model" = "'"${MODEL_SHORT}"'"'
  jq_expr+=' | .spec.template.metadata.labels."llm-d.ai/model" = "'"${MODEL_SHORT}"'"'

  if [[ "$MODEL_SOURCE" == "pvc" ]]; then
    jq_expr+=' | .spec.template.spec.volumes += [{"name": "model-storage", "persistentVolumeClaim": {"claimName": "'"${PVC_NAME}"'"}}]'
    jq_expr+=' | .spec.template.spec.containers[0].volumeMounts += [{"name": "model-storage", "mountPath": "'"${MODEL_MOUNT_PATH}"'"}]'
    jq_expr+=' | .spec.template.spec.containers[0].env += [{"name": "HF_HOME", "value": "'"${MODEL_MOUNT_PATH}"'"}]'
    jq_expr+=' | .spec.template.spec.containers[0].env += [{"name": "HF_HUB_OFFLINE", "value": "1"}]'
    jq_expr+=' | del(.spec.template.spec.containers[0].env[] | select(.name == "HF_TOKEN"))'
    jq_expr+=' | .spec.template.spec.securityContext.runAsUser = 0'
  elif [[ "$MODEL_SOURCE" == "hostpath" ]]; then
    jq_expr+=' | .spec.template.spec.volumes += [{"name": "model-storage", "hostPath": {"path": "'"${HOST_PATH}"'", "type": "DirectoryOrCreate"}}]'
    jq_expr+=' | .spec.template.spec.containers[0].volumeMounts += [{"name": "model-storage", "mountPath": "'"${MODEL_MOUNT_PATH}"'"}]'
    jq_expr+=' | .spec.template.spec.containers[0].env += [{"name": "HF_HOME", "value": "'"${MODEL_MOUNT_PATH}"'"}]'
    jq_expr+=' | .spec.template.spec.securityContext.runAsUser = 0'
  fi

  jq_expr+=" else . end"

  yq -y --slurpfile newargs "$args_file" "${jq_expr}" "$rendered" > "$output"
}

# --- Generate router override values ---
generate_router_override() {
  local tmpfile="$1"

  if [[ -n "$EPP_CONFIG" && "$EPP_CONFIG" != "null" ]]; then
    local config_filename
    config_filename="custom-plugins.yaml"

    cat > "$tmpfile" <<HELMEOF
router:
  epp:
    pluginsConfigFile: "${config_filename}"
    pluginsCustomConfig:
      ${config_filename}: |
$(sed 's/^/        /' "$EPP_CONFIG")
  modelServers:
    matchLabels:
      llm-d.ai/guide: "${GUIDE}"
HELMEOF
  fi
}

# --- Cleanup mode ---
if $CLEANUP; then
  log "Cleaning up deployment for guide=${GUIDE} in namespace=${NAMESPACE}..."

  log "Uninstalling Helm release: ${GUIDE}"
  helm uninstall "$GUIDE" -n "$NAMESPACE" 2>/dev/null || warn "Helm release not found"

  log "Deleting model server resources..."
  kustomize build "$MODELSERVER_PATH" | kubectl delete -n "$NAMESPACE" -f - 2>/dev/null || warn "Some resources not found"

  if [[ "$ROUTER_MODE" == "gateway" ]]; then
    log "Deleting the automatically created Gateway..."
    kustomize build "$GATEWAY_RECIPE_DIR" | kubectl delete -n "$NAMESPACE" -f - 2>/dev/null || warn "Gateway not found"
  fi

  log "Keeping namespace '${NAMESPACE}' and mlperf-harness pod intact."

  log "Cleanup complete (llm-d resources removed, namespace preserved)."
  exit 0
fi

# --- Temp directory for generated files ---
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

# --- Deploy / Dry-run ---

if ! $DRY_RUN; then
  if [[ "$ROUTER_MODE" == "gateway" ]]; then
    # Gateway mode creates an HTTPRoute, which needs the standard Gateway API CRDs
    # in addition to the Gateway API Inference Extension CRDs below.
    GATEWAY_API_VERSION="${GATEWAY_API_VERSION:-v1.5.1}"
    log "Installing Gateway API CRDs (${GATEWAY_API_VERSION})..."
    kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml" || warn "Some Gateway API CRDs managed by platform (continuing)"

    ISTIO_VERSION="${ISTIO_VERSION:-1.29.2}"
    ISTIO_DIR="${WORK_DIR}/istio-${ISTIO_VERSION}"
    log "Installing Istio (${ISTIO_VERSION}) with Gateway API Inference Extension support..."
    (
      cd "$WORK_DIR"
      curl --fail --silent --show-error --location https://istio.io/downloadIstio | ISTIO_VERSION="$ISTIO_VERSION" sh -
    )
    [[ -x "${ISTIO_DIR}/bin/istioctl" ]] || die "istioctl was not downloaded as expected"
    "${ISTIO_DIR}/bin/istioctl" install -y \
      --set profile=minimal \
      --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true
    kubectl rollout status deployment/istiod -n istio-system --timeout=5m
  fi

  # Install GAIE CRDs
  log "Installing Gateway API Inference Extension CRDs (${GAIE_VERSION})..."
  kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GAIE_VERSION}/v1-manifests.yaml"

  # Create namespace
  log "Creating namespace: ${NAMESPACE}"
  kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

  # Create HF token secret (needed for huggingface and hostpath sources)
  if [[ "$MODEL_SOURCE" == "huggingface" || "$MODEL_SOURCE" == "hostpath" ]]; then
    if [[ -n "${HF_TOKEN:-}" ]]; then
      log "Creating HF token secret..."
      kubectl create secret generic llm-d-hf-token \
        --from-literal="HF_TOKEN=${HF_TOKEN}" \
        --namespace "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    elif [[ "$MODEL_SOURCE" == "huggingface" ]]; then
      die "HF_TOKEN environment variable is required for huggingface model source"
    else
      warn "HF_TOKEN not set — creating dummy secret for hostpath source"
      kubectl create secret generic llm-d-hf-token \
        --from-literal="HF_TOKEN=dummy" \
        --namespace "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    fi
  fi

  if [[ "$ROUTER_MODE" == "gateway" ]]; then
    log "Creating Gateway: ${GATEWAY_NAME}"
    kustomize build "$GATEWAY_RECIPE_DIR" | kubectl apply -n "$NAMESPACE" -f -
    kubectl wait --for=jsonpath='{.status.conditions[?(@.type=="Programmed")].status}=True' \
      "gateway/${GATEWAY_NAME}" -n "$NAMESPACE" --timeout=5m || warn "Gateway wait timed out (may already be programmed)"
  fi
fi

# --- Pull router chart locally ---
CHART_CACHE="${SCRIPT_DIR}/.charts"
CHART_TGZ="${CHART_CACHE}/${ROUTER_CHART_NAME}-${ROUTER_CHART_VERSION}.tgz"

if [[ ! -f "$CHART_TGZ" ]]; then
  log "Pulling router chart ${ROUTER_CHART_VERSION}..."
  mkdir -p "$CHART_CACHE"
  echo '{}' > "${WORK_DIR}/config.json"
  DOCKER_CONFIG="$WORK_DIR" helm pull "$ROUTER_CHART" \
    --version "$ROUTER_CHART_VERSION" --destination "$CHART_CACHE"
  [[ -f "$CHART_TGZ" ]] || die "Failed to pull router chart"
fi

# --- Router ---
log "Preparing router..."

HELM_VALUES_ARGS=(-f "$ROUTER_BASE_VALUES")
if [[ -n "$GUIDE_VALUES" ]]; then
  HELM_VALUES_ARGS+=(-f "$GUIDE_VALUES")
fi

ROUTER_OVERRIDE="${WORK_DIR}/router-override.values.yaml"
if [[ -n "$EPP_CONFIG" && "$EPP_CONFIG" != "null" ]]; then
  generate_router_override "$ROUTER_OVERRIDE"
  HELM_VALUES_ARGS+=(-f "$ROUTER_OVERRIDE")
fi

if [[ "$ROUTER_MODE" == "gateway" ]]; then
  HELM_VALUES_ARGS+=(
    --set "provider.name=istio"
    --set "httpRoute.create=true"
    --set "httpRoute.inferenceGatewayName=llm-d-inference-gateway"
  )
fi

if $DRY_RUN; then
  if [[ "$ROUTER_MODE" == "gateway" ]]; then
    echo ""
    echo "================================================================"
    echo "  AUTOMATIC GATEWAY MANIFESTS"
    echo "================================================================"
    echo ""
    kustomize build "$GATEWAY_RECIPE_DIR"
  fi

  echo ""
  echo "================================================================"
  echo "  ROUTER HELM TEMPLATE"
  echo "================================================================"
  echo ""
  helm template "$GUIDE" "$CHART_TGZ" \
    "${HELM_VALUES_ARGS[@]}" \
    -n "$NAMESPACE"
else
  log "Installing ${ROUTER_MODE} router Helm chart..."
  helm upgrade --install "$GUIDE" "$CHART_TGZ" \
    "${HELM_VALUES_ARGS[@]}" \
    -n "$NAMESPACE"
fi

# --- Model Server ---
log "Preparing model server..."

RENDERED="${WORK_DIR}/rendered.yaml"
kustomize build "$MODELSERVER_PATH" > "$RENDERED"

ARGS_FILE="${WORK_DIR}/merged-args.json"
build_merged_args "$RENDERED" "$ARGS_FILE"

TRANSFORMED="${WORK_DIR}/transformed.yaml"
transform_modelserver "$RENDERED" "$TRANSFORMED" "$ARGS_FILE"

if $DRY_RUN; then
  echo ""
  echo "================================================================"
  echo "  MODEL SERVER MANIFESTS"
  echo "================================================================"
  echo ""
  cat "$TRANSFORMED"
  echo ""
  log "Dry-run complete. No resources were created."
else
  log "Applying model server manifests..."
  kubectl apply -n "$NAMESPACE" -f "$TRANSFORMED"

  echo ""
  log "Deployment complete!"
  log ""
  if [[ "$ROUTER_MODE" == "standalone" ]]; then
    log "Get the proxy IP:"
    log "  kubectl get service ${GUIDE}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}'"
  else
    log "Get the Gateway address:"
    log "  kubectl get gateway ${GATEWAY_NAME} -n ${NAMESPACE} -o jsonpath='{.status.addresses[0].value}'"
  fi
  log ""
  log "To clean up:"
  log "  $(basename "$0") -o ${OVERRIDE_FILE} --cleanup"
fi
