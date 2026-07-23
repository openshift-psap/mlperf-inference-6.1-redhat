# MLPerf Inference 6.1 Harness — GPT-OSS-120B

This guide provides step-by-step instructions for reproducing the MLPerf Inference 6.1 submission for GPT-OSS-120B on NVIDIA GB200 NVL4 using llm-d on OpenShift.

For latest setup instructions and code:
- Clone the repo: https://github.com/openshift-psap/mlperf-inference-6.1-redhat
- Follow the instructions in [README.md](https://github.com/openshift-psap/mlperf-inference-6.1-redhat/blob/master/harness/README.md)
- Open an issue if you encounter any blocker

## Table of Contents

1. [Hardware and Software Overview](#hardware-and-software-overview)
2. [Prerequisites](#prerequisites)
3. [Model Storage Setup](#model-storage-setup)
4. [LLM-D Deployment](#llm-d-deployment)
5. [Client Pod Setup](#client-pod-setup)
6. [Running Tests with run_submission.py](#running-tests-with-run_submissionpy)
7. [Creating Submission](#creating-submission)
8. [Quick Reference](#quick-reference)
9. [Troubleshooting](#troubleshooting)

---

## Hardware and Software Overview

### Hardware: NVIDIA GB200 NVL4

| Component | Specification |
|-----------|---------------|
| GPUs | 4x NVIDIA GB200 (Blackwell), 189GB HBM3e each |
| CPUs | 2x NVIDIA Grace (ARM), 72 cores each (144 total) |
| GPU–GPU Interconnect | NVLink 5.0, 1.8 TB/s bidirectional |
| CPU–GPU Interconnect | NVLink-C2C, 900 GB/s bidirectional |
| NUMA Topology | GPU 0,1 → NUMA 0 (CPUs 0–71); GPU 2,3 → NUMA 1 (CPUs 72–143) |

### Software Stack

| Component | Version / Details |
|-----------|-------------------|
| vLLM | 0.24.0 (`vllm/vllm-openai:v0.24.0`) |
| FlashInfer | 0.6.12 (FLASHINFER_TRTLLM_MXFP4_MXFP8 MoE backend) |
| llm-d | `main` branch |
| Gateway | Istio 1.29.2 with Gateway API Inference Extension |
| Platform | Red Hat OpenShift (LVM operator for block storage) |
| Python | 3.12 |

### Model: GPT-OSS-120B

| Property | Value |
|----------|-------|
| Model ID | `openai/gpt-oss-120b` |
| Architecture | Sparse MoE — 117 B total params, 5.1 B active per token |
| Expert Layout | 128 experts, top-4 routing |
| Native Weight Format | MXFP4 |
| Disk Footprint | ~500 GB |

### Deployment Configuration

| Parameter | Server | Offline |
|-----------|--------|---------|
| Replicas | 4 | 4 |
| Tensor Parallel Size | 1 | 1 |
| GPU Memory Utilization | 0.98 | 0.98 |
| KV Cache Dtype | fp8 | fp8 |
| MoE Backend | flashinfer_trtllm | flashinfer_trtllm |
| MoE Activation Quant | mxfp8 | mxfp8 |
| Max Model Length | 49 000 | 49 000 |
| Max Num Seqs | 1 024 | 1 024 |
| Max Num Batched Tokens | 8 192 | 16 384 |
| Max CUDAGraph Capture Size | 2 048 | 8 192 |
| Prefix Caching | Disabled | Disabled |
| EPP Scorers | queue(2) + kv-cache(2) | queue(2) + kv-cache(2) |

---

## Prerequisites

### Cluster Requirements

- OpenShift (or Kubernetes) cluster with:
  - **4 NVIDIA GB200 GPUs** on a single node
  - **32+ CPUs** for model-server pods (8 per replica × 4 replicas)
  - **8 CPUs / 32 Gi RAM** for the client pod
  - Istio 1.29+ with `ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true`
  - A storage provisioner (LVM operator, Ceph, local-path, etc.)
- Cluster-admin access (required for SCC grants on OpenShift)

### Local Machine Tools

```bash
# macOS
brew install yq jq kubectl helm kustomize git curl

# RHEL / Fedora
sudo dnf install yq jq kubectl helm kustomize git curl
```

OpenShift CLI (`oc`): download from  
<https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/>

### Verify Cluster Access

```bash
export KUBECONFIG=/path/to/your/kubeconfig
oc get nodes
```

### OpenShift: Pod PID Limit

The MLPerf harness can launch thousands of threads. Increase `podPidsLimit`:

```yaml
apiVersion: machineconfiguration.openshift.io/v1
kind: KubeletConfig
metadata:
  name: high-pid-limit
spec:
  kubeletConfig:
    podPidsLimit: 32768
```

> This requires a node reboot to take effect.

---

## Model Storage Setup

All pods (model servers + client) mount a shared PVC for model weights, datasets, and tokenizer files.

### 1. Create Namespace and PVC

```bash
oc create namespace llm-d-bench
```

```yaml
# model-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: llm-d-bench
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi
  storageClassName: lvms-nvme   # ← change to your storage class
```

```bash
oc apply -f model-pvc.yaml
```

**Storage class notes:**

| Requirement | Detail |
|-------------|--------|
| Access mode | `ReadWriteOnce` |
| Binding mode | `WaitForFirstConsumer` (GPU-node affinity) |
| Backing | NVMe recommended for fast model loading |

> `ReadWriteOnce` means the PVC binds to one node. All pods must schedule there. On GB200 NVL4 this is fine — all 4 GPUs are on the same node.

```bash
# Check available storage classes
oc get storageclass
```

### 2. Bind the PVC

The PVC stays `Pending` until a pod mounts it. Create a temporary binder pod:

```bash
cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: model-pvc-binder
  namespace: llm-d-bench
spec:
  restartPolicy: Never
  securityContext:
    runAsUser: 0
  containers:
    - name: binder
      image: vllm/vllm-openai:v0.24.0
      command: ["sleep", "infinity"]
      volumeMounts:
        - name: model-storage
          mountPath: /mnt/models
      resources:
        requests:
          cpu: "4"
          memory: 8Gi
  volumes:
    - name: model-storage
      persistentVolumeClaim:
        claimName: model-pvc
EOF

# Grant SCC (OpenShift only)
oc adm policy add-scc-to-user privileged -z default -n llm-d-bench

# Wait for binding
oc wait pod/model-pvc-binder -n llm-d-bench --for=condition=Ready --timeout=120s
oc get pvc model-pvc -n llm-d-bench   # STATUS should be "Bound"
```

### 3. Populate the PVC

#### Option A — Download from HuggingFace

Requires a HuggingFace token with access to `openai/gpt-oss-120b`.

```bash
oc exec -it model-pvc-binder -n llm-d-bench -- bash

# Inside the pod
pip install huggingface-hub
huggingface-cli login --token YOUR_HF_TOKEN
huggingface-cli download openai/gpt-oss-120b --cache-dir /mnt/models
exit
```

#### Option B — Copy from host path

If model weights already exist on the node (e.g. `/var/lib/mlperf/models`):

```bash
# Delete binder and recreate with both mounts
oc delete pod model-pvc-binder -n llm-d-bench --wait

cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: model-pvc-binder
  namespace: llm-d-bench
spec:
  restartPolicy: Never
  securityContext:
    runAsUser: 0
  containers:
    - name: binder
      image: vllm/vllm-openai:v0.24.0
      command: ["sleep", "infinity"]
      volumeMounts:
        - name: pvc
          mountPath: /mnt/models
        - name: host
          mountPath: /mnt/hostpath
          readOnly: true
      resources:
        requests:
          cpu: "4"
          memory: 8Gi
  volumes:
    - name: pvc
      persistentVolumeClaim:
        claimName: model-pvc
    - name: host
      hostPath:
        path: /var/lib/mlperf/models
        type: DirectoryOrCreate
EOF

oc wait pod/model-pvc-binder -n llm-d-bench --for=condition=Ready --timeout=120s
oc exec -it model-pvc-binder -n llm-d-bench -- bash

# Inside the pod — copy weights, encodings, and datasets
cp -a /mnt/hostpath/hub            /mnt/models/
cp -a /mnt/hostpath/gpt-oss-encoding /mnt/models/
cp -a /mnt/hostpath/datasets       /mnt/models/
exit
```

### 4. Fix File Permissions

LVM-provisioned volumes may carry restrictive ownership. Fix recursively:

```bash
oc exec model-pvc-binder -n llm-d-bench -- chmod -R 777 /mnt/models/
```

On OpenShift with SELinux **enforcing**, you may also need to set permissive mode on the node:

```bash
oc debug node/<node-name> -- chroot /host setenforce 0
```

### 5. Verify PVC Contents

```bash
oc exec model-pvc-binder -n llm-d-bench -- ls -la /mnt/models/
```

Expected layout:

```
/mnt/models/
├── hub/
│   └── models--openai--gpt-oss-120b/
│       ├── blobs/          # Safetensors weight shards
│       ├── refs/main       # Points to the snapshot hash
│       └── snapshots/<hash>/
├── gpt-oss-encoding/       # Tiktoken encoding files for GPT-OSS-120B
└── datasets/
    └── gpt-oss_data/       # MLPerf benchmark datasets
```

### 6. Copy MLPerf Datasets

If not already present:

```bash
# Download from https://inference.mlcommons-storage.org/index.html#gpt-oss-benchmark
oc exec -it model-pvc-binder -n llm-d-bench -- mkdir -p /mnt/models/datasets/gpt-oss_data
oc cp /path/to/local/datasets/ llm-d-bench/model-pvc-binder:/mnt/models/datasets/gpt-oss_data/
```

### 7. Clean Up Binder Pod

```bash
oc delete pod model-pvc-binder -n llm-d-bench
```

---

## LLM-D Deployment

### 1. Clone the Repository

```bash
git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.1-redhat.git
cd mlperf-inference-6.1-redhat
```

### 2. Deploy

```bash
cd setup/llm-d/GB200/
```

**Server scenario:**

```bash
bash deploy_gptoss120b.sh server
```

**Offline scenario:**

```bash
bash deploy_gptoss120b.sh offline
```

The script runs two phases:

| Phase | Script | What it does |
|-------|--------|--------------|
| 1 | `install_llmd.sh` | Clones llm-d repo; installs Istio 1.29 with Gateway API Inference Extension; installs GAIE CRDs; creates namespace + HF token secret; deploys Istio gateway; installs router Helm chart with EPP config; renders model-server manifests (PVC mount, `HF_HOME`, `HF_HUB_OFFLINE=1`, `runAsUser: 0`); applies manifests |
| 2 | `apply_ocp_fixes.sh` | Grants SCC (`privileged` for model-server SA, `anyuid` for gateway SA); overrides vLLM image to v0.24.0; deletes stale ReplicaSets; sets `TIKTOKEN_ENCODINGS_BASE`, `TIKTOKEN_RS_CACHE_DIR`, `HF_HUB_OFFLINE=1` env vars; fixes SELinux labels (hostpath only); sets HTTPRoute timeout to 7 200 s; waits for vLLM pods (5–15 min); fixes gateway `ulimit -n` to 65 536; verifies end-to-end inference |

### 3. Verify

```bash
# All pods should be Running
oc get pods -n llm-d-bench

# Expected output (4 model servers + EPP + gateway):
# optimized-baseline-nvidia-gpu-vllm-decode-xxxx   1/1   Running
# optimized-baseline-nvidia-gpu-vllm-decode-xxxx   1/1   Running
# optimized-baseline-nvidia-gpu-vllm-decode-xxxx   1/1   Running
# optimized-baseline-nvidia-gpu-vllm-decode-xxxx   1/1   Running
# optimized-baseline-epp-xxxx                      1/1   Running
# llm-d-inference-gateway-istio-xxxx               1/1   Running

# Check model-server logs
oc logs -n llm-d-bench -l llm-d.ai/model=gpt-oss-120b --tail=5

# Verify gateway
oc get gateway llm-d-inference-gateway -n llm-d-bench
```

### 4. Gateway URL

```
http://llm-d-inference-gateway.llm-d-bench.svc.cluster.local:80
```

> This URL is only reachable from within the cluster.

### Switching Scenarios

```bash
bash deploy_gptoss120b.sh server --cleanup
bash deploy_gptoss120b.sh offline
```

### Re-fixing Gateway Ulimits

The file-descriptor fix is **temporary** — it does not survive pod restarts. If the gateway pod restarts:

```bash
./apply_ocp_fixes.sh -o override_gptoss120b_server.yaml --fix-ulimits
```

---

## Client Pod Setup

The client pod runs the MLPerf harness inside the cluster so it can reach the gateway.

### 1. Grant SCC

```bash
oc adm policy add-scc-to-user privileged -z default -n llm-d-bench
```

### 2. Create the Client Pod

```bash
oc apply -f setup/client/GB200/client-pod.yaml
oc wait --for=condition=Ready pod/mlperf-client -n llm-d-bench --timeout=120s
```

### 3. Run the Setup Script

```bash
oc cp setup/client/GB200/client_setup.sh llm-d-bench/mlperf-client:/client_setup.sh
oc exec -it mlperf-client -n llm-d-bench -- bash -c 'bash /client_setup.sh'
```

The script installs system packages, clones `mlperf-inference-6.1-redhat`, creates a Python 3.12 venv (`gptoss_harness`), and installs all benchmark + harness dependencies.

### 4. Configure Environment

```bash
oc exec -it mlperf-client -n llm-d-bench -- bash

# Activate venv
source /gptoss_harness/bin/activate
cd /mlperf-inference-6.1-redhat/harness

# Source helper
source scripts/set_env_vars.sh

# Required variables
export API_SERVER_URL=http://llm-d-inference-gateway.llm-d-bench.svc.cluster.local:80
export DATASET_DIR=/mnt/models/datasets/gpt-oss_data/
export HF_HOME=/mnt/models
export TIKTOKEN_ENCODINGS_BASE=/mnt/models/gpt-oss-encoding
export TIKTOKEN_RS_CACHE_DIR=/mnt/models/gpt-oss-encoding
export OUTPUT_DIR=./harness_output
export AWS_ACCESS_KEY_ID=dummy
export AWS_SECRET_ACCESS_KEY=dummy
ulimit -n 65536

# Verify
print_env_vars
validate_env_vars
```

### 5. Verify Connectivity

```bash
curl -s ${API_SERVER_URL}/v1/models | python3 -m json.tool
```

Should list `openai/gpt-oss-120b`.

---

## Running Tests with run_submission.py

### Prerequisites

1. LLM-D deployed and all model-server pods `Running`
2. Client pod created, venv activated, environment variables set
3. `ulimit -n 65536` in the current shell

### Offline Tests

```bash
python3 scripts/run_submission.py --scenario Offline run-offline
```

Runs: Offline Performance → Accuracy → Compliance (TEST07, TEST09).

Generate a bash script instead:

```bash
python3 scripts/run_submission.py --print-bash --scenario Offline run-offline > run_offline.sh
bash run_offline.sh
```

### Server Tests

```bash
# Adjust --server-target-qps to your measured throughput
python3 scripts/run_submission.py \
  --scenario Server \
  --server-target-qps <QPS> \
  --num-workers 8 \
  run-server
```

Generate a bash script:

```bash
python3 scripts/run_submission.py \
  --scenario Server \
  --print-bash \
  --server-target-qps <QPS> \
  --num-workers 8 \
  run-server > run_server.sh
bash run_server.sh
```

> `--num-workers 8` is recommended for GB200 NVL4 with 4 replicas. A single worker bottlenecks on one asyncio event loop; 128+ workers overwhelm the gateway.

### All Tests

```bash
python3 scripts/run_submission.py --server-target-qps <QPS> run-all
```

### Individual Test Types

```bash
# Performance only
python3 scripts/run_submission.py --scenario Offline run-performance
python3 scripts/run_submission.py --scenario Server --server-target-qps <QPS> run-performance

# Accuracy only
python3 scripts/run_submission.py --scenario Offline run-accuracy
python3 scripts/run_submission.py --scenario Server --server-target-qps <QPS> run-accuracy

# Compliance
python3 scripts/run_submission.py --scenario Offline run-compliance
python3 scripts/run_submission.py --scenario Offline run-compliance TEST07
python3 scripts/run_submission.py --scenario Server --server-target-qps <QPS> run-compliance TEST09
```

### Dry Run

```bash
python3 scripts/run_submission.py --dry-run --scenario Offline run-offline
```

### Command Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--scenario` | `Server` or `Offline` | Yes |
| `--server-target-qps` | Target QPS (Server only) | Yes (Server) |
| `--num-workers` | Async workers (default 1; recommend 8) | No |
| `--output-dir` | Output directory (default `./harness_output`) | No |
| `--dataset-dir` | Dataset directory | No (env var) |
| `--api-server-url` | Gateway URL | No (env var) |
| `--user-conf` | Custom user.conf | No |
| `--audit-config` | Audit config for compliance | No |
| `--dry-run` | Print commands, don't execute | No |
| `--print-bash` | Emit a bash script | No |
| `--tag` | MLflow tags (`k=v,k=v`) | No |

### Commands

| Command | Description |
|---------|-------------|
| `run-server` | All Server tests (perf + accuracy + compliance) |
| `run-offline` | All Offline tests |
| `run-all` | Both scenarios |
| `run-performance` | Performance only |
| `run-accuracy` | Accuracy only |
| `run-compliance` | Compliance (TEST07 + TEST09) |

---

## Creating Submission

### 1. Verify Results

```bash
ls -la harness_output/
# Should contain Server/ and/or Offline/ with performance, accuracy, compliance results
```

### 2. Package

```bash
cd harness
bash create_submission.sh harness_output
```

The script runs compliance checks, verifies accuracy, converts to MLPerf directory structure, truncates accuracy logs, copies system JSON, and runs the submission checker.

### 3. Verify

```bash
ls -la SUBMISSION_TEST/_truncated_v6/closed/RedHat/
```

Expected:

```
SUBMISSION_TEST/_truncated_v6/closed/RedHat/
├── results/
│   └── 4xGB200-LLM-D-Openshift/
│       └── gpt-oss-120b/
│           ├── Server/
│           │   ├── performance/
│           │   ├── accuracy/
│           │   └── compliance/
│           └── Offline/
│               ├── performance/
│               ├── accuracy/
│               └── compliance/
├── systems/
│   └── 4xGB200-LLM-D-Openshift.json
└── ...
```

---

## Quick Reference

```bash
# 1. Clone
git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.1-redhat.git
cd mlperf-inference-6.1-redhat

# 2. Storage (see "Model Storage Setup" for full details)
oc create namespace llm-d-bench
oc apply -f model-pvc.yaml
# … populate PVC with weights, encodings, datasets …

# 3. Deploy llm-d
cd setup/llm-d/GB200/
bash deploy_gptoss120b.sh server    # or "offline"

# 4. Client pod
cd ../../../
oc apply -f setup/client/GB200/client-pod.yaml
oc cp setup/client/GB200/client_setup.sh llm-d-bench/mlperf-client:/client_setup.sh
oc exec -it mlperf-client -n llm-d-bench -- bash -c 'bash /client_setup.sh'

# 5. Run tests (inside client pod)
oc exec -it mlperf-client -n llm-d-bench -- bash
source /gptoss_harness/bin/activate
cd /mlperf-inference-6.1-redhat/harness
source scripts/set_env_vars.sh
export API_SERVER_URL=http://llm-d-inference-gateway.llm-d-bench.svc.cluster.local:80
export DATASET_DIR=/mnt/models/datasets/gpt-oss_data/
export HF_HOME=/mnt/models
export TIKTOKEN_ENCODINGS_BASE=/mnt/models/gpt-oss-encoding
export TIKTOKEN_RS_CACHE_DIR=/mnt/models/gpt-oss-encoding
export AWS_ACCESS_KEY_ID=dummy
export AWS_SECRET_ACCESS_KEY=dummy
ulimit -n 65536

python3 scripts/run_submission.py --scenario Server --server-target-qps <QPS> --num-workers 8 run-server
python3 scripts/run_submission.py --scenario Offline run-offline

# 6. Create submission
bash create_submission.sh harness_output
```

---

## Troubleshooting

### Model-Server Pods Not Starting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `unable to validate against any security context constraint` | SCC not granted before pod creation | Grant SCC, then delete the stale ReplicaSet so the deployment creates new pods: `oc delete rs -n llm-d-bench $(oc get rs -n llm-d-bench --no-headers -o custom-columns=":metadata.name" \| grep optimized-baseline-nvidia-gpu-vllm-decode)` |
| `PermissionError: …/flashinfer_cubin` | Container not running as root | Ensure `runAsUser: 0` is in the pod security context. `install_llmd.sh` sets this for PVC source automatically. |
| `LocalEntryNotFoundError: Cannot find cached snapshot` | HF cache incomplete or `HF_HUB_OFFLINE=1` set before cache is populated | Verify `refs/main`, `snapshots/`, and `blobs/` exist under `/mnt/models/hub/models--openai--gpt-oss-120b/`. If downloading is needed, temporarily remove `HF_HUB_OFFLINE`. |
| `OSError: Repo id must be in the form 'namespace/repo_name': '/mnt/models/gpt-oss-120b'` | Model arg is a filesystem path instead of HF name | The model arg should be `openai/gpt-oss-120b` (resolved via `HF_HOME`), not a path. |

### PVC Permission Denied

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Permission denied: '/mnt/models/'` as root | SELinux blocking LVM volume access | Set permissive on the node: `oc debug node/<name> -- chroot /host setenforce 0` |
| `Permission denied: '/mnt/models/hub/…/refs/main'` | File permissions from LVM provisioner | `oc exec <pod> -- chmod -R 777 /mnt/models/` |

### Gateway / Networking

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Too many open files` | Default ulimit too low | `ulimit -n 65536` before running tests. For gateway pods: `./apply_ocp_fixes.sh -o <override> --fix-ulimits` |
| `ServerDisconnectedError` / `ConnectionResetError` | Gateway overwhelmed or restarted | Harness retries automatically. If persistent, check gateway logs and ensure HTTPRoute timeout = 7200 s. |
| `can't start new thread` | Pod PID limit too low | Apply `KubeletConfig` with `podPidsLimit: 32768` (requires node reboot). |

### Accuracy

- Threshold: **82.30 %**
- If below threshold: verify all 15 safetensors shards present, tokenizer files intact, TIKTOKEN encoding directory set correctly.

---

## Additional Resources

- GB200 deployment configs: `setup/llm-d/GB200/`
- EPP scorer configs: `setup/llm-d/GB200/epp-configs/`
- Environment variables: `harness/scripts/set_env_vars.sh`
- Submission converter: `harness/scripts/convert_to_submission.py`
