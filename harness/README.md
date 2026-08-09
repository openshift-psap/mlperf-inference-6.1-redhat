# MLPerf Inference 6.1 Harness — GPT-OSS-120B

This guide provides step-by-step instructions for reproducing the MLPerf Inference 6.1 submission for GPT-OSS-120B on NVIDIA GB200 NVL4 on Red Hat OpenShift.

For latest setup instructions and code:

- Clone the repo: [https://github.com/openshift-psap/mlperf-inference-6.1-redhat](https://github.com/openshift-psap/mlperf-inference-6.1-redhat)
- Follow the instructions in [README.md](https://github.com/openshift-psap/mlperf-inference-6.1-redhat/blob/master/harness/README.md)
- Open an issue if you encounter any blocker

## Table of Contents

1. [Hardware and Software Overview](#hardware-and-software-overview)
2. [Prerequisites](#prerequisites)
3. [Model Storage Setup](#model-storage-setup)
4. [Deployment Setup](#deployment-setup)
5. [Client Pod Setup](#client-pod-setup)
6. [Running Tests](#running-tests)
7. [Creating Submission](#creating-submission)
8. [Quick Reference](#quick-reference)
9. [Troubleshooting](#troubleshooting)

---

## Hardware and Software Overview

### Hardware: NVIDIA GB200 NVL4


| Component            | Specification                                                |
| -------------------- | ------------------------------------------------------------ |
| GPUs                 | 4x NVIDIA GB200 (Blackwell), 189GB HBM3e each                |
| CPUs                 | 2x NVIDIA Grace (ARM), 72 cores each (144 total)             |
| GPU–GPU Interconnect | NVLink 5.0, 1.8 TB/s bidirectional                           |
| CPU–GPU Interconnect | NVLink-C2C, 900 GB/s bidirectional                           |
| NUMA Topology        | GPU 0,1 → NUMA 0 (CPUs 0–71); GPU 2,3 → NUMA 1 (CPUs 72–143) |


### Software Stack


| Component | Version / Details                   |
| --------- | ----------------------------------- |
| vLLM      | 0.24.0 (`vllm/vllm-openai:v0.24.0`) |
| Platform  | Red Hat OpenShift 4.21              |
| Storage   | LVM operator for block storage      |
| Python    | 3.12                                |


### Deployment Configuration


| Parameter                  | Server            | Offline           |
| -------------------------- | ----------------- | ----------------- |
| Replicas                   | 4                 | 4                 |
| Tensor Parallel Size       | 1                 | 1                 |
| GPU Memory Utilization     | 0.98              | 0.98              |
| KV Cache Dtype             | fp8               | fp8               |
| MoE Backend                | flashinfer_trtllm | flashinfer_trtllm |
| MoE Activation Quant       | mxfp8             | mxfp8             |
| Max Model Length           | 49 000            | 49 000            |
| Max Num Seqs               | 1 024             | 1 024             |
| Max Num Batched Tokens     | 4 096             | 16 384            |
| Max CUDAGraph Capture Size | 4 096             | 2 048             |
| Performance Mode           | interactivity     | throughput        |
| Prefix Caching             | Disabled          | Disabled          |


---

## Prerequisites

### Cluster Requirements

- OpenShift (or Kubernetes) cluster with:
  - **4 NVIDIA GB200 GPUs** on a single node
  - A storage provisioner (LVM operator, Ceph, local-path, etc.)
- Cluster-admin access (required for SCC grants on OpenShift)

### Local Machine Tools

```bash
# macOS
brew install jq kubectl helm kustomize git curl
pip3 install yq

# RHEL / Fedora
sudo dnf install jq kubectl helm kustomize git curl
pip3 install yq
```

> **Important:** Install `yq` via `pip3 install yq` (Python wrapper around `jq`), NOT via `dnf`/`brew`/`snap` which install the Go version (`mikefarah/yq`). The deployment scripts require the Python `yq` syntax. Verify with: `yq --version` — it should show `yq <version>` without `(https://github.com/mikefarah/yq/)`.

OpenShift CLI (`oc`): download from  
[https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/](https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/)

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

All pods (model servers + client) mount a shared PVC for model weights, datasets, and gpt-oss-120b tokenizer files.

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


| Requirement  | Detail                                     |
| ------------ | ------------------------------------------ |
| Access mode  | `ReadWriteOnce`                            |
| Binding mode | `WaitForFirstConsumer` (GPU-node affinity) |
| Backing      | NVMe recommended for fast model loading    |


> `ReadWriteOnce` means the PVC binds to one node. All pods must schedule there. On GB200 NVL4 this is fine — all 4 GPUs are on the same node.

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
HF_HOME=/mnt/models huggingface-cli download openai/gpt-oss-120b
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

## Deployment Setup

### 1. Clone the Repository

```bash
git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.1-redhat.git
cd mlperf-inference-6.1-redhat
```

### 2. Deploy vLLM Model Servers

```bash
cd setup/llm-d/GB200/
```

**Server scenario:**

```bash
bash deploy_gptoss120b.sh server --standalone
```

**Offline scenario:**

```bash
bash deploy_gptoss120b.sh offline --standalone
```

The script deploys 4 vLLM model server replicas on the cluster, applies OpenShift fixes (SCC, vLLM image override, environment variables), and creates a `vllm-direct` K8s service for client access.

### 3. Verify

```bash
# All 4 model server pods should be Running
oc get pods -n llm-d-bench

# Check model-server logs
oc logs -n llm-d-bench -l llm-d.ai/model=gpt-oss-120b --tail=5

# Verify the API service
curl -s http://vllm-direct.llm-d-bench.svc.cluster.local:8000/v1/models
```

### 4. API URL

```
http://vllm-direct.llm-d-bench.svc.cluster.local:8000
```

> This URL is only reachable from within the cluster.

### Switching Scenarios

Server and Offline scenarios use different vLLM configurations. To switch:

```bash
bash deploy_gptoss120b.sh server --standalone --cleanup
bash deploy_gptoss120b.sh offline --standalone
```

---

## Client Pod Setup

The client pod runs the MLPerf harness inside the cluster.

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

### 4. Enter the Client Pod

```bash
oc exec -it mlperf-client -n llm-d-bench -- bash
source /mnt/models/test-mlperf/gptoss_harness/bin/activate
cd /mnt/models/test-mlperf/mlperf-inference-6.1-redhat/harness
ulimit -n 65536
```

---

## Running Tests

All tests are run from inside the client pod at `/mnt/models/test-mlperf/mlperf-inference-6.1-redhat/harness`.

### Server Scenario

```bash
# Run all (performance + accuracy + compliance)
bash run_server.sh

# Run individual test types
bash run_server.sh performance
bash run_server.sh accuracy
bash run_server.sh compliance
bash run_server.sh compliance test07
bash run_server.sh compliance test09
```

**Configuration via environment variables:**


| Variable            | Default                 | Description                           |
| ------------------- | ----------------------- | ------------------------------------- |
| `SERVER_TARGET_QPS` | 39                      | Target queries per second             |
| `NUM_WORKERS`       | 12                      | Async workers for concurrent requests |
| `OUTPUT_DIR`        | `harness_output/server` | Output directory                      |
| `MAX_PERF_RETRIES`  | 5                       | Retry performance if INVALID          |


Example with custom QPS:

```bash
SERVER_TARGET_QPS=36 bash run_server.sh performance
```

### Offline Scenario

```bash
# Run all (performance + accuracy + compliance)
bash run_offline.sh

# Run individual test types
bash run_offline.sh performance
bash run_offline.sh accuracy
bash run_offline.sh compliance
bash run_offline.sh compliance test07
bash run_offline.sh compliance test09
```

**Configuration via environment variables:**


| Variable              | Default                  | Description                             |
| --------------------- | ------------------------ | --------------------------------------- |
| `NUM_WORKERS`         | 12                       | Async workers                           |
| `OFFLINE_CONCURRENCY` | 6396                     | Max concurrent requests                 |
| `OUTPUT_DIR`          | `harness_output/offline` | Output directory                        |
| `MAX_PERF_RUNS`       | 5                        | Number of performance runs (picks best) |


### Important Notes

- `ulimit -n 65536` must be set before running tests
- Server performance retries up to `MAX_PERF_RETRIES` times if INVALID
- Offline performance runs `MAX_PERF_RUNS` times and picks the best VALID result
- Results are saved to the `OUTPUT_DIR` with subdirectories: `performance/`, `accuracy/`, `compliance/`

---

## Creating Submission

### 1. Prepare Results

Ensure both Server and Offline scenarios have completed with VALID results:

```bash
grep "Result is" harness_output/server/performance/mlperf/mlperf_log_summary.txt
grep "Result is" harness_output/offline/performance/mlperf/mlperf_log_summary.txt
```

### 2. Structure for Submission Checker

```bash
rm -rf SUBMISSION_CHECK
mkdir -p SUBMISSION_CHECK/server SUBMISSION_CHECK/offline

cp -R harness_output/server/performance SUBMISSION_CHECK/server/
cp -R harness_output/server/accuracy SUBMISSION_CHECK/server/
cp -R harness_output/server/compliance SUBMISSION_CHECK/server/

cp -R harness_output/offline/performance SUBMISSION_CHECK/offline/
cp -R harness_output/offline/accuracy SUBMISSION_CHECK/offline/
cp -R harness_output/offline/compliance SUBMISSION_CHECK/offline/
```

### 3. Run Compliance and Accuracy Checks

```bash
export DATASET_DIR=/mnt/models/datasets/gpt-oss_data/
export HF_HOME=/mnt/models
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

bash scripts/run_compliance_checks.sh SUBMISSION_CHECK/
bash scripts/check_accuracy.sh SUBMISSION_CHECK/server/accuracy/
bash scripts/check_accuracy.sh SUBMISSION_CHECK/offline/accuracy/
```

### 4. Convert and Package

```bash
python3 scripts/convert_to_submission.py \
  --input-dir SUBMISSION_CHECK/ \
  --output-dir SUBMISSION_TEST \
  --system-name "4xGB200-Openshift" \
  --model "gpt-oss-120b"

export SUBMIT_ROOT=./SUBMISSION_TEST/
export TRUNC_ROOT="$SUBMIT_ROOT/_truncated_v6"
rm -rf "$TRUNC_ROOT"
python3 ../tools/submission/truncate_accuracy_log.py \
  --input "$SUBMIT_ROOT" --submitter RedHat --output "$TRUNC_ROOT"
```

### 5. Copy System JSON and Run Checker

```bash
cp scripts/4xGB200-LLM-D-Openshift.json ./SUBMISSION_TEST/_truncated_v6/closed/RedHat/systems/
cp default.conf ./SUBMISSION_TEST/_truncated_v6/closed/RedHat/results/4xGB200-Openshift/gpt-oss-120b/Server/user.conf
cp offline.conf ./SUBMISSION_TEST/_truncated_v6/closed/RedHat/results/4xGB200-Openshift/gpt-oss-120b/Offline/user.conf

python3 ../tools/submission/submission_checker/main.py \
  --input ./SUBMISSION_TEST/_truncated_v6 --version v6.1
```

Expected: `SUMMARY: submission looks OK`

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

# 3. Deploy (server scenario)
cd setup/llm-d/GB200/
bash deploy_gptoss120b.sh server --standalone

# 4. Client pod
cd ../../../
oc apply -f setup/client/GB200/client-pod.yaml
oc cp setup/client/GB200/client_setup.sh llm-d-bench/mlperf-client:/client_setup.sh
oc exec -it mlperf-client -n llm-d-bench -- bash -c 'bash /client_setup.sh'

# 5. Run tests (inside client pod)
oc exec -it mlperf-client -n llm-d-bench -- bash
source /mnt/models/test-mlperf/gptoss_harness/bin/activate
cd /mnt/models/test-mlperf/mlperf-inference-6.1-redhat/harness
ulimit -n 65536

bash run_server.sh          # Full server scenario
bash run_offline.sh         # Full offline scenario (requires offline deploy first)

# 6. Create submission
# See "Creating Submission" section
```

---

## Troubleshooting

### Model-Server Pods Not Starting


| Symptom                                                      | Cause                               | Fix                                                                                                                                                                                                     |
| ------------------------------------------------------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unable to validate against any security context constraint` | SCC not granted before pod creation | Grant SCC, then delete the stale ReplicaSet: `oc delete rs -n llm-d-bench $(oc get rs -n llm-d-bench --no-headers -o custom-columns=":metadata.name" | grep optimized-baseline-nvidia-gpu-vllm-decode)` |
| `PermissionError: …/flashinfer_cubin`                        | Container not running as root       | Ensure `runAsUser: 0` is in the pod security context                                                                                                                                                    |
| `LocalEntryNotFoundError: Cannot find cached snapshot`       | HF cache incomplete                 | Verify `refs/main`, `snapshots/`, and `blobs/` exist under `/mnt/models/hub/models--openai--gpt-oss-120b/`                                                                                              |


### PVC Permission Denied


| Symptom                                            | Cause                                 | Fix                                                                             |
| -------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------- |
| `Permission denied: '/mnt/models/'` as root        | SELinux blocking LVM volume access    | Set permissive on the node: `oc debug node/<name> -- chroot /host setenforce 0` |
| `Permission denied: '/mnt/models/hub/…/refs/main'` | File permissions from LVM provisioner | `oc exec <pod> -- chmod -R 777 /mnt/models/`                                    |


### Harness Errors


| Symptom                                            | Cause                                            | Fix                                                                     |
| -------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------- |
| `Too many open files`                              | Default ulimit too low                           | `ulimit -n 65536` before running tests                                  |
| `ServerDisconnectedError` / `ConnectionResetError` | vLLM temporarily dropping connections under load | Harness retries automatically. If persistent, check pod logs            |
| `can't start new thread`                           | Pod PID limit too low                            | Apply `KubeletConfig` with `podPidsLimit: 32768` (requires node reboot) |


### Accuracy

- Threshold: **82.30 %**
- If below threshold: verify all 15 safetensors shards present, tokenizer files intact, TIKTOKEN encoding directory set correctly.

---

## Additional Resources

- Deployment configs: `setup/llm-d/GB200/`
- Environment variables: `harness/scripts/set_env_vars.sh`
- Submission converter: `harness/scripts/convert_to_submission.py`

