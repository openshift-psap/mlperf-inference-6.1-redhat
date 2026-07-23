# MLPerf Inference Benchmark Results

This repository contains our MLPerf Inference v6.1 benchmark results and setup documentation.

## Results Summary

### NVIDIA Hardware Results

| Model Category | Model | GPU Configuration | Offline Scenario (throughput) | Server Scenario (throughput) | Software Stack |
|---------------|-------|-------------------|-------------------------------|------------------------------|----------------|
| **Reasoning Model** | gpt-oss-120b | 4x GB200 (189 GB HBM3e) | TBD | TBD | OpenShift, llm-d, vLLM 0.24.0 |

## Submission Results

For detailed submission results, see: [MLCommons Inference Results](https://mlcommons.org/benchmarks/inference/) <!-- Update with actual submission link -->

## Setup Documentation

Detailed setup and configuration instructions for each benchmark:

- **GPT-OSS-120B**: See [harness/README.md](harness/README.md) for harness setup and configuration

## Repository Structure

```
.
├── README.md (this file)
├── harness/              # GPT-OSS-120B harness and configuration
├── setup/
│   ├── llm-d/GB200/      # llm-d deployment for GB200 NVL4
│   └── client/GB200/     # Client pod setup
└── language/             # Language model benchmarks
```

## Quick Start

```bash
# 1. Clone
git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.1-redhat.git
cd mlperf-inference-6.1-redhat

# 2. Deploy llm-d on GB200
cd setup/llm-d/GB200/
bash deploy_gptoss120b.sh server

# 3. Set up client pod
cd ../../../
oc apply -f setup/client/GB200/client-pod.yaml
oc cp setup/client/GB200/client_setup.sh llm-d-bench/mlperf-client:/client_setup.sh
oc exec -it mlperf-client -n llm-d-bench -- bash -c 'bash /client_setup.sh'

# 4. Run tests (inside client pod)
# See harness/README.md for full instructions
```

## About MLPerf Inference

MLPerf Inference is a benchmark suite for measuring how fast systems can run models in a variety of deployment scenarios. For more information, visit [MLCommons](https://mlcommons.org/).

## Scenarios

- **Offline**: Batch inference maximizing throughput
- **Server**: Online serving under TTFT (Time To First Token) and TPOT (Time Per Output Token) constraints
