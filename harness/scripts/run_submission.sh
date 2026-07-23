DESC=$1
PORT=$2
QPS=$3
MLFLOW_HOST=$4
DATASET_DIR="/mnt/data

rm audit.config

echo "----------------------- AUDIT 07---------------------"
echo "Running TEST07 compliance performance"
DIR=${DESC}_OFFLINE_COMPLIANCE_TEST07
rm -rf ${DIR}
cp ../compliance/TEST07/gpt-oss-120b/audit.config ./
  python3 harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ${DATASET_DIR}/acc/acc_eval_compliance_gpqa.parquet --test-mode performance --api-server-url http://localhost:${PORT} --scenario Offline --backend vllm --lg-model-name gpt-oss-120b --output-dir ${DIR} --audit-config audit.config  --mlflow-experiment-name GPT-OSS-120B-Experiments --mlflow-host ${MLFLOW_HOST} --mlflow-description "${DESC} Offline Compliance TEST07" --mlflow-tag "system:8xH200,group:FullSubmission,type:Offline Compliance TEST07"
python3 ../compliance/TEST07/run_verification.py -c ${DIR}/mlperf/ -o ${DIR} \
    --audit-config ../compliance/TEST07/gpt-oss-120b/audit.config \
    --accuracy-script "python3 ../language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log ${DIR}/mlperf/mlperf_log_accuracy.json \
        --reference-data ${DATASET_DIR}/acc/acc_eval_compliance_gpqa.parquet  \
        --tokenizer openai/gpt-oss-120b"
 python3 upload_to_mlflow.py --metadata-file ${DIR}/mlflow_metadata.yaml --mlflow-host ${MLFLOW_HOST} --mlflow-port 5000 --mlflow-experiment-name GPT-OSS-120B-Experiments

DIR=${DESC}_SERVER_QPS${QPS}_COMPLIANCE_TEST07
rm -rf ${DIR}
cp ../compliance/TEST07/gpt-oss-120b/audit.config ./
  python3 harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ${DATASET_DIR}/acc/acc_eval_compliance_gpqa.parquet --test-mode performance --api-server-url http://localhost:${PORT} --scenario Server --server-target-qps ${QPS} --backend vllm --lg-model-name gpt-oss-120b --output-dir ${DIR} --audit-config audit.config  --mlflow-experiment-name GPT-OSS-120B-Experiments --mlflow-host ${MLFLOW_HOST} --mlflow-description "${DESC} Server QPS${QPS} Compliance TEST07"  --mlflow-tag "system:8xH200,group:FullSubmission,type:Server Compliance TEST07"
python3 ../compliance/TEST07/run_verification.py -c ${DIR}/mlperf/ -o ${DIR} \
    --audit-config ../compliance/TEST07/gpt-oss-120b/audit.config \
    --accuracy-script "python3 ../language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log ${DIR}/mlperf/mlperf_log_accuracy.json \
        --reference-data ${DATASET_DIR}/acc/acc_eval_compliance_gpqa.parquet  \
        --tokenizer openai/gpt-oss-120b"
 python3 upload_to_mlflow.py --metadata-file ${DIR}/mlflow_metadata.yaml --mlflow-host ${MLFLOW_HOST} --mlflow-port 5000 --mlflow-experiment-name GPT-OSS-120B-Experiments


echo "----------------------- AUDIT 09---------------------"

echo "Running TEST09 offline compliance performance"
DIR=${DESC}_OFFLINE_COMPLIANCE_TEST09
rm -rf ${DIR}
cp ../compliance/TEST09/gpt-oss-120b/audit.config ./
  python3 harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ${DATASET_DIR}/perf/perf_eval_ref.parquet --test-mode performance --api-server-url http://localhost:${PORT} --scenario Offline --backend vllm --lg-model-name gpt-oss-120b --output-dir ${DIR} --audit-config audit.config  --mlflow-experiment-name GPT-OSS-120B-Experiments --mlflow-host ${MLFLOW_HOST} --mlflow-description "${DESC} Offline Compliance TEST09"  --mlflow-tag "system:8xH200,group:FullSubmission,type:Offline Compliance TEST09"
python3 ../compliance/TEST09/run_verification.py  -c ${DIR}/mlperf/  -o ${DIR}   --audit-config ../compliance/TEST09/gpt-oss-120b/audit.config
 python3 upload_to_mlflow.py --metadata-file ${DIR}/mlflow_metadata.yaml --mlflow-host ${MLFLOW_HOST} --mlflow-port 5000 --mlflow-experiment-name GPT-OSS-120B-Experiments

echo "Running TEST09 server compliance performance"
DIR=${DESC}_SERVER_QPS${QPS}_COMPLIANCE_TEST09
rm -rf ${DIR}
cp ../compliance/TEST09/gpt-oss-120b/audit.config ./
  python3 harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ${DATASET_DIR}/perf/perf_eval_ref.parquet --test-mode performance --api-server-url http://localhost:${PORT} --scenario Server --server-target-qps ${QPS} --backend vllm --lg-model-name gpt-oss-120b --output-dir ${DIR} --audit-config audit.config  --mlflow-experiment-name GPT-OSS-120B-Experiments --mlflow-host ${MLFLOW_HOST} --mlflow-description "${DESC} Server QPS${QPS} Compliance TEST09" --mlflow-tag "system:8xH200,group:FullSubmission,type:Server Compliance TEST09"
python3 ../compliance/TEST09/run_verification.py  -c ${DIR}/mlperf/  -o ${DIR}   --audit-config ../compliance/TEST09/gpt-oss-120b/audit.config
 python3 upload_to_mlflow.py --metadata-file ${DIR}/mlflow_metadata.yaml --mlflow-host ${MLFLOW_HOST} --mlflow-port 5000 --mlflow-experiment-name GPT-OSS-120B-Experiments

rm audit.config

echo "-------------------------- PERFORMANCE-----------------------"
echo "Running offline performance"
DIR=${DESC}_OFFLINE_PERFORMANCE

rm -rf ${DIR}
   python3 harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ${DATASET_DIR}/perf/perf_eval_ref.parquet --test-mode performance --api-server-url http://localhost:${PORT} --scenario Offline --backend vllm --lg-model-name gpt-oss-120b --output-dir ${DIR} --mlflow-experiment-name GPT-OSS-120B-Experiments --mlflow-host ${MLFLOW_HOST} --mlflow-description "${DESC} Offline Performance" --mlflow-tag "system:8xH200,group:FullSubmission,type:Offline Perf"

echo "Running server performance"
DIR=${DESC}_SERVER_QPS${QPS}_PERFORMANCE
rm -rf ${DIR}
   python3 harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ${DATASET_DIR}/perf/perf_eval_ref.parquet --test-mode performance --api-server-url http://localhost:${PORT} --scenario Server --server-target-qps ${QPS} --backend vllm --lg-model-name gpt-oss-120b --output-dir ${DIR} --mlflow-experiment-name GPT-OSS-120B-Experiments --mlflow-host ${MLFLOW_HOST} --mlflow-description "${DESC} Server ${QPS3} Performance"  --mlflow-tag "system:8xH200,group:FullSubmission,type:Server Perf"



echo "-------------------------- PERFORMANCE-----------------------"

echo "Running offline Accuracy"
DIR=${DESC}_OFFLINE_ACCURACY
rm -rf ${DIR}
  python3 harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ${DATASET_DIR}/acc/acc_eval_ref.parquet --test-mode accuracy --api-server-url http://localhost:${PORT} --scenario Offline --backend vllm --lg-model-name gpt-oss-120b --output-dir ${DIR} --offline-back-to-back --offline-async-concurrency 2048 --mlflow-experiment-name GPT-OSS-120B-Experiments --mlflow-host ${MLFLOW_HOST} --mlflow-description "${DESC} Offline Accuracy" --mlflow-tag "system:8xH200,group:FullSubmission,type:Offline Accuracy"
python3 ../language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log ${DIR}/mlperf/mlperf_log_accuracy.json \
        --reference-data ${DATASET_DIR}/acc/acc_eval_ref.parquet  \
        --tokenizer openai/gpt-oss-120b  >& ${DIR}/accuracy.txt
 python3 upload_to_mlflow.py --metadata-file ${DIR}/mlflow_metadata.yaml --mlflow-host ${MLFLOW_HOST} --mlflow-port 5000 --mlflow-experiment-name GPT-OSS-120B-Experiments

echo "Running server Accuracy"
DIR=${DESC}_SERVER_QPS${QPS}_ACCURACY
rm -rf ${DIR}
   python3 harness_main.py --model-category gpt-oss-120b --model openai/gpt-oss-120b --dataset-path ${DATASET_DIR}/acc/acc_eval_ref.parquet --test-mode accuracy --api-server-url http://localhost:${PORT} --scenario Server --server-target-qps ${QPS} --backend vllm --lg-model-name gpt-oss-120b --output-dir ${DIR} --mlflow-experiment-name GPT-OSS-120B-Experiments --mlflow-host ${MLFLOW_HOST} --mlflow-description "${DESC} Server ${QPS3} Accuracy" --mlflow-tag "system:8xH200,group:FullSubmission,type:Server Accuracy"
python3 ../language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log ${DIR}/mlperf/mlperf_log_accuracy.json \
        --reference-data ${DATASET_DIR}/acc/acc_eval_ref.parquet  \
        --tokenizer openai/gpt-oss-120b  >& ${DIR}/accuracy.txt

 python3 upload_to_mlflow.py --metadata-file ${DIR}/mlflow_metadata.yaml --mlflow-host ${MLFLOW_HOST} --mlflow-port 5000 --mlflow-experiment-name GPT-OSS-120B-Experiments
