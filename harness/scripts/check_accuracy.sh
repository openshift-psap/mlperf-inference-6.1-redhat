DIR=$1
python3 ../language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log ${DIR}/mlperf/mlperf_log_accuracy.json \
        --reference-data ${DATASET_DIR}/acc/acc_eval_ref.parquet  \
        --tokenizer openai/gpt-oss-120b >& ${DIR}/accuracy.txt
