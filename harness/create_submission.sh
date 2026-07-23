
DIR=$1
rm -rf SUBMISSION_CHECK
rm -rf SUBMISSION_TEST
cp -R ${DIR} SUBMISSION_CHECK
bash scripts/run_compliance_checks.sh SUBMISSION_CHECK/

bash scripts/check_accuracy.sh SUBMISSION_CHECK/server/accuracy/
bash scripts/check_accuracy.sh SUBMISSION_CHECK/offline/accuracy/

python3 scripts/convert_to_submission.py --input-dir SUBMISSION_CHECK/ --output-dir SUBMISSION_TEST --system-name "8xH200-LLM-D-Openshift" --model "gpt-oss-120b"


export SUBMIT_ROOT=./SUBMISSION_TEST/
export TRUNC_ROOT="$SUBMIT_ROOT/_truncated_v6"
python3 ../tools/submission/truncate_accuracy_log.py  --input "$SUBMIT_ROOT"  --submitter RedHat  --output "$TRUNC_ROOT"

cp scripts/8xH200-LLM-D-Openshift.json ./SUBMISSION_TEST/_truncated_v6/closed/RedHat/systems/8xH200-LLM-D-Openshift.json
cp default.conf ./SUBMISSION_TEST/_truncated_v6/closed/RedHat/results/8xH200-LLM-D-Openshift/gpt-oss-120b/Server/user.conf
cp offline.conf ./SUBMISSION_TEST/_truncated_v6/closed/RedHat/results/8xH200-LLM-D-Openshift/gpt-oss-120b/Offline/user.conf
 
python3 ../tools/submission/submission_checker/main.py  --input "$TRUNC_ROOT"  --version v6.0

