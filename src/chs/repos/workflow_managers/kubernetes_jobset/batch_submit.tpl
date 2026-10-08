#!/bin/bash
set -eo pipefail
if [ "{application_name}" = "chs" ]; then
    echo "Skipping JobSet submission for {application_name} scaffolding experiment."
    echo "Submitted {experiment_name} at $(date)" > {log_file}
    exit 0
fi

kubectl delete -f "{experiment_run_dir}/jobset.yaml" --ignore-not-found --timeout=60s || true
kubectl apply -f "{experiment_run_dir}/jobset.yaml" --request-timeout=60s
echo "Submitted {experiment_name} at $(date)" > {log_file}
