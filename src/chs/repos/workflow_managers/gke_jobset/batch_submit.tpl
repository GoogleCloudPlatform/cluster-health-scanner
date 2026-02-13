#!/bin/bash
if [ "{application_name}" = "chs-gke" ]; then
    echo "Skipping JobSet submission for {application_name} scaffolding experiment."
    echo "Submitted {experiment_name} at $(date)" > {log_file}
    exit 0
fi

kubectl delete -f "{experiment_run_dir}/jobset.yaml" --ignore-not-found || true
kubectl apply -f "{experiment_run_dir}/jobset.yaml"
echo "Submitted {experiment_name} at $(date)" > {log_file}
