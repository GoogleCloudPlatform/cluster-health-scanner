#!/bin/bash

if [[ $# -ne 3 ]]; then
    echo "Error: expected exactly 3 arguments to uninstall_from_cluster.sh"
    exit 1 # exit with error
fi

/builder/helm.bash # call image's entrypoint to initialize credentials for communicating across clusters

release_name=$1
cronjob_regex=$2
job_regex=$3

helm uninstall "$release_name"
kubectl get cronjobs -o name | grep "cronjob.batch/$cronjob_regex" | xargs -r kubectl delete
kubectl get jobs -o name | grep "job.batch/$job_regex" | xargs -r kubectl delete

exit 0 # exit with success, even if any of the steps failed
# they might fail, for instance, if the job had already been uninstalled from the cluster by a previous failed run
