#!/bin/bash
set -eo pipefail
if [ "{application_name}" = "chs" ]; then
    exit 0
fi

# We use the sanitized name and --ignore-not-found so it doesn't throw a fatal error if already deleted
kubectl delete jobset "{sanitized_job_name}" --namespace "{kubernetes_namespace}" --ignore-not-found --timeout=60s
