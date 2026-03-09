#!/bin/bash
if [ "{application_name}" = "chs" ]; then
    echo "Skipping JobSet wait for {application_name} scaffolding experiment."
    exit 0
fi

JOBSET_NAME="{sanitized_job_name}"
LOG_FILE="{log_file}"

echo "========================"
echo "Waiting for JobSet to Complete: $JOBSET_NAME"
echo "========================"

TIMEOUT_OCCURRED=0
if ! kubectl wait --for=condition=Completed "jobset/$JOBSET_NAME" --timeout=300s; then
    if ! kubectl wait --for=condition=Failed "jobset/$JOBSET_NAME" --timeout=5s; then
        TIMEOUT_OCCURRED=1
    fi
fi

echo "Extracting logs..."

# The primary pod for both standard and MPI templates is ALWAYS rank 0 in the "w" replicatedJob
HEAD_POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$JOBSET_NAME,batch.kubernetes.io/job-completion-index=0 -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)

if [ -z "$HEAD_POD" ]; then
    echo "Error: Head pod not found." > "$LOG_FILE"
    exit 1
fi

echo "Saving logs to $LOG_FILE"
kubectl logs "$HEAD_POD" > "$LOG_FILE"

if [ "$TIMEOUT_OCCURRED" -eq 1 ]; then
    echo "Error: JobSet $JOBSET_NAME failed to complete within 5 minutes." >> "$LOG_FILE"
    exit 1
fi
