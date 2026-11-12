{custom_resources_section}
apiVersion: batch/v1
kind: Job
metadata:
  name: {sanitized_job_name}
  namespace: {kubernetes_namespace}
{job_labels_section}
{job_annotations_section}
spec:
  {suspend_section}
  completionMode: Indexed
  backoffLimit: 0
  podFailurePolicy:
    rules:
    - action: FailJob
      onExitCodes:
        operator: NotIn
        values:
        - 0
  parallelism: {n_nodes}
  completions: {n_nodes}
  activeDeadlineSeconds: {job_deadline}
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
{pod_labels_section}
{pod_annotations_section}
    spec:
      restartPolicy: Never
      subdomain: {worker_service_name}
      terminationGracePeriodSeconds: 5
{node_selector_section}
{affinity_section}
{tolerations_section}
{pod_resource_claims_section}
{standard_volumes_section}
      containers:
      - name: standard-container
        stdin: true
        tty: true
        image: "{container_image}"
{container_security_context_section}
{container_resources_section}
        env:
        - name: MY_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        command:
        - bash
        - -c
        - |
          export NODE_RANK=$JOB_COMPLETION_INDEX
          mkdir -p "{container_experiment_run_dir}"
          mkdir -p "{workdir}"
          cd "{workdir}"
          cat > /tmp/chs_test_command.sh <<'CHS_TEST_COMMAND_EOF'
          {indented_standard_command}
          CHS_TEST_COMMAND_EOF
          CHS_TEST_START=$SECONDS
          timeout --kill-after=10 {timeout} bash /tmp/chs_test_command.sh
          CHS_EXIT_CODE=$?
          if [ $(( SECONDS - CHS_TEST_START )) -ge {timeout} ] && [ $CHS_EXIT_CODE -eq 124 -o $CHS_EXIT_CODE -eq 137 ]; then
            CHS_EXIT_CODE=124
            echo "CHS: the test did not finish within its {timeout}s timeout." >&2
          elif [ $CHS_EXIT_CODE -eq 124 ]; then
            CHS_EXIT_CODE=1
          fi
          exit $CHS_EXIT_CODE
{standard_volume_mounts_section}
