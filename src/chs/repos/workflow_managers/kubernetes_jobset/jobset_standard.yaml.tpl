{custom_resources_section}
apiVersion: batch/v1
kind: Job
metadata:
  name: {sanitized_job_name}
  namespace: {kubernetes_namespace}
{jobset_labels_section}
{jobset_annotations_section}
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
  activeDeadlineSeconds: {timeout}
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
          {indented_standard_command}
{standard_volume_mounts_section}
