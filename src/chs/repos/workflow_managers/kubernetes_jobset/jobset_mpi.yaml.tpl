{custom_resources_section}
apiVersion: v1
kind: Secret
metadata:
  name: {sanitized_job_name}-ssh
  namespace: {kubernetes_namespace}
type: Opaque
stringData:
  ssh-privatekey: |
{ssh_private_key_indented}
  ssh-publickey: |
{ssh_public_key_indented}
---
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
      subdomain: chs
      terminationGracePeriodSeconds: 5
{node_selector_section}
{affinity_section}
{tolerations_section}
{init_containers_section}
{pod_resource_claims_section}
      volumes:
      - name: mpi-id
        secret:
          secretName: {sanitized_job_name}-ssh
          items:
          - key: ssh-privatekey
            path: "id_rsa"
          - key: ssh-publickey
            path: "id_rsa.pub"
{mpi_volumes_append}
      containers:
      - name: mpi-container
        stdin: true
        tty: true
        image: "{container_image}"
{container_security_context_section}
{container_resources_section}
        env:
        - name: OMPI_ALLOW_RUN_AS_ROOT
          value: "1"
        - name: OMPI_ALLOW_RUN_AS_ROOT_CONFIRM
          value: "1"
        - name: MY_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        command:
        - bash
        - -c
        - |
          export DEBIAN_FRONTEND=noninteractive
          mkdir -p /run/sshd ~/.ssh
          chmod 700 ~/.ssh
          cp /secrets/ssh/* ~/.ssh/
          cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
          chmod 600 ~/.ssh/*
          /sbin/sshd -p {ssh_port}

          if [ -x /sbin/ldconfig ]; then /sbin/ldconfig; fi

          export NODE_RANK=$JOB_COMPLETION_INDEX

          mkdir -p "{container_experiment_run_dir}"
          mkdir -p "{workdir}"
          cd "{workdir}"

          if [[ "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
            echo "Discovering hosts..."
            for i in $(seq 0 $(( {n_nodes} - 1 ))); do
              OTHER="{sanitized_job_name}-${i}.chs"
              until ssh -p {ssh_port} -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 $OTHER true; do
                echo "Waiting for $OTHER to become available..."
                sleep 5
              done
              echo "${OTHER} slots={processes_per_node}" >> /tmp/hostfile
            done


            echo "--- STARTING MPI EXECUTION ---"
            {indented_mpi_command}
            EXIT_CODE=$?
            echo "--- END MPI EXECUTION ---"

            exit $EXIT_CODE
          else
            echo "Worker started. Idling while launcher coordinates..."
            HEAD_NODE="{sanitized_job_name}-0.chs"
            DNS_DEADLINE=$(( $(date +%s) + {dns_wait_timeout} ))

            # Wait for head node to become reachable
            echo "Waiting for head node ${HEAD_NODE} to become reachable..."
            while ! (echo > /dev/tcp/$HEAD_NODE/{ssh_port}) 2>/dev/null; do
              if [[ $(date +%s) -ge $DNS_DEADLINE ]]; then
                echo "ERROR: Head node ${HEAD_NODE} did not become reachable within {dns_wait_timeout}s." >&2
                exit 1
              fi
              sleep 5
            done

            echo "Head node is up. Idling until it finishes..."
            # Once reachable, stay alive as long as it replies
            while (echo > /dev/tcp/$HEAD_NODE/{ssh_port}) 2>/dev/null; do
              sleep 5
            done

            echo "Head node is no longer reachable. Worker terminating."
          fi
        volumeMounts:
        - name: mpi-id
          mountPath: "/secrets/ssh"
          readOnly: true
{mpi_volume_mounts_append}
