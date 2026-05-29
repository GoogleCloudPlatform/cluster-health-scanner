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
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: {sanitized_job_name}
  namespace: {kubernetes_namespace}
{jobset_labels_section}
{jobset_annotations_section}
spec:
  {suspend_section}
  ttlSecondsAfterFinished: 86400
  network:
    enableDNSHostnames: true
    publishNotReadyAddresses: true
  replicatedJobs:
    - name: w
      template:
        spec:
          parallelism: {n_nodes}
          completions: {n_nodes}
          activeDeadlineSeconds: {deadline}
          template:
            metadata:
{pod_labels_section}
{pod_annotations_section}
            spec:
              restartPolicy: Never
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

                  export POD_NAME=$(hostname -s)
                  export JOBSET_NAME=${POD_NAME%-w-0-*}
                  export NODE_RANK=$JOB_COMPLETION_INDEX

                  mkdir -p "{container_experiment_run_dir}"
                  mkdir -p "{workdir}"
                  cd "{workdir}"

                  if [[ "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
                    echo "Discovering hosts..."
                    for i in $(seq 0 $(( {n_nodes} - 1 ))); do
                      OTHER="${JOBSET_NAME}-w-0-${i}.${JOBSET_NAME}"
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
                    HEAD_NODE="${JOBSET_NAME}-w-0-0.${JOBSET_NAME}"

                    # Wait for head node to become reachable
                    echo "Waiting for head node ${HEAD_NODE} to become reachable..."
                    while ! (echo > /dev/tcp/$HEAD_NODE/{ssh_port}) 2>/dev/null; do
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
              restartPolicy: Never
