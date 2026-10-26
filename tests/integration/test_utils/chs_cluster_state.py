# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Seeds and inspects CHS objects in the `chs` namespace of the Kind cluster.

These helpers stand in for another CHS run that targets nodes on the
cluster, so tests can exercise node conflict detection without a second
live run. Like real CHS Jobs, the seeded Jobs are attributed to their run
by the instance label and pinned to nodes by hostname node affinity.
"""

import json
import subprocess

CHS_NAMESPACE = "chs"
MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
INSTANCE_LABEL = "app.kubernetes.io/instance"
COMPONENT_LABEL = "app.kubernetes.io/component"
JOB_WAIT_TIMEOUT = "120s"
SLICE_CRD = "slices.accelerator.gke.io"


def _kubectl(*args: str, input_str: str | None = None) -> str:
    res = subprocess.run(
        ["kubectl", *args],
        input=input_str,
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout


def _apply(manifest: dict) -> None:
    _kubectl("apply", "-f", "-", input_str=json.dumps(manifest))


def ensure_namespace() -> None:
    _apply(
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": CHS_NAMESPACE},
        }
    )


def run_labels(run_id: str) -> dict[str, str]:
    return {MANAGED_BY_LABEL: "chs", INSTANCE_LABEL: run_id}


def _hostname_affinity(nodes: list[str]) -> dict:
    expression = {
        "key": "kubernetes.io/hostname",
        "operator": "In",
        "values": sorted(nodes),
    }
    required = {"nodeSelectorTerms": [{"matchExpressions": [expression]}]}
    return {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": required
        }
    }


def _create_job(
    name: str,
    nodes: list[str],
    labels: dict[str, str],
    command: list[str],
    suspend: bool,
) -> str:
    """Creates a Job with one pod per node, pinned to the nodes."""
    ensure_namespace()
    _apply(
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": CHS_NAMESPACE,
                "labels": labels,
            },
            "spec": {
                "suspend": suspend,
                "completions": len(nodes),
                "parallelism": len(nodes),
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": {**labels, COMPONENT_LABEL: "worker"}
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "terminationGracePeriodSeconds": 0,
                        "affinity": _hostname_affinity(nodes),
                        "containers": [
                            {
                                "name": "workload",
                                "image": "alpine:latest",
                                "imagePullPolicy": "IfNotPresent",
                                "command": command,
                            }
                        ],
                    },
                },
            },
        }
    )
    return name


def create_suspended_job(run_id: str, nodes: list[str]) -> str:
    """Creates a suspended Job of another run, as if waiting for admission."""
    return _create_job(
        f"chs-{run_id}-suspended",
        nodes,
        run_labels(run_id),
        ["sleep", "3600"],
        suspend=True,
    )


def create_running_job(run_id: str, nodes: list[str]) -> str:
    """Creates a Job of another run and waits until its pods are running."""
    name = _create_job(
        f"chs-{run_id}-running",
        nodes,
        run_labels(run_id),
        ["sleep", "3600"],
        suspend=False,
    )
    _kubectl(
        "wait",
        f"job/{name}",
        "-n",
        CHS_NAMESPACE,
        f"--for=jsonpath={{.status.ready}}={len(nodes)}",
        f"--timeout={JOB_WAIT_TIMEOUT}",
    )
    return name


def create_completed_job(run_id: str, nodes: list[str]) -> str:
    """Creates a Job of another run and waits until it is Complete."""
    name = _create_job(
        f"chs-{run_id}-completed",
        nodes,
        run_labels(run_id),
        ["true"],
        suspend=False,
    )
    _kubectl(
        "wait",
        f"job/{name}",
        "-n",
        CHS_NAMESPACE,
        "--for=condition=Complete",
        f"--timeout={JOB_WAIT_TIMEOUT}",
    )
    return name


def create_unlabeled_suspended_job(nodes: list[str]) -> str:
    """Creates a suspended Job that no CHS run owns."""
    return _create_job(
        "legacy-suspended",
        nodes,
        {"app": "not-chs"},
        ["sleep", "3600"],
        suspend=True,
    )


def create_instance_only_suspended_job(run_id: str, nodes: list[str]) -> str:
    """Creates a suspended Job with a run's instance label but no managed-by.

    Other tools also set the instance label, so such a Job is not a CHS Job.
    """
    return _create_job(
        f"foreign-{run_id}-suspended",
        nodes,
        {INSTANCE_LABEL: run_id},
        ["sleep", "3600"],
        suspend=True,
    )


def create_worker_service(run_id: str) -> str:
    """Creates the headless worker Service another CHS run would own."""
    name = f"chs-{run_id}"
    ensure_namespace()
    _apply(
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name,
                "namespace": CHS_NAMESPACE,
                "labels": run_labels(run_id),
            },
            "spec": {
                "clusterIP": "None",
                "selector": {
                    **run_labels(run_id),
                    COMPONENT_LABEL: "worker",
                },
            },
        }
    )
    return name


def create_ssh_secret(run_id: str, job_name: str) -> str:
    """Creates the SSH Secret another CHS run would own for one of its Jobs."""
    name = f"{job_name}-ssh"
    ensure_namespace()
    _apply(
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": name,
                "namespace": CHS_NAMESPACE,
                "labels": run_labels(run_id),
            },
            "stringData": {"id_rsa": "not-a-real-key"},
        }
    )
    return name


def create_slice(run_id: str, nodepool: str, partition_ids: list[str]) -> str:
    """Creates the dynamic TPU Slice another CHS run would own for a pool."""
    name = f"chs-slice-{nodepool}"
    _apply(
        {
            "apiVersion": "accelerator.gke.io/v1beta1",
            "kind": "Slice",
            "metadata": {"name": name, "labels": run_labels(run_id)},
            "spec": {
                "type": "tpu7x",
                "topology": "4x4x4",
                "partitionIds": partition_ids,
            },
        }
    )
    return name


def _slice_crd_exists() -> bool:
    res = subprocess.run(
        ["kubectl", "get", "crd", SLICE_CRD],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0


def slice_names() -> set[str]:
    if not _slice_crd_exists():
        return set()
    out = _kubectl("get", SLICE_CRD, "-o", "name")
    return {line.split("/", 1)[1] for line in out.split()}


def namespace_exists() -> bool:
    res = subprocess.run(
        ["kubectl", "get", "namespace", CHS_NAMESPACE],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0


def _names(kind: str) -> set[str]:
    if not namespace_exists():
        return set()
    args = ["get", kind, "-n", CHS_NAMESPACE, "-o", "name"]
    return {line.split("/", 1)[1] for line in _kubectl(*args).split()}


def job_names() -> set[str]:
    return _names("jobs")


def service_names() -> set[str]:
    return _names("services")


def secret_names() -> set[str]:
    return _names("secrets")


def pod_phases(job_name: str) -> list[str]:
    out = _kubectl(
        "get",
        "pods",
        "-n",
        CHS_NAMESPACE,
        "-l",
        f"batch.kubernetes.io/job-name={job_name}",
        "-o",
        "jsonpath={.items[*].status.phase}",
    )
    return out.split()


def reset_namespace() -> None:
    """Deletes every CHS Slice and every Job, Service and Secret in `chs`.

    The namespace itself is kept because CHS never deletes it on run
    paths; only its contents are cleared so each test starts from a clean
    cluster. Jobs are deleted in the foreground so that a terminating Job
    of a previous test cannot still target nodes in the next one.
    """
    if _slice_crd_exists():
        _kubectl(
            "delete",
            SLICE_CRD,
            "-l",
            f"{MANAGED_BY_LABEL}=chs",
            "--ignore-not-found",
            "--wait=true",
            "--timeout=120s",
        )
    if not namespace_exists():
        return
    for kind in ("jobs", "services", "secrets"):
        _kubectl(
            "delete",
            kind,
            "--all",
            "-n",
            CHS_NAMESPACE,
            "--ignore-not-found",
            "--cascade=foreground",
            "--wait=true",
            "--timeout=120s",
        )
