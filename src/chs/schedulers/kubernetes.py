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

"""Kubernetes scheduler implementation for Cluster Health Scanner."""

import urllib.request
import importlib.resources as pkg_resources
import json
import logging
import os
import yaml


import re
import subprocess
import click
from typing import Any, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from chs.utils.app_data import ChsAppData


from chs.schedulers.base import BaseScheduler
from chs.utils.kueue import KueueManager
from chs.exceptions import ChsSchedulerException, KueueException
from chs.utils.lifecycle import JobStatus
from chs.preflights.kubernetes_preflight import KubernetesPreflightChecker
from chs import resources
from chs.constants import CHS_NAMESPACE


def boolean_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:bool", str(data))


yaml.Dumper.add_representer(bool, boolean_representer)

logger = logging.getLogger(__name__)


class KubernetesScheduler(BaseScheduler):
    """Kubernetes scheduler implementation."""

    def __init__(
        self,
        nodes: str | None = None,
        nodelists: str | None = None,
        nodepools: str | None = None,
        partitions: str | None = None,
        app_data: "ChsAppData | None" = None,
    ):
        super().__init__(app_data)
        self.nodes_arg = nodes
        self.nodepools_arg = nodepools
        self._resolved_nodes: list[str] | None = None
        self.workspace_config_dir: str | None = None
        self.workspace_name: str | None = None
        self.kueue_supported: bool = False
        self._jobset_labels_yaml: str = ""
        self._suspend_jobset: str = "false"
        if nodelists:
            logger.warning(
                "nodelists argument is ignored for Kubernetes scheduler."
                " Did you want to use nodes or nodepools?"
            )
        if partitions:
            logger.warning(
                "partitions argument is ignored for Kubernetes scheduler."
                " Did you want to use nodes or nodepools?"
            )

    def _resolve_project_id(self, node: str) -> str:
        """Resolves GCP project ID from Kubernetes node providerID."""
        try:
            provider_id = self._run_kubectl(
                ["get", "node", node, "-o", "jsonpath={.spec.providerID}"]
            ).strip()
            # providerID format: gce://project-id/zone/instance-name
            if provider_id.startswith("gce://"):
                parts = provider_id.replace("gce://", "").split("/")
                if len(parts) >= 1:
                    return parts[0]
            raise ValueError(f"Unexpected providerID format: {provider_id}")
        except Exception as e:
            raise ChsSchedulerException(
                f"Failed to resolve project ID from node: {e}"
            )

    def run_preflight_checks(
        self,
        tests: list[dict[str, Any]] | None = None,
        workspace_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Checks for JobSet and Kueue dependencies on Kubernetes."""
        preflight_checker = KubernetesPreflightChecker(self)
        filtered_tests, kueue_supported = preflight_checker.check(tests)
        self.kueue_supported = kueue_supported
        if workspace_name:
            self.workspace_name = workspace_name
            if self._app_data:
                run_entry = self._app_data.get_run(workspace_name) or {}
                if run_id:
                    run_entry.update({"run_id": run_id, "scheduler": self.name})
                self._app_data.update_run(workspace_name, run_entry)
        return filtered_tests

    def setup_scheduler_resources(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None = None,
        machine_type: str = "",
        workspace_name: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Sets up scheduler-specific resources like Kueue configurations."""
        self._ensure_namespace_exists()
        self.workspace_config_dir = workspace_config_dir
        os.makedirs(workspace_config_dir, exist_ok=True)

        if self.kueue_supported:
            try:
                nodes = self.get_nodes_under_test()
                kueue_mgr = KueueManager(self)
                self._jobset_labels_yaml = kueue_mgr.setup(
                    workspace_config_dir, machine_type, nodes or []
                )
                self._suspend_jobset = "true"
            except KueueException as e:
                logger.warning(
                    "Failed to setup Kueue resources: %s. "
                    "Falling back to standard scheduling without Kueue.",
                    e,
                )
                self.kueue_supported = False

                filtered_tests = (
                    KubernetesPreflightChecker(
                        self
                    )._filter_tests_without_kueue(tests)
                    or []
                )
                return filtered_tests

        return tests

    @property
    def name(self) -> str:
        return "kubernetes"

    def get_cluster_name(self) -> str | None:
        """Retrieves cluster name from GKE configmap or kubectl."""
        # 1. Try GCP Metadata Server (in-cluster GKE)
        try:
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/attributes/cluster-name",
                headers={"Metadata-Flavor": "Google"},
            )
            val = urllib.request.urlopen(req, timeout=2).read().decode("utf-8")
            if val:
                return val
        except Exception:
            pass

        # 2. Try ConfigMap (in-cluster GKE)
        try:
            val = self._run_kubectl(
                [
                    "get",
                    "configmap",
                    "gke-common-webhook",
                    "-n",
                    "kube-system",
                    "-o",
                    "jsonpath={.data.clusterName}",
                ],
                timeout=5,
            )
            if val:
                return val
        except Exception:
            pass

        # 3. Try node labels
        try:
            val = self._run_kubectl(
                [
                    "get",
                    "nodes",
                    "-o",
                    "jsonpath={.items[0].metadata.labels}",
                ],
                timeout=5,
            )
            if val:
                labels = json.loads(val)
                gke_cluster = labels.get(
                    "container.googleapis.com/cluster_name"
                )
                if gke_cluster:
                    return gke_cluster
        except Exception:
            pass

        # 4. Try current-context (local execution)
        try:
            val = self._run_kubectl(["config", "current-context"], timeout=5)
            if val:
                if val.startswith("gke_"):
                    parts = val.split("_")
                    if len(parts) >= 4:
                        return parts[-1]
                return val
        except Exception:
            pass
        return None

    def _ensure_namespace_exists(self) -> None:
        """Ensures the chs namespace exists before deploying resources."""

        try:
            self._run_kubectl(["get", "namespace", CHS_NAMESPACE])
        except subprocess.CalledProcessError:
            self._run_kubectl(["create", "namespace", CHS_NAMESPACE])

    def _delete_namespace(self) -> None:
        """Deletes the chs namespace during cleanup."""
        try:
            self._run_kubectl(
                ["delete", "namespace", CHS_NAMESPACE, "--ignore-not-found"]
            )
        except Exception as e:
            logger.warning("Failed to delete namespace: %s", e)

    def _run_kubectl(
        self,
        args: list[str],
        timeout: int | None = None,
        namespaced: bool = False,
    ) -> str:
        """Runs a kubectl command and returns filtered output."""

        cmd = ["kubectl"]
        if namespaced:
            cmd.extend(["-n", CHS_NAMESPACE])
        cmd.extend(args)
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout.strip()

    def _delete_job_secrets(self, job_ids: list[str]) -> None:
        """Deletes SSH secrets associated with the given job IDs."""
        if not job_ids:
            return
        secret_ids = [f"{job_id}-ssh" for job_id in job_ids]
        args_secrets = (
            ["delete", "secret"] + secret_ids + ["--ignore-not-found"]
        )
        try:
            self._run_kubectl(args_secrets, namespaced=True)
            logger.debug(
                "Successfully deleted Secrets: %s", ", ".join(secret_ids)
            )
        except Exception as e:
            logger.warning("Failed to delete secrets: %s", e)

    def _discover_tpu_topology(self, node_name: str) -> str:
        """Queries node labels to discover TPU topology.

        # TODO: This is GKE-specific. Generalize for generic Kubernetes.
        """
        try:
            args = ["get", "node", node_name, "-o", "json"]
            output = self._run_kubectl(args)
            if not output:
                return ""

            data = json.loads(output)
            labels = data.get("metadata", {}).get("labels", {})
            return labels.get("cloud.google.com/gke-tpu-topology", "")
        except Exception as e:
            logger.warning(
                "Failed to discover TPU topology for node %s: %s", node_name, e
            )
            return ""

    def _group_nodes_by_slice(
        self, node_names: list[str]
    ) -> dict[str, list[str]]:
        """Groups TPU nodes into slices using placement policy or node pool."""
        if not node_names:
            return {}

        slices = defaultdict(list)
        try:
            args = ["get", "nodes"] + node_names + ["-o", "json"]
            output = self._run_kubectl(args)
            if output:
                data = json.loads(output)
                items = data.get("items", [])
                for item in items:
                    node_name = item.get("metadata", {}).get("name")
                    if not node_name:
                        continue
                    labels = item.get("metadata", {}).get("labels", {})

                    slice_id = labels.get("cloud.google.com/gke-nodepool")
                    if not slice_id:
                        slice_id = "default-slice"

                    slices[slice_id].append(node_name)
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Failed to run kubectl for nodes %s: %s", node_names, e
            )
            slices["failed-nodes"].extend(node_names)
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse json for nodes %s: %s", node_names, e
            )
            slices["failed-nodes"].extend(node_names)

        return dict(slices)

    def _discover_networks(self) -> tuple[list[str], list[str]]:
        """Discovers all network names using kubectl and groups them by type.

        # TODO: This is GKE-specific (uses networks.networking.gke.io CRD).

        Returns:
            A tuple of (l3_networks, device_networks).
            L3 networks have 'default' first.
        """
        args = ["get", "networks.networking.gke.io", "-o", "json"]
        try:
            output = self._run_kubectl(args)
            if not output:
                return [], []
            data = json.loads(output)
            items = data.get("items", [])
            if not isinstance(items, list):
                logger.debug(
                    "Unexpected network discovery output format: "
                    "'items' is not a list"
                )
                return [], []

            networks = defaultdict(list)

            for item in items:
                metadata = item.get("metadata", {})
                spec = item.get("spec", {})
                name = metadata.get("name")
                net_type = spec.get("type")

                if name and net_type:
                    networks[net_type].append(name)

            l3_networks = networks["L3"]
            device_networks = networks["Device"]

            if "default" not in l3_networks:
                logger.warning(
                    "Default L3 network not found. Discovered L3: %s",
                    ", ".join(l3_networks) or "none",
                )
                return [], []
            l3_networks.sort(key=lambda n: (n != "default", n))
            device_networks.sort()

            logger.debug(
                "Discovered L3 networks: %s, Device networks: %s",
                l3_networks,
                device_networks,
            )
            return l3_networks, device_networks

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse network discovery JSON: %s", e)
            return [], []
        except Exception as e:
            logger.warning("Unexpected error during network discovery: %s", e)
            return [], []

    def get_nodes_under_test(self) -> list[str]:
        if self._resolved_nodes is not None:
            return self._resolved_nodes

        if self.nodes_arg:
            self._resolved_nodes = [
                n.strip() for n in self.nodes_arg.split(",") if n.strip()
            ]
            return self._resolved_nodes

        try:
            args = ["get", "nodes"]
            if self.nodepools_arg:
                pools = [
                    p.strip()
                    for p in self.nodepools_arg.split(",")
                    if p.strip()
                ]
                if pools:
                    pools_str = ",".join(pools)
                    args.extend(
                        [
                            "-l",
                            f"cloud.google.com/gke-nodepool in ({pools_str})",
                        ]
                    )
            else:
                args.extend(["-l", "cloud.google.com/gke-nodepool!=system"])

            args.extend(["-o", "jsonpath={.items[*].metadata.name}"])

            output = self._run_kubectl(args)
            self._resolved_nodes = [n for n in output.split() if n]
            logger.debug("Resolved nodes: %s", self._resolved_nodes)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(
                "Failed to automatically discover nodes using kubectl: %s", e
            )
            self._resolved_nodes = []

        return self._resolved_nodes

    def get_topology(self) -> dict[str, dict[str, list[str]]]:
        """Discovers network topology using kubectl to get node labels.

        # TODO: This uses GCE topology labels (gce-topology-block).

        Returns a dictionary mapping block names to dicts of subblocks to lists
        of nodes.
        """
        nodes = self.get_nodes_under_test()
        if not nodes:
            return {}

        try:
            args = ["get", "nodes"] + nodes + ["-o", "json"]
            output = self._run_kubectl(args)
            if not output:
                return {}

            data = json.loads(output)
            items = (
                data.get("items", [data])
                if data.get("kind") == "List"
                else [data]
            )

            topology = defaultdict(lambda: defaultdict(list))
            for item in items:
                metadata = item.get("metadata", {})
                name = metadata.get("name")
                labels = metadata.get("labels", {})

                block = labels.get(
                    "cloud.google.com/gce-topology-block", "default-block"
                )
                subblock = labels.get(
                    "cloud.google.com/gce-topology-subblock", "default-subblock"
                )

                if name:
                    topology[block][subblock].append(name)

            # Convert defaultdict to regular dict
            return {b: dict(s) for b, s in topology.items()}

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("Failed to get node topology via kubectl: %s", e)
            return {}
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse kubectl output as JSON: %s", e)
            return {}

    def write_ramble_config(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None,
        machine_type: str,
        timeout: int | None = None,
        deadline: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self.workspace_config_dir = workspace_config_dir
        nodes = self.get_nodes_under_test()
        logger.debug("Configuring workspace with kubernetes_template.yaml")
        template_path = pkg_resources.files("chs.schedulers").joinpath(
            "kubernetes_template.yaml"
        )
        with pkg_resources.as_file(template_path) as p:
            with open(p, "r") as f:
                config = yaml.safe_load(f)

        if resources.is_tpu(machine_type):
            l3_networks = ["default"]
            device_networks = []
        else:
            l3_networks, device_networks = self._discover_networks()
            if not l3_networks or not device_networks:
                logger.warning(
                    "Network discovery did not find any Kubernetes network "
                    "resources. Falling back to standard default network "
                    "interface."
                )
                l3_networks = ["default"]
                device_networks = []
        all_networks = l3_networks + device_networks
        all_interfaces = [
            {"interfaceName": f"eth{i}", "network": net_name}
            for i, net_name in enumerate(all_networks)
        ]

        default_interface_name = "eth0"

        variables = config["ramble"]["variables"]
        variables["nodes"] = nodes or []
        variables["machine_type"] = machine_type
        variables["tests"] = tests or []
        variables["default_interface_name"] = default_interface_name
        variables["all_interfaces_json"] = json.dumps(all_interfaces)
        variables["timeout"] = timeout
        variables["deadline"] = deadline

        variables["chs_run_id"] = run_id

        variables["kubernetes_namespace"] = CHS_NAMESPACE

        if resources.is_tpu(machine_type) and nodes:
            topology = self._discover_tpu_topology(nodes[0])
            if topology:
                variables["tpu_topology"] = topology
            variables["tpu_slices"] = self._group_nodes_by_slice(nodes)

        if self._jobset_labels_yaml:
            variables["jobset_labels_yaml"] = self._jobset_labels_yaml
        if self._suspend_jobset == "true":
            variables["suspend_jobset"] = "true"

        logger.debug("Configuration: %r", config)

        self._write_topology_json(workspace_config_dir)

        with open(os.path.join(workspace_config_dir, "ramble.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def check_jobs_running(self, job_ids: list[str]) -> bool:
        """Checks if the given jobs are still running."""
        if not job_ids:
            logger.warning(
                "No job IDs provided to check. Did chs run submit jobs?"
            )
            return False

        still_running = False
        try:
            # Pass all job IDs at once and ignore ones that were already deleted
            args = (
                ["get", "jobset"]
                + job_ids
                + ["-o", "json", "--ignore-not-found"]
            )
            output = self._run_kubectl(args, namespaced=True)

            # If stdout is empty, no jobs were found (all completed and deleted)
            if not output:
                return False

            data = json.loads(output)

            # If a single item is returned, it's not wrapped in a List
            items = (
                data.get("items", [data])
                if data.get("kind") == "List"
                else [data]
            )

            for job_data in items:
                conditions = job_data.get("status", {}).get("conditions", [])

                completed = False
                failed = False

                for cond in conditions:
                    if (
                        cond.get("type") == "Completed"
                        and cond.get("status") == "True"
                    ):
                        completed = True
                    if (
                        cond.get("type") == "Failed"
                        and cond.get("status") == "True"
                    ):
                        failed = True

                if not completed and not failed:
                    still_running = True
                    logger.debug(
                        "JobSet %s is still running.",
                        job_data.get("metadata", {}).get("name", "Unknown"),
                    )

        except subprocess.CalledProcessError as e:
            logger.warning("Failed to run kubectl get jobset: %s", e.stderr)
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.warning("Failed to parse status for JobSets: %s", e)

        return still_running

    def _check_and_update_deadlines(
        self, failed_jobset_ids: list[str], statuses: dict[str, str]
    ) -> None:
        """Batch checks jobs for deadline exceeded and updates statuses."""
        try:
            label_selector = (
                "jobset.sigs.k8s.io/jobset-name in "
                f"({','.join(failed_jobset_ids)})"
            )
            args = [
                "get",
                "jobs",
                "-l",
                label_selector,
                "-o",
                "json",
            ]
            jobs_output = self._run_kubectl(args, namespaced=True)
            if not jobs_output:
                return

            jobs_data = json.loads(jobs_output).get("items", [])
            deadlined_jobsets = set()

            for job in jobs_data:
                jobset_name = (
                    job.get("metadata", {})
                    .get("labels", {})
                    .get("jobset.sigs.k8s.io/jobset-name")
                )
                if not jobset_name or jobset_name in deadlined_jobsets:
                    continue

                for condition in job.get("status", {}).get("conditions", []):
                    if (
                        condition.get("type") == "Failed"
                        and condition.get("status") == "True"
                        and condition.get("reason") == "DeadlineExceeded"
                    ):
                        deadlined_jobsets.add(jobset_name)
                        statuses[jobset_name] = JobStatus.DEADLINE.value
                        break
        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
        ) as e:
            logger.debug("Failed to batch check deadlines: %s", e)

    def _evaluate_jobset_status(self, jobset_data: dict) -> str:
        """Evaluates JobSet conditions and fallbacks to active count."""
        status = jobset_data.get("status", {})

        conditions = status.get("conditions", [])
        for condition in conditions:
            condition_type = condition.get("type", "")
            is_true = condition.get("status") == "True"

            if condition_type == "Completed" and is_true:
                return "Completed"
            elif condition_type == "Failed" and is_true:
                return "Failed"

        replicated_jobs = status.get("replicatedJobsStatus", [])
        is_pending = all(
            job.get("active", 0) == 0 and job.get("failed", 0) == 0
            for job in replicated_jobs
        )
        return "PENDING" if is_pending else "RUNNING"

    def get_job_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Gets the status of the given jobs using kubectl."""
        if not job_ids:
            return {}

        statuses = {j: JobStatus.UNKNOWN.value for j in job_ids}

        try:
            args = (
                ["get", "jobset"]
                + job_ids
                + ["-o", "json", "--ignore-not-found"]
            )
            output = self._run_kubectl(args, namespaced=True)
            if not output:
                return statuses
            data = json.loads(output)
            items = (
                data.get("items", []) if data.get("kind") == "List" else [data]
            )

            failed_jobset_ids = []
            for item in items:
                job_id = item.get("metadata", {}).get("name")
                if job_id in statuses:
                    raw_status = self._evaluate_jobset_status(item)
                    if raw_status == "Failed":
                        failed_jobset_ids.append(job_id)
                    statuses[job_id] = JobStatus.from_string(raw_status)

            if failed_jobset_ids:
                self._check_and_update_deadlines(failed_jobset_ids, statuses)
        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
        ) as e:
            logger.warning("Failed to get statuses for JobSets: %s", e)

        return statuses

    @staticmethod
    def _is_tpu_job(job_id: str) -> bool:
        """Checks if the job is a TPU benchmark based on naming convention."""
        return "tpu-" in job_id

    def _get_pod_names(
        self, job_id: str, head_pod_only: bool = False
    ) -> list[str]:
        """Queries Kubernetes for pod names."""
        if head_pod_only:
            args = [
                "get",
                "pods",
                "-l",
                f"jobset.sigs.k8s.io/jobset-name={job_id},batch.kubernetes.io/job-completion-index=0",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ]
        else:
            args = [
                "get",
                "pods",
                "-l",
                f"jobset.sigs.k8s.io/jobset-name={job_id}",
                "-o",
                "jsonpath={.items[*].metadata.name}",
            ]
        try:
            output = self._run_kubectl(args, namespaced=True)
            return [p for p in output.split() if p]
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to query pods for JobSet %s: %s", job_id, e)
            return []

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed Kubernetes job."""
        logger.debug("Fetching logs for JobSet %s to %s", job_id, log_path)
        try:
            is_tpu = self._is_tpu_job(job_id)
            pod_names = self._get_pod_names(job_id, head_pod_only=not is_tpu)

            if not pod_names:
                logger.warning(
                    "No pods found to fetch logs for JobSet %s.", job_id
                )
                return

            logger.debug(
                "Fetching logs from %d pods: %s", len(pod_names), pod_names
            )

            with open(log_path, "w") as f:
                for pod_name in pod_names:
                    subprocess.run(
                        ["kubectl", "logs", pod_name, "-n", CHS_NAMESPACE],
                        stdout=f,
                        check=True,
                        text=True,
                    )
            logger.debug("Logs saved to %s", log_path)

        except subprocess.CalledProcessError as e:
            logger.error("Failed to fetch logs for JobSet %s: %s", job_id, e)
        except OSError as e:
            logger.error("Failed to write log file %s: %s", log_path, e)

    def parse_job_ids(self, output: str) -> list[str]:
        """Parses job IDs from the experiment execution output."""
        regex = (
            r"jobset[\.a-zA-Z0-9-]*?/"
            r"([a-zA-Z0-9.-]+)\s+(?:created|configured|unchanged)"
        )
        return re.findall(regex, output)

    def cleanup_scheduler_resources(
        self,
        workspace_name: str | None = None,
        run_entry: dict[str, Any] | None = None,
    ) -> None:
        """Cleans up scheduler-specific resources like Kueue configurations."""
        if not run_entry and workspace_name and self._app_data:
            run_entry = self._app_data.get_run(workspace_name) or {}
        else:
            run_entry = run_entry or {}

        workspace_path = run_entry.get("workspace_path")
        config_dir = self.workspace_config_dir
        if not config_dir and workspace_path:
            config_dir = os.path.join(workspace_path, "configs")

        kueue_config_path = ""
        if config_dir:
            kueue_config_path = os.path.join(config_dir, "kueue-configs.yaml")

        file_exists = kueue_config_path and os.path.exists(kueue_config_path)

        if file_exists:
            kueue_mgr = KueueManager(self)
            kueue_mgr.cleanup(kueue_config_path)
        else:
            logger.debug(
                "Kueue configuration file not found at %s; skipping deletion.",
                kueue_config_path,
            )

        job_ids = list(run_entry.get("job_entries", {}).keys())
        self._delete_job_secrets(job_ids)

        self._delete_namespace()

    def _cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancels the given JobSets using kubectl delete."""
        if not job_ids:
            logger.warning("No job IDs to cancel.")
            return

        try:
            args = ["delete", "jobset"] + job_ids + ["--ignore-not-found"]
            self._run_kubectl(args, namespaced=True)
            logger.debug("Successfully deleted JobSets: %s", ", ".join(job_ids))
        except FileNotFoundError:
            logger.warning("kubectl command not found. Cannot cancel jobs.")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to delete JobSets. Error: %s", e.stderr)
            click.echo(f"Failed to delete JobSets: {e.stderr}")

        self._delete_job_secrets(job_ids)

    def cancel_jobs_by_prefix(self, prefix: str) -> None:
        """Cancels all JobSets matching the specified prefix natively."""
        logger.debug("Attempting native cancel for prefix: %s", prefix)
        # Match the sanitized prefix as a substring in the JobSet name.
        sanitized_match = prefix.replace("_", "-").replace(".", "-").lower()

        try:
            # List all JobSets
            output = self._run_kubectl(
                ["get", "jobset", "-o", "name"], namespaced=True
            )
            if not output:
                self._delete_namespace()
                return

            job_ids = []
            for line in output.splitlines():
                if sanitized_match in line:
                    job_name = line.split("/")[-1]
                    job_ids.append(job_name)

            if job_ids:
                logger.info(
                    "Found %d JobSets to cancel matching %s.",
                    len(job_ids),
                    prefix,
                )
                self._cancel_jobs(job_ids)
            else:
                logger.debug("No matching JobSets found for %s.", prefix)

            self._delete_namespace()
        except FileNotFoundError:
            logger.warning(
                "kubectl command not found. Cannot perform native cancel."
            )
        except Exception as e:
            logger.warning("Native cancel failed: %s", e)
