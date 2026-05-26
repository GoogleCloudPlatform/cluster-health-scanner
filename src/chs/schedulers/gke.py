"""GKE scheduler implementation for Cluster Health Scanner."""

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
from chs.exceptions import KueueException
from chs.utils.lifecycle import JobStatus
from chs.preflights.gke_preflight import GkePreflightChecker


def boolean_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:bool", str(data))


yaml.Dumper.add_representer(bool, boolean_representer)

logger = logging.getLogger(__name__)


class GKEScheduler(BaseScheduler):
    """GKE scheduler implementation."""

    def __init__(
        self,
        nodes: str | None = None,
        nodelist: str | None = None,
        nodepool: str | None = None,
        partition: str | None = None,
        app_data: "ChsAppData | None" = None,
    ):
        super().__init__(app_data)
        self.nodes_arg = nodes
        self.nodepool_arg = nodepool
        self._resolved_nodes: list[str] | None = None
        self.workspace_config_dir: str | None = None
        self.workspace_name: str | None = None
        self.kueue_supported: bool = False
        self._jobset_labels_yaml: str = ""
        self._suspend_jobset: str = "false"
        if nodelist:
            logger.warning(
                "nodelist argument is ignored for GKE scheduler. Did you want"
                " to use nodes or nodepool?"
            )
        if partition:
            logger.warning(
                "partition argument is ignored for GKE scheduler. Did you want"
                " to use nodes or nodepool?"
            )

    def run_preflight_checks(
        self,
        tests: list[dict[str, Any]] | None = None,
        workspace_name: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Checks for JobSet and Kueue dependencies on GKE."""
        preflight_checker = GkePreflightChecker(self)
        filtered_tests, kueue_supported = preflight_checker.check(tests)
        self.kueue_supported = kueue_supported
        if workspace_name:
            self.workspace_name = workspace_name
            if self._app_data:
                run_entry = self._app_data.get_run(workspace_name) or {}
                run_entry.update({"run_id": workspace_name})
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
                    GkePreflightChecker(self)._filter_tests_without_kueue(tests)
                    or []
                )
                return filtered_tests

        return tests

    @property
    def name(self) -> str:
        return "gke"

    def _run_kubectl(self, args: list[str]) -> str:
        """Runs a kubectl command and returns filtered output."""
        cmd = ["kubectl"] + args
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    def _discover_tpu_topology(self, node_name: str) -> str:
        """Queries node labels to discover TPU topology."""
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

    def _discover_networks(self) -> tuple[list[str], list[str]]:
        """Discovers all network names using kubectl and groups them by type.

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
            if self.nodepool_arg:
                args.extend(
                    ["-l", f"cloud.google.com/gke-nodepool={self.nodepool_arg}"]
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
        workspace_name: str | None = None,
    ) -> None:
        self.workspace_config_dir = workspace_config_dir
        nodes = self.get_nodes_under_test()
        logger.debug("Configuring workspace with gke_template.yaml")
        template_path = pkg_resources.files("chs.schedulers").joinpath(
            "gke_template.yaml"
        )
        with pkg_resources.as_file(template_path) as p:
            with open(p, "r") as f:
                config = yaml.safe_load(f)

        if machine_type.startswith("tpu"):
            l3_networks = ["default"]
            device_networks = []
        else:
            l3_networks, device_networks = self._discover_networks()
            if not l3_networks or not device_networks:
                logger.warning(
                    "Network discovery did not find any GKE network resources. "
                    "Falling back to standard default network interface."
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

        if machine_type.startswith("tpu") and nodes:
            topology = self._discover_tpu_topology(nodes[0])
            if topology:
                variables["tpu_topology"] = topology

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
            output = self._run_kubectl(args)

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
            jobs_output = self._run_kubectl(args)
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
            output = self._run_kubectl(args)
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
        """Queries GKE for pod names."""
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
            output = self._run_kubectl(args)
            return [p for p in output.split() if p]
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to query pods for JobSet %s: %s", job_id, e)
            return []

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed GKE job."""
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
                        ["kubectl", "logs", pod_name],
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

    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancels the given JobSets using kubectl delete."""
        if not job_ids:
            logger.warning("No job IDs to cancel.")
            return

        try:
            args = ["delete", "jobset"] + job_ids + ["--ignore-not-found"]
            self._run_kubectl(args)
            logger.debug("Successfully deleted JobSets: %s", ", ".join(job_ids))
        except FileNotFoundError:
            logger.warning("kubectl command not found. Cannot cancel jobs.")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to delete JobSets. Error: %s", e.stderr)
            click.echo(f"Failed to delete JobSets: {e.stderr}")

    def cancel_jobs_by_prefix(self, prefix: str) -> None:
        """Cancels all JobSets matching the specified prefix natively."""
        logger.debug("Attempting native cancel for prefix: %s", prefix)
        # Match the sanitized prefix as a substring in the JobSet name.
        sanitized_match = prefix.replace("_", "-").replace(".", "-").lower()

        try:
            # List all JobSets
            output = self._run_kubectl(["get", "jobset", "-o", "name"])
            if not output:
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
                self.cancel_jobs(job_ids)
            else:
                logger.debug("No matching JobSets found for %s.", prefix)
        except FileNotFoundError:
            logger.warning(
                "kubectl command not found. Cannot perform native cancel."
            )
        except Exception as e:
            logger.warning("Native cancel failed: %s", e)
