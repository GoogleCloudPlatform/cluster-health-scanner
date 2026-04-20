"""GKE scheduler implementation for Cluster Health Scanner."""

import importlib.resources as pkg_resources
import json
import logging
import os
import re
import subprocess
import click
from typing import Any, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from chs.utils.app_data import ChsAppData


from chs.schedulers.base import BaseScheduler
from chs.exceptions import NetworkDiscoveryException
from chs.utils.lifecycle import JobStatus

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

    def write_ramble_config(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None,
        machine_type: str,
        timeout: int | None = None,
        deadline: int | None = None,
    ) -> None:
        nodes = self.get_nodes_under_test()
        logger.debug("Configuring workspace with gke_template.yaml")
        template_path = pkg_resources.files("chs.schedulers").joinpath(
            "gke_template.yaml"
        )
        with pkg_resources.as_file(template_path) as p:
            with open(p, "r") as f:
                template_content = f.read()

        if nodes:
            nodes_str = json.dumps(nodes)
        else:
            nodes_str = "[]"

        tests_str = json.dumps(tests) if tests else "[]"

        l3_networks, device_networks = self._discover_networks()
        if not l3_networks or not device_networks:
            raise NetworkDiscoveryException(
                "Network discovery failed on GKE. "
                "Could not identify L3 or Device networks."
            )
        all_networks = l3_networks + device_networks
        all_interfaces = [
            {"interfaceName": f"eth{i}", "network": net_name}
            for i, net_name in enumerate(all_networks)
        ]

        default_interface_name = "eth0"
        all_interfaces_json = json.dumps(all_interfaces)

        rendered_content = template_content.format(
            nodes=nodes_str,
            machine_type=machine_type,
            tests=tests_str,
            default_interface_name=default_interface_name,
            all_interfaces_json=all_interfaces_json,
            timeout=timeout,
            deadline=deadline,
        )
        logger.debug("Configuration: %r", rendered_content)

        with open(os.path.join(workspace_config_dir, "ramble.yaml"), "w") as f:
            f.write(rendered_content)

    def check_jobs_running(self, job_ids: list[str]) -> bool:
        """Checks if the given jobs are still running."""
        if not job_ids:
            logger.warning(
                "No job IDs provided to check. Did chs run submit jobs?"
            )
            return False

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

            still_running = False

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
            args = ["get", "jobs", "-l", label_selector, "-o", "json"]
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

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed GKE job."""
        logger.debug("Fetching logs for JobSet %s to %s", job_id, log_path)
        try:
            # Find the head pod (rank 0)
            # This logic mirrors batch_wait.tpl
            args = [
                "get",
                "pods",
                "-l",
                f"jobset.sigs.k8s.io/jobset-name={job_id},batch.kubernetes.io/job-completion-index=0",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ]
            head_pod = self._run_kubectl(args)

            if not head_pod:
                logger.warning("Head pod not found for JobSet %s.", job_id)
                return

            logger.debug("Found head pod: %s", head_pod)

            # Streaming logs to file
            with open(log_path, "w") as f:
                subprocess.run(
                    ["kubectl", "logs", head_pod],
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
