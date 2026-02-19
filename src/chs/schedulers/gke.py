"""GKE scheduler implementation for Cluster Health Scanner."""

import importlib.resources as pkg_resources
import json
import logging
import os
import re
import subprocess
import click

import chs.schedulers
from chs.schedulers.base import BaseScheduler

logger = logging.getLogger(__name__)


class GKEScheduler(BaseScheduler):
    """GKE scheduler implementation."""

    def __init__(
        self,
        nodes: str | None = None,
        nodelist: str | None = None,
        nodepool: str | None = None,
        partition: str | None = None,
    ):
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

    def get_nodes_under_test(self) -> list[str]:
        if self._resolved_nodes is not None:
            return self._resolved_nodes

        if self.nodes_arg:
            self._resolved_nodes = [
                n.strip() for n in self.nodes_arg.split(",") if n.strip()
            ]
            return self._resolved_nodes

        try:
            cmd = [
                "kubectl",
                "get",
                "nodes",
            ]
            if self.nodepool_arg:
                cmd.extend(
                    ["-l", f"cloud.google.com/gke-nodepool={self.nodepool_arg}"]
                )
            else:
                cmd.extend(["-l", "cloud.google.com/gke-nodepool!=system"])

            cmd.extend(
                [
                    "-o",
                    "jsonpath={.items[*].metadata.name}",
                ]
            )
            output = subprocess.run(
                cmd,
                text=True,
                check=True,
                capture_output=True,
            ).stdout.strip()
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
        suite: str,
        tests: list[str] | None,
        machine_type: str,
    ) -> None:
        nodes = self.get_nodes_under_test()
        logger.debug("Configuring workspace with gke_template.yaml")

        with pkg_resources.path(chs.schedulers, "gke_template.yaml") as p:
            with open(p, "r") as f:
                template_content = f.read()

        nodes_str = f"[{', '.join(nodes)}]" if nodes else "[]"
        tests_str = f"[{', '.join(tests)}]" if tests else "[]"

        rendered_content = template_content.format(
            nodes=nodes_str,
            machine_type=machine_type,
            tests=tests_str,
        )
        logger.debug("Configuration: %r", rendered_content)

        with open(os.path.join(workspace_config_dir, "ramble.yaml"), "w") as f:
            f.write(rendered_content)

    def check_jobs_running(self, job_ids: list[str]) -> bool:
        """Checks if the given jobs are still running."""
        if not job_ids:
            logger.warning(
                "No job IDs provided to check. Proceeding with analyzing "
                "logs. Did chs run submit jobs?"
            )
            return False

        still_running = False
        for job_id in job_ids:
            try:
                result = subprocess.run(
                    ["kubectl", "get", "jobset", job_id, "-o", "json"],
                    text=True,
                    check=True,
                    capture_output=True,
                )

                job_data = json.loads(result.stdout)
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
                    logger.debug("JobSet %s is still running.", job_id)
            except subprocess.CalledProcessError:
                # If JobSet is not found, assume it's no longer running (job
                # completed)
                pass
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                logger.warning(
                    "Failed to check status for JobSet %s: %s", job_id, e
                )

        return still_running

    def parse_job_ids(self, output: str) -> list[str]:
        """Parses job IDs from the experiment execution output."""
        # Match kubectl apply output for JobSets.
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

        for job_id in job_ids:
            cmd = ["kubectl", "delete", "jobset", job_id]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.debug("Successfully deleted JobSet: %s", job_id)
            except FileNotFoundError:
                logger.warning("kubectl command not found. Cannot cancel jobs.")
            except subprocess.CalledProcessError as e:
                logger.error(
                    "Failed to delete JobSet %s. Error: %s", job_id, e.stderr
                )
                click.echo(f"Failed to delete JobSet {job_id}: {e.stderr}")
