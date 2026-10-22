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

"""Base scheduler module."""

import abc
import json
import os
from typing import Any
import logging
from google.auth.exceptions import GoogleAuthError
from chs.exceptions import ChsSchedulerException
from chs.utils import gcp

logger = logging.getLogger(__name__)

CHS_JOB_PREFIX = "chs-"


class BaseScheduler(abc.ABC):
    """Base class for cluster schedulers."""

    submits_directly = False

    def __init__(self):
        """Initializes the scheduler."""
        self.node_to_instance_id: dict[
            str, str
        ] = {}  # Store node -> GCE instance_id
        self._instances_info: dict[str, dict[str, Any]] | None = None
        self.node_to_zone: dict[str, str] = {}  # Store node -> GCE zone
        self.project_id: str | None = None  # Set by detect_nodes_hardware

    def _resolve_project_id(self, _node: str | None = None) -> str:
        """Resolves the GCP project ID from Application Default Credentials."""
        try:
            _, project = gcp.get_default_credentials()
            if not project:
                raise ChsSchedulerException(
                    "Could not determine GCP project from credentials. Please"
                    " set it using 'gcloud auth application-default login"
                    " --project <PROJECT>' or the GOOGLE_CLOUD_PROJECT"
                    " environment variable."
                )
            return project
        except GoogleAuthError as e:
            raise ChsSchedulerException("Failed to get GCP credentials.") from e

    def _get_instances_info(
        self, nodes: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Returns GCE instance metadata for the given nodes."""
        try:
            if self._instances_info is None:
                project = self._resolve_project_id(nodes[0])
                self.project_id = project
                self._instances_info = gcp.query_gce_instances_info(
                    project, nodes
                )
            return self._instances_info
        except Exception as e:
            logger.debug("Could not retrieve GCE metadata for nodes: %s", e)
            return {}

    def detect_nodes_hardware(self, nodes: list[str]) -> list[str]:
        """Detects machine types and maps instance IDs for the given nodes.

        Stores mappings node_to_instance_id and node_to_zone for future use.
        Returns a list of unique detected machine types (GCE names).
        """
        if not nodes:
            return []

        instances_info = self._get_instances_info(nodes)

        unique_types = set()
        for node in nodes:
            instance_info = instances_info.get(node)
            if instance_info:
                machine_type = instance_info["machine_type"]
                unique_types.add(machine_type)
                self.node_to_instance_id[node] = instance_info["instance_id"]
                if instance_info.get("zone"):
                    self.node_to_zone[node] = instance_info["zone"]
            else:
                logger.warning(
                    "Could not retrieve GCE metadata for node '%s'. Machine"
                    " type defaulted to 'unknown'. Ensure Compute Viewer IAM"
                    " permissions if running on GKE/GCE.",
                    node,
                )
                unique_types.add("unknown")

        return sorted(list(unique_types))

    def get_cluster_name(self) -> str | None:
        """Retrieves the cluster name. Overridden by subclasses.

        Returns:
            The cluster name as a string, or None.
        """
        return None

    def run_preflight_checks(
        self,
        tests: list[dict[str, Any]] | None = None,
        workspace_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Runs preflight checks for the scheduler. Optional to implement."""
        return tests

    def setup_scheduler_resources(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None = None,
        machine_type: str = "",
        workspace_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Sets up scheduler-specific infrastructure resources.

        Optional to implement.
        """
        return tests

    def apply_scheduler_resources(self) -> None:
        """Applies generated scheduler resources to the cluster.

        Optional to implement.
        """
        pass

    @abc.abstractmethod
    def cancel_run(self, run_id: str) -> bool:
        """Cancels the run's jobs and deletes every resource it created.

        Selects the run's jobs and resources by its exact run ID, so other
        runs are never touched.

        Returns:
          Whether the run was active, i.e. had jobs or slices on the cluster.
        """
        pass

    @abc.abstractmethod
    def cancel_all_runs(self) -> None:
        """Cancels every CHS job and deletes every CHS resource."""
        pass

    @abc.abstractmethod
    def find_runs_targeting_nodes(
        self, nodes: list[str]
    ) -> dict[str, list[str]]:
        """Finds CHS runs whose jobs target the given nodes.

        A job targets the nodes it is pinned to while the scheduler still
        lists it: pending, suspended, running, and on Kubernetes also
        finished until the run's teardown deletes it.

        Returns:
          Mapping of run ID to the sorted subset of `nodes` its jobs target.
          Empty when no run targets any of the nodes.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the scheduler."""
        pass

    def get_extra_env(self) -> dict[str, str]:
        """Returns extra environment variables for the scheduler.

        Returns:
            A dictionary of extra environment variables.
        """
        return {}

    @abc.abstractmethod
    def get_topology(self) -> dict[str, dict[str, list[str]]]:
        """Determine the network topology of the cluster."""
        pass

    def allow_interdomain_pairs(self, machine_type: str = "") -> bool:
        """Returns whether the scheduler allows cross-domain node pairs."""
        return True

    def _write_topology_json(self, workspace_config_dir: str) -> None:
        """Writes the topology to topology.json in the workspace directory."""
        workspace_dir = os.path.dirname(workspace_config_dir)
        topo_path = os.path.join(workspace_dir, "topology.json")
        with open(topo_path, "w") as f:
            json.dump(self.get_topology(), f)

    @abc.abstractmethod
    def resolve_raw_nodes(self) -> list[str]:
        """Resolves raw targeted nodes based on CLI arguments purely."""
        pass

    @abc.abstractmethod
    def get_nodes_under_test(self) -> list[str]:
        """Determine which nodes to test based on provided parameters.

        Busy and not-ready nodes are excluded, and so are nodes targeted by
        other CHS runs, which are logged with the runs targeting them.
        """
        pass

    @abc.abstractmethod
    def write_ramble_config(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None,
        machine_type: str,
        deadline: int | None = None,
        run_id: str | None = None,
        images_dir: str | None = None,
    ) -> None:
        """Writes the Ramble configuration file for the workspace.

        Args:
          workspace_config_dir: Directory to write the config file to.
          tests: List of tests to run.
          machine_type: The machine type of the cluster nodes.
          deadline: The deadline for the job in seconds.
          run_id: The user-facing run ID.
        """
        pass

    @abc.abstractmethod
    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancels specific jobs by their IDs."""
        pass

    @abc.abstractmethod
    def extract_job_id(self, experiment_dir: str) -> str:
        """Extracts the job ID from the experiment directory.

        Args:
            experiment_dir: The path to the experiment directory.

        Returns:
            The job ID as a string.
        """
        pass

    @abc.abstractmethod
    def get_job_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Gets the status of the given jobs.

        Args:
            job_ids: A list of job IDs to check.

        Returns:
            A dictionary mapping job IDs to their status (e.g., 'RUNNING',
            'COMPLETED', 'FAILED').
        """
        pass

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        """Starts waiting jobs that can run and returns job statuses."""
        return self.get_job_statuses(job_ids)

    def get_unschedulable_jobs(self, job_ids: list[str]) -> list[str]:
        """Returns active jobs whose pods have timed out waiting to schedule."""
        return []

    def submit_experiments(self, experiment_dirs: list[str]) -> dict[str, str]:
        """Submits experiments without running the Ramble batch scripts."""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed job and saves it to the given path.

        Args:
            job_id: The ID of the job to fetch logs for.
            log_path: The path to save the log file to.
        """
        pass
