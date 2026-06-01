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
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chs.utils.app_data import ChsAppData


class BaseScheduler(abc.ABC):
    """Base class for cluster schedulers."""

    def __init__(self, app_data: "ChsAppData | None" = None):
        """Initializes the scheduler.

        Args:
          app_data: ChsAppData instance or None.
        """
        self._app_data = app_data

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
    ) -> list[dict[str, Any]] | None:
        """Sets up scheduler-specific infrastructure resources.

        Optional to implement.
        """
        return tests

    def cleanup_scheduler_resources(
        self, workspace_name: str | None = None
    ) -> None:
        """Cleans up scheduler-specific resources. Optional to implement."""
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

    def _write_topology_json(self, workspace_config_dir: str) -> None:
        """Writes the topology to topology.json in the workspace directory."""
        workspace_dir = os.path.dirname(workspace_config_dir)
        topo_path = os.path.join(workspace_dir, "topology.json")
        with open(topo_path, "w") as f:
            json.dump(self.get_topology(), f)

    @abc.abstractmethod
    def get_nodes_under_test(self) -> list[str]:
        """Determine which nodes to test based on provided parameters."""
        pass

    @abc.abstractmethod
    def write_ramble_config(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None,
        machine_type: str,
        timeout: int | None = None,
        deadline: int | None = None,
        run_id: str | None = None,
    ) -> None:
        """Writes the Ramble configuration file for the workspace.

        Args:
          workspace_config_dir: Directory to write the config file to.
          tests: List of tests to run.
          machine_type: The machine type of the cluster nodes.
          timeout: The timeout for the job in seconds.
          deadline: The deadline for the job in seconds.
          run_id: The user-facing run ID.
        """
        pass

    @abc.abstractmethod
    def cancel_jobs_by_prefix(self, prefix: str) -> None:
        """Cancels all jobs matching the specified prefix."""
        pass

    @abc.abstractmethod
    def check_jobs_running(self, job_ids: list[str]) -> bool:
        """Checks if the given jobs are still running."""
        pass

    @abc.abstractmethod
    def parse_job_ids(self, output: str) -> list[str]:
        """Parses job IDs from the experiment execution output."""
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

    @abc.abstractmethod
    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed job and saves it to the given path.

        Args:
            job_id: The ID of the job to fetch logs for.
            log_path: The path to save the log file to.
        """
        pass
