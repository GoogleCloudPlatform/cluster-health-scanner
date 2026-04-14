"""Base scheduler module."""

import abc
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
    ) -> None:
        """Writes the Ramble configuration file for the workspace.

        Args:
          workspace_config_dir: Directory to write the config file to.
          tests: List of tests to run.
          machine_type: The machine type of the cluster nodes.
        """
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
    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancels the given jobs."""
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
