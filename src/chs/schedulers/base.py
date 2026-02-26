"""Base scheduler module."""

import abc


class BaseScheduler(abc.ABC):
    """Base class for cluster schedulers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the scheduler."""
        pass

    @abc.abstractmethod
    def get_nodes_under_test(self) -> list[str]:
        """Determine which nodes to test based on provided parameters."""
        pass

    @abc.abstractmethod
    def write_ramble_config(
        self,
        workspace_config_dir: str,
        suite: str,
        tests: list[str] | None,
        machine_type: str,
    ) -> None:
        """Writes the ramble configuration into the given workspace configs
        directory."""
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
