"""Default scheduler implementation for Cluster Health Scanner."""

import importlib.resources as pkg_resources
import logging
import os
import shutil

import chs
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chs.utils.app_data import ChsAppData


from chs.schedulers.base import BaseScheduler

logger = logging.getLogger(__name__)


class DefaultScheduler(BaseScheduler):
    """Fallback scheduler for non-specific schedulers (e.g.

    bare metal default).
    """

    def __init__(self, name: str, app_data: "ChsAppData | None" = None):
        super().__init__(app_data)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_nodes_under_test(self) -> list[str]:
        return []

    def write_ramble_config(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None,
        machine_type: str,
        timeout: int | None = None,
        deadline: int | None = None,
    ) -> None:
        with pkg_resources.path(chs, "ramble.yaml") as p:
            shutil.copy(p, os.path.join(workspace_config_dir, "ramble.yaml"))
            logger.debug(
                "Copied default ramble.yaml to %s", workspace_config_dir
            )

    def check_jobs_running(self, job_ids: list[str]) -> bool:
        """Checks if the given jobs are still running."""
        return False

    def parse_job_ids(self, output: str) -> list[str]:
        """Parses job IDs from the experiment execution output."""
        return []

    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancels the given jobs."""
        pass

    def get_job_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Gets the status of the given jobs."""
        return {jid: "UNKNOWN" for jid in job_ids}

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed job."""
        pass
