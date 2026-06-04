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

    def _resolve_project_id(self, node: str) -> str:
        return "dummy-project"

    @property
    def name(self) -> str:
        return self._name

    def get_nodes_under_test(self) -> list[str]:
        return []

    def get_topology(self) -> dict[str, dict[str, list[str]]]:
        """Determine the network topology of the cluster."""
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
        p = pkg_resources.files(chs).joinpath("ramble.yaml")
        with pkg_resources.as_file(p) as file_path:
            shutil.copy(
                file_path, os.path.join(workspace_config_dir, "ramble.yaml")
            )
            logger.debug(
                "Copied default ramble.yaml to %s", workspace_config_dir
            )

    def check_jobs_running(self, job_ids: list[str]) -> bool:
        """Checks if the given jobs are still running."""
        return False

    def parse_job_ids(self, output: str) -> list[str]:
        """Parses job IDs from the experiment execution output."""
        return []

    def cancel_jobs_by_prefix(self, prefix: str) -> None:
        """Cancels all jobs matching the specified prefix."""
        pass

    def get_job_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Gets the status of the given jobs."""
        return {jid: "UNKNOWN" for jid in job_ids}

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed job."""
        pass
