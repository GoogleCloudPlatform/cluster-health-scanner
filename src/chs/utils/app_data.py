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

"""App data module for CHS.

This module provides the ChsAppData class, which manages the persistence of
CHS run metadata, configuration, and workspace directories.
"""

import json
import logging
import os
import pathlib
from typing import Any

from chs.utils import file_utils

import chs.exceptions as ex
import platformdirs
from chs.utils.lifecycle import JobStatus

logger = logging.getLogger(__name__)


class ChsAppData:
    """Manages CHS app data for runs, configuration, and workspaces."""

    def __init__(self, data_dir: str | None = None) -> None:
        """Initializes the ChsAppData.

        Args:
          data_dir: Optional custom directory for app data files. If None, uses
            the default user data directory.
        """
        if data_dir:
            self._data_dir = pathlib.Path(data_dir)
        else:
            self._data_dir = pathlib.Path(
                platformdirs.user_data_dir(appname="chs")
            )

        self._ensure_data_dir()
        self._runs_file = self._data_dir / "runs.json"
        self._metadata_file = self._data_dir / "metadata.json"

    def _ensure_data_dir(self) -> None:
        """Ensures the data directory exists."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(self._data_dir, os.W_OK):
                raise OSError(f"Directory {self._data_dir} is not writable")
        except OSError as e:
            logger.debug(
                "Could not create/access data dir at %s: %s. "
                "Falling back to /tmp/chs",
                self._data_dir,
                e,
            )
            self._data_dir = pathlib.Path("/tmp/chs")
            self._data_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "Could not use default data directory for CHS data. "
                "Falling back to /tmp/chs, data may be deleted by the OS."
            )

    def _load_json(self, file_path: pathlib.Path) -> dict[str, Any]:
        """Loads JSON data from a file."""
        try:
            if file_path.exists():
                with open(file_path, "r") as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(
                "Failed to read the internal data file at '%s'. Error: %s",
                file_path,
                e,
            )
        return {}

    def _save_json(self, file_path: pathlib.Path, data: dict[str, Any]) -> None:
        """Saves dictionary data to a JSON file."""
        try:
            with file_utils.atomic_open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning(
                "Failed to persist state to the internal "
                "data file at '%s'. Error: %s",
                file_path,
                e,
            )

    def set_storage_config(self, config: dict) -> None:
        """Sets the storage configuration in the app data."""
        metadata = self._load_json(self._metadata_file)
        metadata["storage"] = config
        self._save_json(self._metadata_file, metadata)
        logger.debug("AppData: Set storage config to %s", config)

    def get_storage_config(self) -> dict:
        """Retrieves the storage configuration from the app data."""
        metadata = self._load_json(self._metadata_file)
        return metadata.get("storage", {})

    def remove_storage_config(self) -> None:
        """Removes the storage configuration from the app data."""
        metadata = self._load_json(self._metadata_file)
        if metadata.pop("storage", None) is not None:
            self._save_json(self._metadata_file, metadata)
            logger.debug("AppData: Removed storage config")

    def update_run(self, run_id: str, run_data: dict[str, Any]) -> None:
        """Updates or adds a run entry in the app data."""
        runs = self._load_json(self._runs_file)

        if run_id in runs:
            if "job_entries" in run_data and "job_entries" in runs[run_id]:
                existing_entries = dict(runs[run_id]["job_entries"])
                new_entries = run_data["job_entries"]
                for job_id, new_entry in new_entries.items():
                    if job_id not in existing_entries:
                        existing_entries[job_id] = new_entry
                    else:
                        curr_status = existing_entries[job_id].get("status")
                        new_status = new_entry.get("status")
                        if (
                            JobStatus.is_terminal(curr_status)
                            and new_status
                            and not JobStatus.is_terminal(new_status)
                        ):
                            new_entry = {
                                k: v
                                for k, v in new_entry.items()
                                if k != "status"
                            }
                        existing_entries[job_id].update(new_entry)
                run_data["job_entries"] = existing_entries
            runs[run_id].update(run_data)
        else:
            runs[run_id] = run_data

        self._save_json(self._runs_file, runs)
        logger.debug("AppData: Updated run %s", run_id)

    def update_run_status(
        self, run_id: str, new_job_statuses: dict[str, str]
    ) -> None:
        """Updates the job statuses for a run.

        Terminal statuses are preserved.
        """
        runs = self._load_json(self._runs_file)
        changed = False
        if run_id not in runs:
            runs[run_id] = {}
            changed = True
        run_entry = runs[run_id]
        if "job_entries" not in run_entry:
            run_entry["job_entries"] = {}
            changed = True
        job_entries = run_entry["job_entries"]
        for job_id, new_status in new_job_statuses.items():
            if job_id not in job_entries:
                job_entries[job_id] = {}
                changed = True
            job_entry = job_entries[job_id]
            # Update the status only if the job is in a non-terminal state.
            curr_status = job_entry.get("status")
            if not JobStatus.is_terminal(curr_status):
                resolved_status = (
                    new_status.value
                    if isinstance(new_status, JobStatus)
                    else str(new_status)
                )
                if curr_status != resolved_status:
                    job_entry["status"] = resolved_status
                    changed = True
        if changed:
            self._save_json(self._runs_file, runs)
            logger.debug("AppData: Updated status for run %s", run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieves a run entry from the app data."""
        runs = self._load_json(self._runs_file)
        return runs.get(run_id)

    def get_all_runs(self) -> dict[str, Any]:
        """Retrieves all run entries from the app data."""
        return self._load_json(self._runs_file)

    def get_latest_runs(self, limit: int = 5) -> dict[str, Any]:
        """Retrieves the latest N run entries from the app data.

        Args:
            limit: The maximum number of runs to return.

        Returns:
            A dictionary of run entries, sorted by submission_time descending.
        """
        runs = self._load_json(self._runs_file)
        sorted_runs = sorted(
            runs.items(),
            key=lambda item: item[1].get("submission_time", ""),
            reverse=True,
        )
        return dict(sorted_runs[:limit])

    def get_workspaces_dir(self) -> str:
        """Returns the default workspaces directory path, creating if necessary.

        Returns:
          The path to the workspaces directory as a string.

        Raises:
          WorkspaceCreationException: If the directory cannot be created.
        """
        try:
            workspaces_dir = self._data_dir / "workspaces"
            workspaces_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Using workspaces directory: %s", workspaces_dir)
            return str(workspaces_dir)
        except OSError as e:
            raise ex.WorkspaceCreationException(
                f"Failed to create workspaces directory at {workspaces_dir}"
            ) from e

    def get_images_dir(self) -> str | None:
        """Returns the default images directory path, creating if necessary.

        Returns:
          The path to the images directory as a string, or None if it cannot
          be created.
        """
        try:
            images_dir = self._data_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Using images directory: %s", images_dir)
            return str(images_dir)
        except OSError as e:
            logger.warning(
                "Failed to create the required image assets "
                "directory at '%s'. Error: %s",
                self._data_dir / "images",
                e,
            )
            return None

    def get_latest_run_id(self) -> str:
        """Finds and returns the user-facing run_id of the most recent run."""
        runs = self.get_all_runs()
        if not runs:
            raise ex.NoPreviousRunsFoundException(
                "No previous runs found in registry."
            )
        sorted_runs = sorted(
            runs.items(),
            key=lambda item: item[1].get("submission_time", ""),
            reverse=True,
        )
        return sorted_runs[0][1].get("run_id")

    def resolve_run_id(self, run_id: str) -> str:
        """Resolves a user-friendly run_id to its physical workspace name."""
        runs = self.get_all_runs()
        for ws_name, run_data in runs.items():
            if run_data.get("run_id") == run_id:
                return ws_name

        raise ex.WorkspaceNotFoundException(
            f"Run ID '{run_id}' not found in registry."
        )
