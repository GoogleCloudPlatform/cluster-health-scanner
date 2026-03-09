"""App data module for CHS.

This module provides the ChsAppData class, which manages the persistence of
CHS run metadata, configuration, and workspace directories.
"""

import datetime
import json
import logging
import os
import pathlib
from typing import Any

from chs.exceptions import (
    NoPreviousRunsFoundException,
    WorkspaceCreationException,
    WorkspaceNotFoundException,
)
import platformdirs

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
            logger.warning("Failed to read data file %s: %s", file_path, e)
        return {}

    def _save_json(self, file_path: pathlib.Path, data: dict[str, Any]) -> None:
        """Saves dictionary data to a JSON file."""
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning("Failed to write to data file %s: %s", file_path, e)

    def set_bucket(self, bucket_name: str) -> None:
        """Sets the GCS bucket name in the app data."""
        metadata = self._load_json(self._metadata_file)
        metadata["bucket_name"] = bucket_name
        self._save_json(self._metadata_file, metadata)
        logger.debug("AppData: Set bucket_name to %s", bucket_name)

    def get_bucket(self) -> str | None:
        """Retrieves the GCS bucket name from the app data."""
        metadata = self._load_json(self._metadata_file)
        return metadata.get("bucket_name")

    def update_run(self, run_id: str, run_data: dict[str, Any]) -> None:
        """Updates or adds a run entry in the app data.

        Args:
            run_id: The unique identifier for the run.
            run_data: A dictionary containing run details.
        """
        runs = self._load_json(self._runs_file)

        # Merge with existing data if present
        if run_id in runs:
            runs[run_id].update(run_data)
        else:
            runs[run_id] = run_data

        self._save_json(self._runs_file, runs)
        logger.debug("AppData: Updated run %s", run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieves a run entry from the app data."""
        runs = self._load_json(self._runs_file)
        return runs.get(run_id)

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
            raise WorkspaceCreationException(
                f"Failed to create workspaces directory at {workspaces_dir}"
            ) from e

    def get_latest_run_id(self) -> str:
        """Finds and returns the most recent run_id from the workspaces.

        Returns:
          The run_id of the most recently created workspace.

        Raises:
          WorkspaceNotFoundException: If workspaces directory does not exist.
          NoPreviousRunsFoundException: If no valid runs are found.
        """
        workspaces_dir = pathlib.Path(self.get_workspaces_dir())
        if not workspaces_dir.exists():
            raise WorkspaceNotFoundException("No workspaces directory found.")

        valid_runs = []
        date_format = "%Y-%m-%d_%H.%M.%S.%f"
        for d in workspaces_dir.iterdir():
            if d.is_dir():
                try:
                    parsed_time = datetime.datetime.strptime(
                        d.name, date_format
                    )
                    valid_runs.append((parsed_time, d))
                except ValueError:
                    continue

        if not valid_runs:
            raise NoPreviousRunsFoundException("No previous runs found.")

        latest_time, latest_dir = max(valid_runs, key=lambda item: item[0])
        return latest_dir.name
