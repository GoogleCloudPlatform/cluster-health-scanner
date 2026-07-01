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

"""Local Path storage provider implementation for CHS."""

import logging
import pathlib
import shutil
import tempfile
from chs.exceptions import ChsValidationError
from chs.storage.base import StorageProvider

logger = logging.getLogger(__name__)


class PathStorageProvider(StorageProvider):
    """Storage provider for local or mounted directory paths."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.target_path = (
            pathlib.Path(config.get("path", "")).expanduser().resolve()
        )

    def _get_full_path(self, relative_path: str) -> pathlib.Path:
        """Helper to resolve relative path against target path."""
        return (self.target_path / relative_path).resolve()

    def upload_file(self, local_src: str, remote_dest: str) -> None:
        """Copies a file to the local storage target."""
        dest_path = self._get_full_path(remote_dest)
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_src, dest_path)
            logger.debug("Copied %s to %s", local_src, dest_path)
        except OSError as e:
            raise OSError(
                f"Failed to copy {local_src} to {dest_path}: {e}"
            ) from e

    def download_file(self, remote_src: str, local_dest: str) -> None:
        """Copies a file from the local storage target."""
        src_path = self._get_full_path(remote_src)
        try:
            shutil.copy2(src_path, local_dest)
            logger.debug("Copied %s to %s", src_path, local_dest)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {src_path}") from e
        except OSError as e:
            raise OSError(
                f"Failed to copy {src_path} to {local_dest}: {e}"
            ) from e

    def validate(self) -> None:
        """Validates writeability of the target path.

        Creates missing directories, writes a temp file, deletes it, and cleans
        up created directories on failure.
        """
        if not self.target_path.parts:
            raise ChsValidationError("Storage path cannot be empty.")

        created_dirs = []
        # Find which directories in the path do not exist yet
        current = self.target_path
        dirs_to_create = []
        while current and current != current.parent:
            if not current.exists():
                dirs_to_create.insert(0, current)
            else:
                break
            current = current.parent

        # Attempt to create missing directories
        try:
            for d in dirs_to_create:
                d.mkdir(exist_ok=True)
                created_dirs.append(d)
                logger.debug("Created validation directory: %s", d)
        except OSError as e:
            raise ChsValidationError(
                f"Failed to create target directory structure: {e}"
            ) from e

        # Write test
        temp_file = None
        try:
            try:
                # Use NamedTemporaryFile to ensure unique name
                with tempfile.NamedTemporaryFile(
                    dir=self.target_path,
                    prefix=".chs-write-test-",
                    suffix=".tmp",
                    delete=False,
                ) as f:
                    temp_file = pathlib.Path(f.name)
                    f.write(b"connectivity-test")
                    f.flush()
                logger.debug("Wrote validation temp file: %s", temp_file)
            except OSError as e:
                # Cleanup created directories on write failure
                self._cleanup_dirs(created_dirs)
                raise ChsValidationError(
                    f"Directory '{self.target_path}' is not writable. "
                    f"Please ensure you have write permissions. "
                    f"Details: {e}"
                ) from e
        finally:
            # Delete test
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.debug("Deleted validation temp file: %s", temp_file)
                except OSError as e:
                    logger.debug(
                        "Failed to clean up temporary validation file %s: %s",
                        temp_file,
                        e,
                    )

    def _cleanup_dirs(self, dirs: list[pathlib.Path]) -> None:
        """Cleans up directories in reverse order if they are empty."""
        for d in reversed(dirs):
            try:
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()
                    logger.debug("Cleaned up empty validation directory: %s", d)
            except OSError as e:
                logger.debug("Failed to clean up directory %s: %s", d, e)

    def get_uri(self, remote_path: str) -> str:
        """Returns the full local absolute path."""
        return str(self._get_full_path(remote_path))
