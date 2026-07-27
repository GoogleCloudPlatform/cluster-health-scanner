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

"""File utilities for the Cluster Health Scanner."""

import os
import pathlib
import shutil
import tempfile
from contextlib import contextmanager
from typing import Generator, IO

# Cache umask at import time to avoid thread-safety issues with os.umask()
_UMASK = os.umask(0)
os.umask(_UMASK)


@contextmanager
def atomic_open(
    file_path: str | pathlib.Path,
    mode: str = "w",
    encoding: str | None = "utf-8",
) -> Generator[IO, None, None]:
    """Context manager for atomic file writes.

    Writes to a temporary file in the same directory and then atomically
    replaces the target file on success. This prevents race conditions where
    another process might read an incomplete or empty file during the write.

    Args:
        file_path: Path to the target file.
        mode: File open mode (e.g., 'w', 'wb').
        encoding: File encoding, defaults to 'utf-8'. Ignored in binary mode.

    Yields:
        An open file object to write to.
    """
    if "w" not in mode:
        raise ValueError(
            f"atomic_open requires a write mode (e.g., 'w'). Got: '{mode}'. "
            "Other modes (like 'r', 'a') are not supported."
        )

    path = str(file_path)
    dir_name = os.path.dirname(path) or "."

    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name)
    try:
        if "b" in mode:
            f = os.fdopen(tmp_fd, mode)
        else:
            f = os.fdopen(tmp_fd, mode, encoding=encoding)
    except Exception:
        os.close(tmp_fd)
        os.remove(tmp_path)
        raise

    try:
        with f:
            yield f
            f.flush()
            os.fsync(f.fileno())

        if os.path.exists(path):
            try:
                shutil.copymode(path, tmp_path)
            except FileNotFoundError:
                pass
        else:
            # Apply standard open() permissions respecting the user's umask
            os.chmod(tmp_path, 0o666 & ~_UMASK)

        # Atomically replace the target file with the temporary one
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temporary file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
