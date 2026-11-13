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

import fcntl
import json
import logging
import os
import pathlib
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Generator, IO, TypeVar

_JsonContainer = TypeVar("_JsonContainer", dict, list)

logger = logging.getLogger(__name__)

# Cache umask at import time to avoid thread-safety issues with os.umask()
_UMASK = os.umask(0)
os.umask(_UMASK)


@contextmanager
def exclusive_lock(
    lock_path: str | pathlib.Path,
) -> Generator[None, None, None]:
    """Holds an exclusive advisory lock on `lock_path` for the block.

    The lock serializes processes and threads alike, since every entry opens
    its own file description. It is released when the block exits, even on
    error, and by the kernel if the process dies. Not reentrant: nesting it
    on the same path from one thread deadlocks.

    Args:
        lock_path: Path of the lock file. Created if missing, never removed.
    """
    fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


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


def load_json(
    path: str | pathlib.Path,
    expected_type: type[_JsonContainer],
    *,
    is_valid: Callable[[Any], bool] | None = None,
    warn: bool = False,
) -> _JsonContainer:
    """Reads a JSON object or array, tolerating absent or malformed files.

    Args:
        path: File to read.
        expected_type: `dict` or `list`; anything else in the file yields an
            empty one.
        is_valid: Optional filter applied to each list entry, or to each
            (key, value) item of a dict; rejected entries are dropped.
        warn: Log a warning (instead of debug) when the file exists but
            cannot be used.

    Returns:
        The parsed container, or an empty `expected_type()` when the file is
        missing, unreadable, not JSON, or of another type.
    """
    if not os.path.isfile(path):
        return expected_type()
    log = logger.warning if warn else logger.debug
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        log("Ignoring unreadable JSON file %s: %s", path, e)
        return expected_type()
    if not isinstance(data, expected_type):
        log("Ignoring %s: expected a JSON %s", path, expected_type.__name__)
        return expected_type()
    if is_valid is None:
        return data
    if isinstance(data, dict):
        return dict(item for item in data.items() if is_valid(item))
    return [entry for entry in data if is_valid(entry)]
