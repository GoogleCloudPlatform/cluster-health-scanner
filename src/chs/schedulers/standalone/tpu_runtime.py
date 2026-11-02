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

"""Builds and locates the virtualenv for standalone TPU tests."""

import logging
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import platformdirs

from chs import exceptions

logger = logging.getLogger(__name__)

AMB_REPO = "https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git"
AMB_COMMIT = "ca2ec4ffd88eb1bd4cc60dc7bc484f2339482308"
IMAGE_ROOT = pathlib.Path("/opt/chs/runtimes")
_MARKER = ".amb_commit"


def _is_complete(path: pathlib.Path) -> bool:
    """Returns whether `path` is built for `AMB_COMMIT`."""
    marker = path / _MARKER
    return marker.is_file() and marker.read_text().strip() == AMB_COMMIT


def _user_root() -> pathlib.Path:
    return pathlib.Path(platformdirs.user_data_dir(appname="chs")) / "runtimes"


def find_runtime() -> pathlib.Path | None:
    """Returns the built TPU runtime that ensure_runtime() uses, or None."""
    for root in (IMAGE_ROOT, _user_root()):
        if _is_complete(root / "tpu"):
            return root / "tpu"
    return None


def ensure_runtime(root: pathlib.Path | None = None) -> pathlib.Path:
    """Returns the TPU runtime path, building it if missing or outdated."""
    if root is None:
        found = find_runtime()
        if found is not None:
            return found
        root = _user_root()
    path = root / "tpu"
    if _is_complete(path):
        return path
    if path.exists():
        shutil.rmtree(path)

    logger.info("Building TPU runtime in %s.", path)
    venv = path / "venv"
    code = path / "code"
    try:
        path.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q", code])
        _run(
            [
                "git",
                "-C",
                code,
                "fetch",
                "-q",
                "--depth",
                "1",
                AMB_REPO,
                AMB_COMMIT,
            ]
        )
        _run(["git", "-C", code, "checkout", "-q", "FETCH_HEAD"])
        _run(
            [
                sys.executable,
                "-m",
                "uv",
                "venv",
                "--no-cache",
                "--python",
                sys.executable,
                venv,
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--no-cache",
                "--compile-bytecode",
                "--python",
                venv,
                "-r",
                code / "requirements.txt",
            ]
        )
        (path / _MARKER).write_text(AMB_COMMIT + "\n")
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise
    return path


def _run(cmd: list[Any]) -> None:
    """Runs a build command."""
    try:
        subprocess.run(
            [str(arg) for arg in cmd],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise exceptions.TpuRuntimeException(
            "TPU runtime build failed: "
            f"{exceptions.format_subprocess_error(e)}"
        ) from e
