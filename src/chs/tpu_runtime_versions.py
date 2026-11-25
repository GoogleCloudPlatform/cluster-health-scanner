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

"""JAX and libtpu versions that CHS TPU tests run with.

On GKE, TPU pods run the published CHS image, which is built from the pins in
requirements.txt. The installed package does not contain requirements.txt, so
setup.py runs this module at build time to save the pins next to it (load()).
On standalone TPU VMs, the tests run on the packages installed with CHS
(installed()). Standard library only: setup.py runs it before the package's
dependencies are installed.
"""

import dataclasses
import functools
import importlib.metadata
import json
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tpu_runtime_versions.json"
)

_PIN_RE = re.compile(r"^(jax|libtpu)==([^\s;\\]+)", re.MULTILINE)
# libtpu distributions in the order tpu-info's inventory script checks them.
_LIBTPU_DISTRIBUTIONS = ("libtpu-nightly", "libtpu")


@dataclasses.dataclass(frozen=True)
class TpuRuntimeVersions:
    """Versions of the TPU runtime packages."""

    jax: str
    libtpu: str


def parse_requirements(text: str) -> dict[str, str]:
    """Returns the pinned jax and libtpu versions of a requirements file.

    Raises:
        ValueError: If a package is pinned to more than one version, since
            the version the image installs could not be told apart.
    """
    versions: dict[str, str] = {}
    for name, version in _PIN_RE.findall(text):
        if versions.setdefault(name, version) != version:
            raise ValueError(
                f"{name} is pinned to both {versions[name]} and {version}."
            )
    return versions


def write_manifest(requirements_path: str, manifest_path: str) -> None:
    """Saves the jax and libtpu pins of requirements_path to manifest_path."""
    with open(requirements_path, encoding="utf-8") as f:
        versions = parse_requirements(f.read())
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2, sort_keys=True)
        f.write("\n")


@functools.cache
def load() -> TpuRuntimeVersions | None:
    """Returns the versions saved at build time.

    Returns None, so that the Bill of Health leaves the versions out, if the
    file is missing or invalid.
    """
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            data = json.load(f)
        versions = TpuRuntimeVersions(jax=data["jax"], libtpu=data["libtpu"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.debug("The TPU runtime versions are unavailable: %s", e)
        return None
    if not all(isinstance(v, str) and v for v in dataclasses.astuple(versions)):
        logger.debug("The saved TPU runtime versions are invalid: %s", data)
        return None
    return versions


def installed() -> TpuRuntimeVersions | None:
    """Returns the jax and libtpu versions installed with CHS, if both are."""
    try:
        jax = importlib.metadata.version("jax")
    except importlib.metadata.PackageNotFoundError:
        return None
    for name in _LIBTPU_DISTRIBUTIONS:
        try:
            libtpu = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        return TpuRuntimeVersions(jax=jax, libtpu=libtpu)
    return None


if __name__ == "__main__":
    try:
        write_manifest(sys.argv[1], sys.argv[2])
    except (OSError, ValueError) as e:
        sys.exit(f"Could not save the TPU runtime versions: {e}")
