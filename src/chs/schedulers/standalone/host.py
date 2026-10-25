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

"""Probes the local VM for standalone runs."""

import dataclasses
import errno
import glob
import json
import os
import re
import urllib.error
import urllib.request

from chs import resources
from chs.exceptions import ChsSchedulerException, HostCheckException

HOST_MOUNT = "/host"
CHS_CONTAINER_FLAGS = (
    "--privileged --pid=host -v /:/host"
    " -v chs-local-state:/workspace/.local/share"
)

_METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/?recursive=true"
)
_METADATA_TIMEOUT_SECONDS = 5
_TPU_DEVICE_RE = re.compile(r"/dev/vfio/\d+")


@dataclasses.dataclass(frozen=True)
class GceInstance:
    """The GCE VM CHS runs on."""

    name: str
    instance_id: str
    machine_type: str


def in_container() -> bool:
    """Returns whether CHS runs in a Docker container."""
    return os.path.exists("/.dockerenv")


def host_path(path: str) -> str:
    """Returns where the VM's absolute `path` is visible to CHS."""
    return HOST_MOUNT + path if in_container() else path


def check_host_access() -> None:
    """Checks that a CHS container sees the VM's files and processes."""
    if not in_container():
        return
    try:
        if os.readlink("/proc/self/ns/pid") == os.readlink(
            f"{HOST_MOUNT}/proc/1/ns/pid"
        ):
            return
    except OSError:
        pass
    raise HostCheckException(
        "CHS runs in a container without access to the host. Start the"
        f" CHS container with `docker run {CHS_CONTAINER_FLAGS} ...`."
    )


def gce_instance() -> GceInstance:
    """Reads the VM's identity from the GCE metadata server."""
    request = urllib.request.Request(
        _METADATA_URL, headers={"Metadata-Flavor": "Google"}
    )
    try:
        with urllib.request.urlopen(
            request, timeout=_METADATA_TIMEOUT_SECONDS
        ) as response:
            metadata = json.load(response)
        instance = metadata["instance"]
        return GceInstance(
            name=instance["name"],
            instance_id=str(instance["id"]),
            machine_type=resources.normalize_machine_type(
                instance["machineType"]
            ),
        )
    except (OSError, urllib.error.URLError, ValueError, KeyError) as e:
        raise ChsSchedulerException(
            f"Could not read the GCE metadata server: {e}. The standalone"
            " scheduler runs on a GCE VM."
        ) from e


def tpu_devices() -> list[str]:
    """Returns the VM's TPU devices (/dev/vfio/<n>), as seen by CHS."""
    return sorted(glob.glob(host_path("/dev/vfio/[0-9]*")))


def _holds_tpu_device(fd_dir: str) -> bool:
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        return False
    for fd in fds:
        try:
            target = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if _TPU_DEVICE_RE.fullmatch(target):
            return True
    return False


def check_tpus_idle() -> None:
    """Checks that no process has a TPU (/dev/vfio/<n>) open."""
    proc = host_path("/proc")
    processes = []
    for pid in sorted(filter(str.isdigit, os.listdir(proc)), key=int):
        if not _holds_tpu_device(os.path.join(proc, pid, "fd")):
            continue
        try:
            with open(os.path.join(proc, pid, "comm")) as f:
                processes.append(f"process {pid} ({f.read().strip()})")
        except OSError:
            continue
    if processes:
        raise HostCheckException(
            f"TPUs are busy: used by {', '.join(processes)}."
        )
    for device in tpu_devices():
        try:
            os.close(os.open(device, os.O_RDWR))
        except OSError as e:
            if e.errno == errno.EBUSY:
                raise HostCheckException(
                    f"TPUs are busy: {device} is in use."
                ) from e
