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
import logging
import os
import re
import resource
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

from chs import resources
from chs.exceptions import ChsSchedulerException, HostCheckException

logger = logging.getLogger(__name__)

HOST_MOUNT = "/host"
CHS_STATE_VOLUME = "chs-local-state"
CHS_STATE_DIR = "/workspace/.local/share"
CHS_CONTAINER_FLAGS = (
    "--privileged --pid=host -v /:/host"
    f" -v {CHS_STATE_VOLUME}:{CHS_STATE_DIR}"
)

_METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/?recursive=true"
)
_METADATA_TIMEOUT_SECONDS = 5
_COMMAND_TIMEOUT_SECONDS = 60

_PACKAGED_DRIVER_ROOTS = ("/var/lib/nvidia", "/home/kubernetes/bin/nvidia")
_DISTRO_LIB_DIRS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
)
_DRIVER_VERSION_RE = re.compile(r"Kernel Module(?: for \S+)?\s+(\d+(?:\.\d+)+)")
_TPU_DEVICE_RE = re.compile(r"/dev/vfio/\d+")


@dataclasses.dataclass(frozen=True)
class GceInstance:
    """The GCE VM CHS runs on."""

    name: str
    instance_id: str
    machine_type: str
    zone: str = ""
    project_id: str = ""


@dataclasses.dataclass(frozen=True)
class GpuAccess:
    """How a container gets the VM's GPUs."""

    nvidia_runtime: bool
    driver_dir: str = ""


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
            zone=instance.get("zone", "").rsplit("/", 1)[-1],
            project_id=metadata.get("project", {}).get("projectId", ""),
        )
    except (OSError, urllib.error.URLError, ValueError, KeyError) as e:
        raise ChsSchedulerException(
            f"Could not read the GCE metadata server: {e}. The standalone"
            " scheduler runs on a GCE VM."
        ) from e


def tpu_devices() -> list[str]:
    """Returns the VM's TPU devices (/dev/vfio/<n>), as seen by CHS."""
    return sorted(glob.glob(host_path("/dev/vfio/[0-9]*")))


def has_gpus() -> bool:
    """Returns whether the VM has NVIDIA GPU devices (/dev/nvidia<n>)."""
    return bool(glob.glob(host_path("/dev/nvidia[0-9]*")))


def check_tpu_access() -> None:
    """Checks that native TPU tests can lock memory and open the TPUs."""
    _, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if os.geteuid() != 0 and hard != resource.RLIM_INFINITY:
        raise HostCheckException(
            f"The memlock limit is {hard} bytes; native TPU tests need root"
            " or an unlimited memlock limit (e.g. `ulimit -l unlimited` via"
            " /etc/security/limits.conf)."
        )
    denied = [
        path
        for path in [host_path("/dev/vfio/vfio"), *tpu_devices()]
        if not os.access(path, os.R_OK | os.W_OK)
    ]
    if denied:
        raise HostCheckException(
            f"No read/write access to TPU devices {', '.join(denied)}."
        )


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


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, 127, "", "not found")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            argv, 124, "", f"timed out after {_COMMAND_TIMEOUT_SECONDS}s"
        )


def _error_line(result: subprocess.CompletedProcess) -> str:
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    return lines[-1] if lines else f"exit code {result.returncode}"


def find_docker() -> list[str]:
    """Returns the command that runs the VM's Docker CLI."""
    if in_container():
        # The CHS image has no Docker CLI; run the VM's, as root on the VM.
        candidates = [["chroot", HOST_MOUNT, "env", "HOME=/root", "docker"]]
    else:
        candidates = [["docker"], ["sudo", "-n", "docker"]]
    errors = []
    for docker in candidates:
        result = _run([*docker, "version", "--format", "{{.Server.Version}}"])
        if result.returncode == 0:
            return docker
        error = _error_line(result)
        if result.returncode == 127:
            error = f"the VM has no Docker CLI ({error})"
        errors.append(f"`{' '.join(docker)}`: {error}")
    raise HostCheckException(f"Docker is unusable: {'; '.join(errors)}.")


def probe_gpu_access(docker: list[str]) -> GpuAccess:
    """Finds how containers get the VM's GPUs."""
    result = _run([*docker, "info", "--format", "{{json .Runtimes}}"])
    if result.returncode != 0:
        raise HostCheckException(
            f"Docker is unusable: `docker info` failed: {_error_line(result)}."
        )
    if "nvidia" in json.loads(result.stdout or "{}"):
        return GpuAccess(nvidia_runtime=True)
    driver_dir = _packaged_driver_root() or _stage_distro_driver()
    return GpuAccess(nvidia_runtime=False, driver_dir=driver_dir)


def _packaged_driver_root() -> str:
    return next(
        (
            root
            for root in _PACKAGED_DRIVER_ROOTS
            if os.access(host_path(f"{root}/bin/nvidia-smi"), os.X_OK)
        ),
        "",
    )


def _nvidia_driver_version() -> str:
    """Returns the loaded NVIDIA kernel module's version."""
    path = host_path("/proc/driver/nvidia/version")
    try:
        with open(path, encoding="utf-8") as f:
            match = _DRIVER_VERSION_RE.search(f.read())
    except OSError as e:
        raise HostCheckException(
            f"NVIDIA driver is unusable: could not read {path}: {e}."
        ) from e
    if not match:
        raise HostCheckException(
            f"NVIDIA driver is unusable: no driver version in {path}."
        )
    return match.group(1)


def _driver_stage_parent() -> str:
    if in_container() or os.geteuid() == 0:
        return "/var/lib/chs"
    return os.path.expanduser("~/.cache/chs")


def _driver_staged(local_dir: str) -> bool:
    return os.path.isfile(
        os.path.join(local_dir, "bin", "nvidia-smi")
    ) and bool(glob.glob(os.path.join(local_dir, "lib64", "libcuda.so*")))


def _stage_distro_driver() -> str:
    """Stages the distro NVIDIA driver for mounting into containers."""
    version = _nvidia_driver_version()
    parent = _driver_stage_parent()
    target = f"{parent}/nvidia-driver-{version}"
    local_target = host_path(target)
    if _driver_staged(local_target):
        return target
    tmp = ""
    try:
        os.makedirs(host_path(parent), exist_ok=True)
        tmp = tempfile.mkdtemp(
            prefix=f"nvidia-driver-{version}.tmp.", dir=host_path(parent)
        )
        os.makedirs(os.path.join(tmp, "bin"))
        lib64 = os.path.join(tmp, "lib64")
        os.makedirs(lib64)
        shutil.copy2(
            host_path("/usr/bin/nvidia-smi"),
            os.path.join(tmp, "bin", "nvidia-smi"),
        )
        for lib_dir in _DISTRO_LIB_DIRS:
            for pattern in ("libcuda.so*", "libnvidia*.so*"):
                for lib in glob.glob(os.path.join(host_path(lib_dir), pattern)):
                    dest = os.path.join(lib64, os.path.basename(lib))
                    if os.path.islink(lib) and "/" not in (
                        link := os.readlink(lib)
                    ):
                        os.symlink(link, dest)
                    else:
                        shutil.copy2(lib, dest)
        os.chmod(tmp, 0o755)
        if not _driver_staged(tmp):
            raise HostCheckException(
                f"NVIDIA driver is unusable: libcuda.so for driver {version}"
                f" not found in {', '.join(_DISTRO_LIB_DIRS)}."
            )
        os.rename(tmp, local_target)
        tmp = ""
    except OSError as e:
        raise HostCheckException(
            f"NVIDIA driver is unusable: staging driver {version} in"
            f" {target} failed: {e}."
        ) from e
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    logger.debug("Staged NVIDIA driver %s in %s.", version, target)
    return target


def check_gpus_idle() -> None:
    """Checks that the NVIDIA driver works and no process uses the GPUs."""
    root = _packaged_driver_root()
    argv = [
        f"{root}/bin/nvidia-smi" if root else "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader",
    ]
    result = _run(["chroot", HOST_MOUNT, *argv] if in_container() else argv)
    if result.returncode != 0:
        raise HostCheckException(
            f"NVIDIA driver is unusable: nvidia-smi failed:"
            f" {_error_line(result)}."
        )
    processes = [
        f"process {pid.strip()} ({name.strip()})"
        for line in result.stdout.splitlines()
        if line.strip()
        for pid, _, name in [line.partition(",")]
    ]
    if processes:
        raise HostCheckException(
            f"GPUs are busy: used by {', '.join(processes)}."
        )


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
