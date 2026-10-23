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

"""Runs standalone GPU test containers on the VM's Docker daemon."""

import contextlib
import dataclasses
import logging
import os
import re
import subprocess

from chs.exceptions import ChsSchedulerException
from chs.schedulers.standalone import host
from chs.schedulers.standalone.host import GpuAccess
from chs.schedulers.standalone.job_spec import JobSpec
from chs.schedulers.standalone.native import KILL_AFTER_SECONDS

logger = logging.getLogger(__name__)

MANAGED_LABEL = "chs.managed"
RUN_ID_LABEL = "chs.run_id"
CONTAINER_DRIVER_DIR = "/usr/local/nvidia"
_DOCKER_TIMEOUT_SECONDS = 120
_PULL_TIMEOUT_SECONDS = 600
_PERSISTENCED_SOCKET = "/run/nvidia-persistenced/socket"
_VOLUME_ROOT = re.compile(r"/volumes/([^/]+)/_data$")


@dataclasses.dataclass(frozen=True)
class Container:
    name: str
    run_id: str


def _run(
    docker: list[str],
    args: list[str],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout: int = _DOCKER_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [*docker, *args],
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ChsSchedulerException(f"`docker {args[0]}` failed: {e}") from e


def _check(
    docker: list[str], args: list[str], timeout: int = _DOCKER_TIMEOUT_SECONDS
) -> str:
    result = _run(docker, args, timeout=timeout)
    if result.returncode != 0:
        raise ChsSchedulerException(
            f"`docker {args[0]}` failed: {result.stderr.strip()}"
        )
    return result.stdout


def list_containers(
    docker: list[str], run_id: str | None = None
) -> list[Container]:
    """Lists CHS containers, only those of run_id if given."""
    args = ["ps", "-a", "--filter", f"label={MANAGED_LABEL}"]
    if run_id is not None:
        args += ["--filter", f"label={RUN_ID_LABEL}={run_id}"]
    args += [
        "--format",
        f'{{{{.Names}}}}\t{{{{.Label "{RUN_ID_LABEL}"}}}}',
    ]
    return [
        Container(*line.split("\t"))
        for line in _check(docker, args).splitlines()
        if line
    ]


def remove_containers(docker: list[str], names: list[str]) -> None:
    """Force-removes containers; ones already gone are ignored."""
    if not names:
        return
    result = _run(docker, ["rm", "-f", *names])
    errors = [
        line
        for line in result.stderr.strip().splitlines()
        if "no such container" not in line.lower()
    ]
    if result.returncode != 0 and errors:
        raise ChsSchedulerException(
            f"Could not remove containers: {'; '.join(errors)}"
        )


def _unescape_mountinfo(field: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def state_volume(mountinfo: str = "/proc/self/mountinfo") -> str:
    """Returns the name of the Docker volume mounted at CHS_STATE_DIR."""
    state_dir = os.path.normpath(host.CHS_STATE_DIR)
    root = None
    try:
        with open(mountinfo) as f:
            for line in f:
                fields = line.split()
                if len(fields) < 5:
                    continue
                mount_point = os.path.normpath(_unescape_mountinfo(fields[4]))
                if mount_point == state_dir:
                    # The last mount at a path hides the earlier ones.
                    root = _unescape_mountinfo(fields[3])
    except OSError as e:
        raise ChsSchedulerException(f"Could not read {mountinfo}: {e}") from e
    match = _VOLUME_ROOT.search(root) if root else None
    if match is None:
        raise ChsSchedulerException(
            f"The CHS state at {host.CHS_STATE_DIR} must be a named Docker"
            " volume so test containers can mount it; run the CHS container"
            f" with `-v {host.CHS_STATE_VOLUME}:{host.CHS_STATE_DIR}`."
        )
    return match.group(1)


class DockerBackend:
    """Runs job specs as labeled containers."""

    def __init__(self, docker: list[str], gpu: GpuAccess):
        self._docker = docker
        self._gpu = gpu
        self._pull_errors: dict[str, str] = {}
        self._state_volume: str | None = None

    def pull_images(self, specs: list[JobSpec]) -> None:
        """Pulls missing images first; a failed pull fails only its jobs."""
        for image in dict.fromkeys(spec.image for spec in specs):
            if image in self._pull_errors:
                continue
            if _run(self._docker, ["image", "inspect", image]).returncode == 0:
                continue
            logger.info("Pulling image %s.", image)
            try:
                _check(self._docker, ["pull", image], _PULL_TIMEOUT_SECONDS)
            except ChsSchedulerException as e:
                logger.error("Could not pull image %s: %s", image, e)
                self._pull_errors[image] = str(e)

    def _workspace_args(self, run_dir: str) -> list[str]:
        """Mounts the experiment's run dir in the container at the same path."""
        if not host.in_container():
            return ["-v", f"{run_dir}:{run_dir}"]
        state_dir = os.path.normpath(host.CHS_STATE_DIR)
        if os.path.commonpath([run_dir, state_dir]) != state_dir:
            raise ChsSchedulerException(
                f"Run dir {run_dir} is outside {host.CHS_STATE_DIR}, the CHS"
                " container's state volume, so test containers cannot mount it."
            )
        if self._state_volume is None:
            self._state_volume = state_volume()
        # The VM's Docker can't see our paths; mount the state volume by name.
        return [
            "--mount",
            f"type=volume,src={self._state_volume},dst={state_dir}",
        ]

    def _run_args(self, spec: JobSpec) -> list[str]:
        if not (self._gpu.nvidia_runtime or spec.privileged):
            raise ChsSchedulerException(
                f"Job '{spec.job_name}' is not privileged, but mounting the"
                " NVIDIA driver (no NVIDIA container runtime) requires a"
                " privileged container."
            )
        args = [
            "run",
            "-d",
            "--pull=never",
            "--name",
            spec.job_name,
            "--label",
            MANAGED_LABEL,
            "--label",
            f"{RUN_ID_LABEL}={spec.run_id}",
            "--ipc=host",
            "--ulimit",
            "memlock=-1:-1",
            *self._workspace_args(os.path.dirname(spec.script)),
        ]
        if spec.privileged:
            args.append("--privileged")
        if self._gpu.nvidia_runtime:
            args += ["--gpus", "all"]
        else:
            args += ["-v", f"{self._gpu.driver_dir}:{CONTAINER_DRIVER_DIR}:ro"]
            if os.path.exists(host.host_path(_PERSISTENCED_SOCKET)):
                args += ["-v", f"{_PERSISTENCED_SOCKET}:{_PERSISTENCED_SOCKET}"]
        return args + [
            "--entrypoint",
            "timeout",
            spec.image,
            "-k",
            str(KILL_AFTER_SECONDS),
            str(spec.timeout),
            "bash",
            "-c",
            spec.read_script(),
        ]

    def start(self, spec: JobSpec) -> None:
        """Starts the spec's container from an image pull_images() pulled."""
        if spec.image in self._pull_errors:
            raise ChsSchedulerException(
                f"Could not pull image {spec.image}:"
                f" {self._pull_errors[spec.image]}"
            )
        logger.debug("Starting container %s.", spec.job_name)
        _check(self._docker, self._run_args(spec))

    def exit_code(self, spec: JobSpec) -> int | None:
        """Returns the container's exit code, or None while it runs."""
        state = _check(
            self._docker,
            [
                "inspect",
                "--format",
                "{{.State.Status}} {{.State.ExitCode}}",
                spec.job_name,
            ],
        ).split()
        return int(state[1]) if state[0] in ("exited", "dead") else None

    def stop(self, spec: JobSpec) -> None:
        """Kills the container and keeps it for fetch_log()."""
        result = _run(self._docker, ["kill", spec.job_name])
        if result.returncode != 0:
            logger.debug(
                "Could not kill container %s: %s",
                spec.job_name,
                result.stderr.strip(),
            )

    def fetch_log(self, spec: JobSpec, log_path: str) -> None:
        """Writes the container's output to log_path."""
        # A root container may own log_path; unlink it instead of truncating.
        with contextlib.suppress(FileNotFoundError):
            os.remove(log_path)
        with open(log_path, "w") as log:
            result = _run(
                self._docker,
                ["logs", spec.job_name],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise ChsSchedulerException(
                f"Could not read the output of container {spec.job_name}."
            )

    def remove(self, spec: JobSpec) -> None:
        """Removes the spec's container."""
        remove_containers(self._docker, [spec.job_name])
