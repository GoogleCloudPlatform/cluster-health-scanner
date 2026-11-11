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

"""Scheduler that tests the VM CHS runs on."""

import fcntl
import importlib.resources as pkg_resources
import importlib.util
import logging
import os
import signal
from typing import Any

import yaml

from chs.exceptions import (
    ChsCancellationException,
    ChsSchedulerException,
    HostCheckException,
)
from chs.schedulers.base import BaseScheduler
from chs.schedulers.standalone import host
from chs.schedulers.standalone.job_spec import load_job_spec
from chs.schedulers.standalone.native_backend import NativeBackend
from chs.schedulers.standalone.runner import SequentialRunner
from chs.testkit import config as chs_config

logger = logging.getLogger(__name__)

LOCK_PATH = "/run/lock/chs-standalone.lock"
TPU_RUNTIME = "tpu"
BENCHMARKS_PACKAGE = "accelerator_microbenchmarks"

_LOCK_OPEN_FLAGS = os.O_NOFOLLOW | os.O_CLOEXEC


def _open_existing_lock_file(path: str) -> tuple[int, bool]:
    try:
        return os.open(path, _LOCK_OPEN_FLAGS | os.O_RDWR), True
    except PermissionError:
        return os.open(path, _LOCK_OPEN_FLAGS | os.O_RDONLY), False


def _open_lock_file(path: str) -> tuple[int, bool]:
    """Opens or creates the lock file; returns (fd, writable)."""
    try:
        return _open_existing_lock_file(path)
    except FileNotFoundError:
        pass
    try:
        fd = os.open(
            path, _LOCK_OPEN_FLAGS | os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o666
        )
    except FileExistsError:
        return _open_existing_lock_file(path)
    os.fchmod(fd, 0o666)
    return fd, True


class StandaloneScheduler(BaseScheduler):
    """Runs a run's tests one at a time on the local VM."""

    submits_directly = True
    _active: "StandaloneScheduler | None" = None

    def __init__(self):
        super().__init__()
        StandaloneScheduler._active = self
        self.nodes_arg: str | None = None
        self._instance: host.GceInstance | None = None
        self._resolved_nodes: list[str] | None = None
        self._lock_fd: int | None = None
        self._lock_writable = False
        self._runner: SequentialRunner | None = None
        self._run_id: str | None = None

    @property
    def name(self) -> str:
        return "standalone"

    def _gce_instance(self) -> host.GceInstance:
        if self._instance is None:
            self._instance = host.gce_instance()
        return self._instance

    def get_cluster_name(self) -> str | None:
        """Returns the VM's name."""
        return self._gce_instance().name

    def resolve_raw_nodes(self) -> list[str]:
        """Returns the local VM."""
        return [self._gce_instance().name]

    def detect_nodes_hardware(self, nodes: list[str]) -> list[str]:
        """Returns the VM's machine type from the metadata server."""
        instance = self._gce_instance()
        self.node_to_instance_id[instance.name] = instance.instance_id
        if instance.zone:
            self.node_to_zone[instance.name] = instance.zone
        if instance.project_id:
            self.project_id = instance.project_id
        return [instance.machine_type]

    def get_topology(self) -> dict[str, dict[str, list[str]]]:
        return {}

    def _lock_host(self) -> bool:
        """Takes the host-wide lock for this process; False if it is held."""
        try:
            fd, writable = _open_lock_file(host.host_path(LOCK_PATH))
        except OSError as e:
            raise HostCheckException(
                f"Could not open the host lock {LOCK_PATH}: {e}."
            ) from e
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self._lock_fd = fd
        self._lock_writable = writable
        return True

    def _record_run_id(self, run_id: str) -> None:
        """Writes the run ID into the held lock file for `chs cancel`."""
        if self._lock_fd is None or not self._lock_writable:
            return
        try:
            os.ftruncate(self._lock_fd, 0)
            if run_id:
                os.pwrite(self._lock_fd, f"{run_id}\n".encode(), 0)
        except OSError as e:
            logger.debug("Could not record the run ID in %s: %s", LOCK_PATH, e)

    def _active_run_id(self) -> str:
        try:
            with open(host.host_path(LOCK_PATH)) as f:
                return f.read(256).strip()
        except OSError:
            return ""

    def _check_node(self, node: str) -> None:
        """Runs the node check; raises HostCheckException to skip the node."""
        host.check_host_access()
        locked = self._lock_host()
        try:
            if not locked:
                raise HostCheckException(
                    f"Another CHS process holds {LOCK_PATH} on the VM."
                )
            self._record_run_id("")
            if not host.tpu_devices():
                raise HostCheckException(
                    "The VM has no TPU devices (/dev/vfio/<n>)."
                )
            host.check_tpus_idle()
            host.check_tpu_access()
            if importlib.util.find_spec(BENCHMARKS_PACKAGE) is None:
                raise HostCheckException(
                    "The TPU test dependencies are not installed. CHS installs"
                    " them only on Linux x86_64 with Python 3.12."
                )
        except BaseException:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            raise

    def get_nodes_under_test(self) -> list[str]:
        """Returns the local VM if it passes the node check, else []."""
        if self._resolved_nodes is None:
            node = self.resolve_raw_nodes()[0]
            try:
                self._check_node(node)
                self._resolved_nodes = [node]
            except HostCheckException as e:
                logger.warning("Skipping node '%s': %s", node, e)
                self._resolved_nodes = []
        return self._resolved_nodes

    def find_runs_targeting_nodes(
        self, nodes: list[str]
    ) -> dict[str, list[str]]:
        return {}

    def write_ramble_config(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None,
        machine_type: str,
        deadline: int | None = None,
        run_id: str | None = None,
        images_dir: str | None = None,
    ) -> None:
        """Writes the Ramble config and readies the native backend."""
        self._run_id = run_id
        nodes = self.get_nodes_under_test()
        if nodes:
            if run_id:
                self._record_run_id(run_id)
            chip_count = chs_config.load_platform(machine_type)["variables"][
                "tpu_chips_per_node"
            ]
            self._runner = SequentialRunner(
                {TPU_RUNTIME: NativeBackend(chip_count=chip_count)}
            )
        template_path = pkg_resources.files("chs.schedulers").joinpath(
            "standalone_template.yaml"
        )
        with pkg_resources.as_file(template_path) as p:
            with open(p) as f:
                config = yaml.safe_load(f)
        variables = config["ramble"]["variables"]
        variables["nodes"] = nodes
        if nodes:
            variables["node"] = nodes[0]
        variables["machine_type"] = machine_type
        variables["tests"] = tests or []
        variables["chs_run_id"] = run_id
        if deadline is not None:
            variables["deadline"] = deadline
        self._write_topology_json(workspace_config_dir)
        with open(os.path.join(workspace_config_dir, "ramble.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _require_runner(self) -> SequentialRunner:
        if self._runner is None:
            raise ChsSchedulerException(
                "No experiments were submitted on this VM."
            )
        return self._runner

    def submit_experiments(self, experiment_dirs: list[str]) -> dict[str, str]:
        """Queues the experiments' job specs; poll_jobs() starts them."""
        runner = self._require_runner()
        specs = {d: load_job_spec(d) for d in experiment_dirs}
        runner.submit(list(specs.values()))
        return {spec.job_name: d for d, spec in specs.items()}

    def extract_job_id(self, experiment_dir: str) -> str:
        return load_job_spec(experiment_dir).job_name

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        """Finishes and starts jobs, then returns their statuses."""
        self._require_runner().poll()
        return self.get_job_statuses(job_ids)

    def get_job_statuses(self, job_ids: list[str]) -> dict[str, str]:
        return self._require_runner().statuses(job_ids)

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        self._require_runner().fetch_log(job_id, log_path)

    def cancel_jobs(self, job_ids: list[str]) -> None:
        if self._runner is not None:
            self._runner.cancel(job_ids)

    def cancel_run(self, run_id: str) -> bool:
        """Cancels the run's jobs, or stops it if another process runs it."""
        if self._runner is not None:
            return run_id == self._run_id and self._runner.cancel_all()
        return self._stop_lock_holder(run_id)

    def cancel_all_runs(self) -> None:
        """Stops the active run's jobs, or the CHS run holding the VM."""
        if self._runner is not None:
            self._runner.cancel_all()
        else:
            self._stop_lock_holder()

    def _stop_lock_holder(self, run_id: str | None = None) -> bool:
        """Sends SIGTERM to the lock holder, if it runs `run_id` (or any)."""
        if self._lock_fd is not None:
            return False
        host.check_host_access()
        try:
            locked = self._lock_host()
        except HostCheckException as e:
            raise ChsCancellationException(str(e)) from e
        if locked:
            os.close(self._lock_fd)
            self._lock_fd = None
            return False
        pid = self._lock_holder()
        if pid is None or pid == os.getpid():
            return False
        if run_id is not None:
            active = self._active_run_id()
            if active != run_id:
                if active:
                    logger.warning(
                        "Run '%s' isn't active on this VM; the active run is"
                        " '%s'. Use `chs cancel %s` or `chs cancel --all`.",
                        run_id,
                        active,
                        active,
                    )
                else:
                    logger.warning(
                        "Run '%s' isn't active on this VM; the active run"
                        " hasn't recorded its ID yet. Use `chs cancel --all`.",
                        run_id,
                    )
                return False
        logger.info("Stopping the CHS run in process %d.", pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except PermissionError as e:
            raise ChsCancellationException(
                f"The CHS run on this VM is process {pid}, which this user"
                f" can't stop ({e}). Rerun `chs cancel` as its user or"
                " with sudo."
            ) from e
        return True

    def _lock_holder(self) -> int | None:
        """Returns the PID holding the flock on LOCK_PATH, from /proc/locks."""
        try:
            st = os.stat(host.host_path(LOCK_PATH))
            with open(host.host_path("/proc/locks")) as f:
                lines = f.readlines()
        except OSError:
            return None
        lock_id = (
            f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
        )
        for line in lines:
            fields = line.split()
            if (
                len(fields) >= 6
                and fields[1] == "FLOCK"
                and fields[5] == lock_id
                and fields[4].isdigit()
            ):
                return int(fields[4])
        return None
