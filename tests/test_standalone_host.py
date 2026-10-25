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

"""Unit tests for chs.schedulers.standalone.host."""

import errno
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
from unittest import mock

from chs.exceptions import ChsSchedulerException, HostCheckException
from chs.schedulers.standalone import host

_PROPRIETARY_VERSION = (
    "NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.183.01  Sun May 12"
    " 19:39:15 UTC 2024\nGCC version:  gcc version 12.2.0\n"
)
_OPEN_VERSION = (
    "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.86.15"
    "  Release Build\n"
)


def _done(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class FakeHostTestCase(unittest.TestCase):
    """Maps VM paths into a temp dir and runs CHS outside a container."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        for target, value in (
            ("host_path", lambda path: self.root + path),
            ("in_container", lambda: False),
        ):
            patcher = mock.patch.object(host, target, side_effect=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write(self, path, content="", mode=0o644):
        local = self.root + path
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "w") as f:
            f.write(content)
        os.chmod(local, mode)
        return local


class PathsAndAccessTest(unittest.TestCase):
    def test_on_vm_and_in_container(self):
        with mock.patch.object(host.os.path, "exists", return_value=False):
            self.assertFalse(host.in_container())
            self.assertEqual(host.host_path("/dev/vfio"), "/dev/vfio")
            host.check_host_access()

        with mock.patch.object(host.os.path, "exists", return_value=True):
            self.assertTrue(host.in_container())
            self.assertEqual(host.host_path("/dev/vfio"), "/host/dev/vfio")
            self.assertEqual(host._driver_stage_parent(), "/var/lib/chs")

        with (
            mock.patch.object(host.os.path, "exists", return_value=False),
            mock.patch.object(host.os, "geteuid", return_value=0),
        ):
            self.assertEqual(host._driver_stage_parent(), "/var/lib/chs")

        with (
            mock.patch.object(host.os.path, "exists", return_value=False),
            mock.patch.object(host.os, "geteuid", return_value=1000),
        ):
            self.assertEqual(
                host._driver_stage_parent(), os.path.expanduser("~/.cache/chs")
            )

    def test_container_host_access_checks(self):
        with mock.patch.object(host, "in_container", return_value=True):
            with mock.patch.object(
                host.os, "readlink", return_value="pid:[4026531836]"
            ):
                host.check_host_access()

            links = {
                "/proc/self/ns/pid": "pid:[1]",
                "/host/proc/1/ns/pid": "pid:[2]",
            }
            with mock.patch.object(host.os, "readlink", side_effect=links.get):
                with self.assertRaisesRegex(HostCheckException, "--pid=host"):
                    host.check_host_access()

            with mock.patch.object(
                host.os, "readlink", side_effect=FileNotFoundError
            ):
                with self.assertRaisesRegex(HostCheckException, "-v /:/host"):
                    host.check_host_access()


class GceInstanceTest(unittest.TestCase):
    def test_reads_metadata_and_handles_failure(self):
        metadata = {
            "instance": {
                "name": "my-vm",
                "id": 1234,
                "zone": "projects/99/zones/asia-northeast1-b",
                "machineType": "projects/99/machineTypes/ct6e-standard-4t-d4",
            },
            "project": {"projectId": "my-project"},
        }
        response = io.BytesIO(json.dumps(metadata).encode())
        with mock.patch.object(
            host.urllib.request, "urlopen", return_value=response
        ) as urlopen:
            self.assertEqual(
                host.gce_instance(),
                host.GceInstance(
                    name="my-vm",
                    instance_id="1234",
                    machine_type="ct6e-standard-4t",
                    zone="asia-northeast1-b",
                    project_id="my-project",
                ),
            )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Metadata-flavor"), "Google")

        with mock.patch.object(
            host.urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("no route"),
        ):
            with self.assertRaisesRegex(ChsSchedulerException, "GCE VM"):
                host.gce_instance()


class FindDockerTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(host.subprocess, "run")
        self.run = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(host, "in_container", return_value=False)
        self.in_container = patcher.start()
        self.addCleanup(patcher.stop)

    def test_find_docker(self):
        self.run.return_value = _done("27.3.1")
        self.assertEqual(host.find_docker(), ["docker"])

        self.run.side_effect = [
            _done(stderr="permission denied ... docker.sock", returncode=1),
            _done("27.3.1"),
        ]
        self.assertEqual(host.find_docker(), ["sudo", "-n", "docker"])

        self.in_container.return_value = True
        self.run.side_effect = None
        self.run.return_value = _done("27.3.1")
        self.assertEqual(
            host.find_docker(),
            ["chroot", "/host", "env", "HOME=/root", "docker"],
        )

        self.run.return_value = _done(
            stderr="env: 'docker': No such file or directory", returncode=127
        )
        with self.assertRaisesRegex(HostCheckException, "no Docker CLI"):
            host.find_docker()

    def test_unusable_docker_raises(self):
        self.run.side_effect = [
            _done(stderr="permission denied ... docker.sock", returncode=1),
            _done(stderr="sudo: a password is required", returncode=1),
        ]
        with self.assertRaises(HostCheckException) as cm:
            host.find_docker()
        self.assertIn("permission denied", str(cm.exception))
        self.assertIn("password is required", str(cm.exception))

        self.run.side_effect = FileNotFoundError
        with self.assertRaisesRegex(HostCheckException, "not found"):
            host.find_docker()


class ProbeGpuAccessTest(FakeHostTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(host.subprocess, "run")
        self.run = patcher.start()
        self.addCleanup(patcher.stop)

    def test_probe_gpu_access(self):
        self.run.return_value = _done('{"nvidia":{},"runc":{}}')
        self.assertEqual(
            host.probe_gpu_access(["docker"]), host.GpuAccess(True)
        )

        self.run.return_value = _done('{"runc":{}}')
        with mock.patch.object(
            host, "_stage_distro_driver", return_value="/var/lib/chs/d"
        ):
            self.assertEqual(
                host.probe_gpu_access(["docker"]).driver_dir, "/var/lib/chs/d"
            )

        self.write("/var/lib/nvidia/bin/nvidia-smi", mode=0o755)
        self.assertEqual(
            host.probe_gpu_access(["docker"]),
            host.GpuAccess(False, "/var/lib/nvidia"),
        )

        self.run.return_value = _done(stderr="daemon down", returncode=1)
        with self.assertRaisesRegex(HostCheckException, "daemon down"):
            host.probe_gpu_access(["docker"])


class StageDistroDriverTest(FakeHostTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(
            host, "_driver_stage_parent", return_value="/var/lib/chs"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.write("/proc/driver/nvidia/version", _PROPRIETARY_VERSION)
        self.write("/usr/bin/nvidia-smi", "smi", mode=0o755)
        lib_dir = self.root + "/usr/lib/x86_64-linux-gnu"
        self.write("/usr/lib/x86_64-linux-gnu/libcuda.so.535.183.01", "cuda")
        os.symlink("libcuda.so.535.183.01", lib_dir + "/libcuda.so.1")
        self.write("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.183.01")
        self.write("/usr/lib/x86_64-linux-gnu/libc.so.6")
        self.staged = self.root + "/var/lib/chs/nvidia-driver-535.183.01"

    def test_stages_and_reuses_driver(self):
        self.assertEqual(
            host._stage_distro_driver(), "/var/lib/chs/nvidia-driver-535.183.01"
        )
        self.assertEqual(
            sorted(os.listdir(self.staged + "/lib64")),
            [
                "libcuda.so.1",
                "libcuda.so.535.183.01",
                "libnvidia-ml.so.535.183.01",
            ],
        )
        self.assertEqual(
            os.readlink(self.staged + "/lib64/libcuda.so.1"),
            "libcuda.so.535.183.01",
        )
        self.assertTrue(os.access(self.staged + "/bin/nvidia-smi", os.X_OK))

        before = os.stat(self.staged + "/lib64/libcuda.so.535.183.01")
        self.write("/usr/lib/x86_64-linux-gnu/libcuda.so.535.183.01", "new")
        host._stage_distro_driver()
        after = os.stat(self.staged + "/lib64/libcuda.so.535.183.01")
        self.assertEqual(before.st_ino, after.st_ino)

    def test_staging_failures_raise(self):
        for lib in ("libcuda.so.1", "libcuda.so.535.183.01"):
            os.remove(self.root + "/usr/lib/x86_64-linux-gnu/" + lib)
        with self.assertRaisesRegex(HostCheckException, "libcuda.so"):
            host._stage_distro_driver()
        self.assertEqual(os.listdir(self.root + "/var/lib/chs"), [])

        os.remove(self.root + "/usr/bin/nvidia-smi")
        with self.assertRaisesRegex(HostCheckException, "staging driver"):
            host._stage_distro_driver()

    def test_driver_version_and_stage_parent(self):
        self.write("/proc/driver/nvidia/version", _OPEN_VERSION)
        self.assertEqual(host._nvidia_driver_version(), "570.86.15")

        self.write("/proc/driver/nvidia/version", "garbage")
        with self.assertRaisesRegex(HostCheckException, "no driver version"):
            host._nvidia_driver_version()

        os.remove(self.root + "/proc/driver/nvidia/version")
        with self.assertRaisesRegex(HostCheckException, "could not read"):
            host._nvidia_driver_version()


class CheckIdleTest(FakeHostTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(host.subprocess, "run")
        self.run = patcher.start()
        self.addCleanup(patcher.stop)

    def test_check_gpus_idle(self):
        self.run.return_value = _done("")
        host.check_gpus_idle()
        self.assertEqual(self.run.call_args.args[0][0], "nvidia-smi")

        with mock.patch.object(host, "in_container", return_value=True):
            host.check_gpus_idle()
        self.assertEqual(
            self.run.call_args.args[0][:3], ["chroot", "/host", "nvidia-smi"]
        )

        self.write("/var/lib/nvidia/bin/nvidia-smi", mode=0o755)
        host.check_gpus_idle()
        self.assertEqual(
            self.run.call_args.args[0][0], "/var/lib/nvidia/bin/nvidia-smi"
        )

        self.run.return_value = _done("1234, python\n5678, /usr/bin/train\n")
        with self.assertRaisesRegex(
            HostCheckException,
            r"GPUs are busy: used by process 1234 \(python\), process 5678",
        ):
            host.check_gpus_idle()

        self.run.return_value = _done("NVIDIA-SMI has failed", returncode=9)
        with self.assertRaisesRegex(
            HostCheckException, "NVIDIA driver is unusable"
        ):
            host.check_gpus_idle()

    def _process(self, pid, comm, fd_targets):
        self.write(f"/proc/{pid}/comm", comm + "\n")
        fd_dir = f"{self.root}/proc/{pid}/fd"
        os.makedirs(fd_dir)
        for fd, target in enumerate(fd_targets):
            os.symlink(target, f"{fd_dir}/{fd}")

    def test_check_tpus_idle(self):
        self._process(1, "systemd", ["/dev/null"])
        self._process(20, "tpu-info", ["/dev/vfio/vfio"])
        self.write("/dev/vfio/0")
        os.makedirs(self.root + "/proc/sys")
        os.makedirs(self.root + "/proc/7")
        host.check_tpus_idle()

        with mock.patch.object(
            host.os,
            "open",
            side_effect=OSError(errno.EBUSY, "Device or resource busy"),
        ):
            with self.assertRaisesRegex(
                HostCheckException, r"TPUs are busy: .*/dev/vfio/0 is in use\."
            ):
                host.check_tpus_idle()

        self._process(300, "python3", ["/bad/missing", "/dev/vfio/0"])
        real_readlink = os.readlink
        with mock.patch.object(
            host.os,
            "readlink",
            side_effect=lambda p: (
                (_ for _ in ()).throw(FileNotFoundError(p))
                if p.endswith("/0")
                else real_readlink(p)
            ),
        ):
            with self.assertRaisesRegex(
                HostCheckException,
                r"TPUs are busy: used by process 300 \(python3\)",
            ):
                host.check_tpus_idle()


class GpuDevicesTest(FakeHostTestCase):
    def test_has_gpus(self):
        self.assertFalse(host.has_gpus())
        for path in (
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-modeset",
            "/dev/nvidia-caps/nvidia-cap1",
        ):
            self.write(path)
        self.assertFalse(host.has_gpus())
        self.write("/dev/nvidia0")
        self.assertTrue(host.has_gpus())


class TpuHostTest(FakeHostTestCase):
    def setUp(self):
        super().setUp()
        self.write("/dev/vfio/vfio")
        self.write("/dev/vfio/0")
        self.write("/dev/vfio/1")

    def test_tpu_devices(self):
        self.assertEqual(
            host.tpu_devices(),
            [self.root + "/dev/vfio/0", self.root + "/dev/vfio/1"],
        )

    def test_check_tpu_access(self):
        infinity = host.resource.RLIM_INFINITY
        for uid, hard in ((0, 65536), (1000, infinity)):
            with (
                mock.patch.object(host.os, "geteuid", return_value=uid),
                mock.patch.object(
                    host.resource, "getrlimit", return_value=(65536, hard)
                ),
            ):
                host.check_tpu_access()

        with (
            mock.patch.object(host.os, "geteuid", return_value=1000),
            mock.patch.object(
                host.resource, "getrlimit", return_value=(65536, 65536)
            ),
        ):
            with self.assertRaisesRegex(HostCheckException, "memlock"):
                host.check_tpu_access()

        with (
            mock.patch.object(host.os, "geteuid", return_value=0),
            mock.patch.object(
                host.os,
                "access",
                side_effect=lambda p, m: not p.endswith("/1"),
            ),
        ):
            with self.assertRaisesRegex(HostCheckException, "/dev/vfio/1"):
                host.check_tpu_access()


if __name__ == "__main__":
    unittest.main()
