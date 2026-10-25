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
import tempfile
import unittest
import urllib.error
from unittest import mock

from chs.exceptions import ChsSchedulerException, HostCheckException
from chs.schedulers.standalone import host


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


class CheckTpusIdleTest(FakeHostTestCase):
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
