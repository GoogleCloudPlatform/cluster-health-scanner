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

"""Unit tests for chs.schedulers.standalone.docker."""

import dataclasses
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from chs.exceptions import ChsSchedulerException
from chs.schedulers.standalone import docker
from chs.schedulers.standalone.host import GpuAccess
from chs.schedulers.standalone.job_spec import JobSpec

_DOCKER = ["sudo", "-n", "docker"]
_REAL_RUN = subprocess.run


def _done(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class DockerBackendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.script = os.path.join(self.tmp, "job_script")
        with open(self.script, "w") as f:
            f.write("nvidia-smi\n")
        patcher = mock.patch.object(docker.subprocess, "run")
        self.run = patcher.start()
        self.addCleanup(patcher.stop)
        self.run.return_value = _done()
        patcher = mock.patch.object(
            docker.host, "in_container", return_value=False
        )
        self.in_container = patcher.start()
        self.addCleanup(patcher.stop)

    def _spec(self, privileged=False):
        return JobSpec(
            job_name="run1-smi-abc123",
            node="vm",
            image="nvidia/cuda:12",
            script=self.script,
            timeout=600,
            order=1,
            privileged=privileged,
            run_id="run1",
        )

    def _backend(self, gpu=GpuAccess(nvidia_runtime=True)):
        return docker.DockerBackend(_DOCKER, gpu)

    def _argv(self):
        return self.run.call_args.args[0]

    def test_run_args_with_nvidia_runtime_and_timeout(self):
        args = self._backend()._run_args(self._spec())
        self.assertEqual(
            args,
            [
                "run",
                "-d",
                "--pull=never",
                "--name",
                "run1-smi-abc123",
                "--label",
                "chs.managed",
                "--label",
                "chs.run_id=run1",
                "--ipc=host",
                "--ulimit",
                "memlock=-1:-1",
                "-v",
                f"{self.tmp}:{self.tmp}",
                "--gpus",
                "all",
                "--entrypoint",
                "timeout",
                "nvidia/cuda:12",
                "-k",
                "30",
                "600",
                "bash",
                "-c",
                "nvidia-smi\n",
            ],
        )
        # The command after the image is valid for coreutils timeout.
        command = args[args.index("nvidia/cuda:12") + 1 : -1]
        result = _REAL_RUN(
            ["timeout", *command, "echo ok"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.stdout, "ok\n")

        priv_args = self._backend()._run_args(self._spec(privileged=True))
        self.assertIn("--privileged", priv_args)
        self.assertEqual(priv_args[priv_args.index("--gpus") + 1], "all")

    def test_run_args_driver_mount_requires_privileged(self):
        gpu = GpuAccess(
            nvidia_runtime=False, driver_dir="/var/lib/chs/nvidia-driver-535.1"
        )
        args = self._backend(gpu)._run_args(self._spec(privileged=True))
        start = args.index("memlock=-1:-1") + 3
        self.assertEqual(
            args[start : args.index("--entrypoint")],
            [
                "--privileged",
                "-v",
                "/var/lib/chs/nvidia-driver-535.1:/usr/local/nvidia:ro",
            ],
        )
        with self.assertRaisesRegex(ChsSchedulerException, "privileged"):
            self._backend(gpu)._run_args(self._spec())

    def test_run_args_mount_the_persistenced_socket_with_the_driver(self):
        socket = "/run/nvidia-persistenced/socket"
        driver = GpuAccess(nvidia_runtime=False, driver_dir="/d")
        spec = self._spec(privileged=True)
        exists = mock.Mock(return_value=False)
        with (
            mock.patch.object(docker.os.path, "exists", exists),
            mock.patch.object(
                docker.host, "host_path", side_effect=lambda p: "/host" + p
            ),
        ):
            self.assertNotIn(
                f"{socket}:{socket}", self._backend(driver)._run_args(spec)
            )
            exists.assert_called_once_with("/host" + socket)

            exists.return_value = True
            args = self._backend(driver)._run_args(spec)
            self.assertEqual(args[args.index(f"{socket}:{socket}") - 1], "-v")

            exists.reset_mock()
            self.assertNotIn(
                f"{socket}:{socket}", self._backend()._run_args(spec)
            )
            exists.assert_not_called()

    def test_run_args_in_chs_container_mount_state_volume(self):
        run_dir = "/workspace/.local/share/chs/workspaces/w/experiments/e"
        spec = dataclasses.replace(self._spec(), script=f"{run_dir}/job_script")
        self.in_container.return_value = True
        with (
            mock.patch.object(JobSpec, "read_script", return_value="true"),
            mock.patch.object(
                docker, "state_volume", return_value="proj_chs-state"
            ) as state_volume,
        ):
            backend = self._backend()
            args = backend._run_args(spec)
            self.assertIn(
                "type=volume,src=proj_chs-state,dst=/workspace/.local/share",
                args,
            )
            self.assertNotIn("-v", args)
            backend._run_args(spec)
            state_volume.assert_called_once_with()

            spec = dataclasses.replace(spec, script="/tmp/w/job_script")
            with self.assertRaisesRegex(ChsSchedulerException, "outside"):
                self._backend()._run_args(spec)

    def test_start_runs_docker_run_without_pulling(self):
        self._backend().start(self._spec())
        self.run.assert_called_once()
        self.assertEqual(self._argv()[3:6], ["run", "-d", "--pull=never"])
        self.assertEqual(self.run.call_args.kwargs["timeout"], 120)

    def test_pull_images_pulls_missing_images_once(self):
        self.run.side_effect = [
            _done(returncode=1),  # inspect nvidia/cuda:12: missing
            _done(),  # pull nvidia/cuda:12
            _done(),  # inspect debian:12: present
        ]
        other = dataclasses.replace(self._spec(), image="debian:12")
        self._backend().pull_images([self._spec(), self._spec(), other])
        self.assertEqual(
            [call.args[0][3:] for call in self.run.call_args_list],
            [
                ["image", "inspect", "nvidia/cuda:12"],
                ["pull", "nvidia/cuda:12"],
                ["image", "inspect", "debian:12"],
            ],
        )
        self.assertEqual(self.run.call_args_list[1].kwargs["timeout"], 600)

    def test_failed_pull_fails_only_jobs_using_the_image(self):
        backend = self._backend()
        self.run.side_effect = [
            _done(returncode=1),
            _done(stderr="pull access denied", returncode=1),
        ]
        with self.assertLogs(docker.logger, "ERROR"):
            backend.pull_images([self._spec()])
        self.run.reset_mock(side_effect=True)
        self.run.return_value = _done()
        with self.assertRaisesRegex(
            ChsSchedulerException, "Could not pull image nvidia/cuda:12.*denied"
        ):
            backend.start(self._spec())
        self.run.assert_not_called()

        backend.pull_images([self._spec()])
        self.run.assert_not_called()
        backend.start(dataclasses.replace(self._spec(), image="debian:12"))
        self.run.assert_called_once()

    def test_start_failures_raise(self):
        self.run.return_value = _done(
            stderr="pull access denied", returncode=125
        )
        with self.assertRaisesRegex(ChsSchedulerException, "pull access"):
            self._backend().start(self._spec())

        self.run.side_effect = subprocess.TimeoutExpired("docker", 600)
        with self.assertRaises(ChsSchedulerException):
            self._backend().start(self._spec())

    def test_exit_code(self):
        for stdout, expected in (
            ("running 0\n", None),
            ("created 0\n", None),
            ("exited 3\n", 3),
            ("dead 137\n", 137),
        ):
            self.run.return_value = _done(stdout)
            self.assertEqual(self._backend().exit_code(self._spec()), expected)

        self.run.return_value = _done(stderr="No such object", returncode=1)
        with self.assertRaises(ChsSchedulerException):
            self._backend().exit_code(self._spec())

    def test_stop_and_remove(self):
        self.run.return_value = _done(stderr="is not running", returncode=1)
        self._backend().stop(self._spec())
        self.assertEqual(self._argv()[3:], ["kill", "run1-smi-abc123"])

        self.run.return_value = _done()
        self._backend().remove(self._spec())
        self.assertEqual(self._argv()[3:], ["rm", "-f", "run1-smi-abc123"])

    def test_fetch_log(self):
        def fake_run(argv, stdout, stderr, **_):
            self.assertIs(stderr, subprocess.STDOUT)
            stdout.write("hello\n")
            return _done()

        self.run.side_effect = fake_run
        log = os.path.join(self.tmp, "exp.out")
        self._backend().fetch_log(self._spec(), log)
        with open(log) as f:
            self.assertEqual(f.read(), "hello\n")
        self.assertEqual(self._argv()[3:], ["logs", "run1-smi-abc123"])

        self.run.side_effect = None
        self.run.return_value = _done(returncode=1)
        with self.assertRaises(ChsSchedulerException):
            self._backend().fetch_log(self._spec(), log)

    def test_fetch_log_replaces_a_log_it_cannot_write(self):
        def fake_run(argv, stdout, stderr, **_):
            stdout.write("hello\n")
            return _done()

        self.run.side_effect = fake_run
        log = os.path.join(self.tmp, "exp.out")
        with open(log, "w") as f:
            f.write("old\n")
        os.chmod(log, 0o444)
        self._backend().fetch_log(self._spec(), log)
        with open(log) as f:
            self.assertEqual(f.read(), "hello\n")

        self.run.reset_mock()
        with self.assertRaises(FileNotFoundError):
            self._backend().fetch_log(
                self._spec(), os.path.join(self.tmp, "gone", "exp.out")
            )
        self.run.assert_not_called()

    def test_list_containers(self):
        self.run.return_value = _done("a\trun1\nb\trun2\n")
        self.assertEqual(
            docker.list_containers(_DOCKER, run_id="run1"),
            [
                docker.Container("a", "run1"),
                docker.Container("b", "run2"),
            ],
        )
        self.assertIn("label=chs.managed", self._argv())
        self.assertIn("label=chs.run_id=run1", self._argv())

        self.run.return_value = _done("")
        self.assertEqual(docker.list_containers(_DOCKER), [])
        self.assertFalse(
            any(a.startswith("label=chs.run_id") for a in self._argv())
        )

    def test_remove_containers(self):
        self.run.return_value = _done(
            stderr="Error response from daemon: No such container: a",
            returncode=1,
        )
        docker.remove_containers(_DOCKER, ["a", "b"])
        self.assertEqual(self._argv()[3:], ["rm", "-f", "a", "b"])

        self.run.return_value = _done(
            stderr="Error response from daemon: device busy", returncode=1
        )
        with self.assertRaisesRegex(ChsSchedulerException, "device busy"):
            docker.remove_containers(_DOCKER, ["a"])

        self.run.return_value = _done(
            stderr="Error response from daemon: No such file or directory",
            returncode=1,
        )
        with self.assertRaisesRegex(ChsSchedulerException, "No such file"):
            docker.remove_containers(_DOCKER, ["a"])

        self.run.reset_mock()
        docker.remove_containers(_DOCKER, [])
        self.run.assert_not_called()


class StateVolumeTest(unittest.TestCase):
    def _volume(self, *lines):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("".join(line + "\n" for line in lines))
        self.addCleanup(os.remove, f.name)
        return docker.state_volume(f.name)

    @staticmethod
    def _line(root, mount_point):
        return f"1 2 253:1 {root} {mount_point} rw - ext4 /dev/sda1 rw"

    def test_named_volumes(self):
        for name in ("chs-local-state", "proj_chs-state"):
            self.assertEqual(
                self._volume(
                    self._line("/", "/"),
                    self._line(
                        f"/var/lib/docker/volumes/{name}/_data",
                        "/workspace/.local/share",
                    ),
                ),
                name,
            )

    def test_last_mount_and_escapes_and_trailing_slash(self):
        with mock.patch.object(
            docker.host, "CHS_STATE_DIR", "/work space/state/"
        ):
            self.assertEqual(
                self._volume(
                    self._line(
                        "/var/lib/docker/volumes/old/_data",
                        "/work\\040space/state",
                    ),
                    self._line(
                        "/var/lib/docker/volumes/new/_data",
                        "/work\\040space/state",
                    ),
                ),
                "new",
            )

    def test_bind_mount_or_no_mount_raises(self):
        with self.assertRaisesRegex(ChsSchedulerException, "named Docker"):
            self._volume(self._line("/state", "/workspace/.local/share"))
        with self.assertRaisesRegex(ChsSchedulerException, "named Docker"):
            self._volume(self._line("/", "/"))
        with self.assertRaisesRegex(ChsSchedulerException, "Could not read"):
            docker.state_volume(os.path.join(tempfile.gettempdir(), "nope"))


if __name__ == "__main__":
    unittest.main()
