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

"""Tests for chs.schedulers.standalone.tpu_runtime."""

import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from chs import exceptions
from chs.schedulers.standalone import tpu_runtime


class EnsureRuntimeTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)
        self.root = self.tmp / "root"
        image_root = mock.patch.object(
            tpu_runtime, "IMAGE_ROOT", self.tmp / "image"
        )
        image_root.start()
        self.addCleanup(image_root.stop)
        self.calls = []
        run = mock.patch("subprocess.run", side_effect=self.fake_run)
        run.start()
        self.addCleanup(run.stop)

    def fake_run(self, cmd, **kwargs):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    def make_runtime(
        self, root: pathlib.Path | None = None, commit: str | None = None
    ) -> pathlib.Path:
        path = (root or self.root) / "tpu"
        path.mkdir(parents=True, exist_ok=True)
        (path / ".amb_commit").write_text(
            (commit or tpu_runtime.AMB_COMMIT) + "\n"
        )
        return path

    def test_build_steps_and_marker(self):
        path = tpu_runtime.ensure_runtime(self.root)

        self.assertEqual(path, self.root / "tpu")
        self.assertEqual(
            (path / ".amb_commit").read_text().strip(), tpu_runtime.AMB_COMMIT
        )
        venv = str(path / "venv")
        code = str(path / "code")
        self.assertEqual(
            self.calls,
            [
                ["git", "init", "-q", code],
                [
                    "git",
                    "-C",
                    code,
                    "fetch",
                    "-q",
                    "--depth",
                    "1",
                    tpu_runtime.AMB_REPO,
                    tpu_runtime.AMB_COMMIT,
                ],
                ["git", "-C", code, "checkout", "-q", "FETCH_HEAD"],
                [
                    sys.executable,
                    "-m",
                    "uv",
                    "venv",
                    "--no-cache",
                    "--python",
                    "3.12",
                    venv,
                ],
                [
                    sys.executable,
                    "-m",
                    "uv",
                    "pip",
                    "install",
                    "--no-cache",
                    "--python",
                    venv,
                    "-r",
                    str(path / "code" / "requirements.txt"),
                ],
            ],
        )

    def test_existing_runtime_is_reused(self):
        runtime = self.make_runtime()
        self.assertEqual(tpu_runtime.ensure_runtime(self.root), runtime)
        self.assertEqual(self.calls, [])

    def test_incomplete_or_old_runtime_is_rebuilt(self):
        partial = self.make_runtime(commit="old-commit")
        (partial / "venv").mkdir(parents=True)
        path = tpu_runtime.ensure_runtime(self.root)
        self.assertFalse((partial / "venv").exists())
        self.assertEqual(
            (path / ".amb_commit").read_text().strip(), tpu_runtime.AMB_COMMIT
        )

    def test_failed_step_raises_and_removes_partial_runtime(self):
        def fail_install(cmd, **kwargs):
            if "install" in cmd:
                raise subprocess.CalledProcessError(
                    1, cmd, stderr="no matching distribution"
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="")

        with mock.patch("subprocess.run", side_effect=fail_install):
            with self.assertRaisesRegex(
                exceptions.TpuRuntimeException, "no matching distribution"
            ):
                tpu_runtime.ensure_runtime(self.root)
        self.assertFalse((self.root / "tpu").exists())

    def test_uses_image_root_when_present(self):
        image_runtime = self.make_runtime(root=tpu_runtime.IMAGE_ROOT)
        self.assertEqual(tpu_runtime.ensure_runtime(), image_runtime)
        self.assertEqual(self.calls, [])

    def test_find_runtime(self):
        with mock.patch(
            "platformdirs.user_data_dir", return_value=str(self.root)
        ):
            self.assertIsNone(tpu_runtime.find_runtime())
            self.make_runtime(root=self.root / "runtimes", commit="old")
            self.assertIsNone(tpu_runtime.find_runtime())
            user = self.make_runtime(root=self.root / "runtimes")
            self.assertEqual(tpu_runtime.find_runtime(), user)
            image = self.make_runtime(root=tpu_runtime.IMAGE_ROOT)
            self.assertEqual(tpu_runtime.find_runtime(), image)
        self.assertEqual(self.calls, [])

    def test_builds_under_user_data_dir_when_image_root_missing(self):
        with mock.patch(
            "platformdirs.user_data_dir", return_value=str(self.root)
        ):
            path = tpu_runtime.ensure_runtime()
        self.assertEqual(path, self.root / "runtimes" / "tpu")


if __name__ == "__main__":
    unittest.main()
