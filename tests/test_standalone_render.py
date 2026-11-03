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

"""Renders a real standalone Ramble workspace on a faked TPU VM."""

import importlib.util
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import chs
from chs.ramble_manager import RambleManager
from chs.schedulers import standalone
from chs.schedulers.standalone import job_spec
from chs.schedulers.standalone.host import GceInstance

_VM = "tpu-vm-1"


class StandaloneRenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "run/lock").mkdir(parents=True)
        real_find_spec = importlib.util.find_spec
        patches = [
            mock.patch.dict(
                os.environ,
                {
                    "HOME": str(self.tmp / "home"),
                    "XDG_DATA_HOME": str(self.tmp / "data"),
                    "RAMBLE_USER_DIR": str(self.tmp / "ramble"),
                    "PYTHONPATH": os.pathsep.join(
                        [
                            os.path.dirname(os.path.dirname(chs.__file__)),
                            os.environ.get("PYTHONPATH", ""),
                        ]
                    ),
                },
            ),
            mock.patch.object(
                standalone.host,
                "gce_instance",
                return_value=GceInstance(_VM, "1", "ct6e-standard-4t"),
            ),
            mock.patch.object(standalone.host, "check_host_access"),
            mock.patch.object(
                standalone.host, "tpu_devices", return_value=["d"] * 4
            ),
            mock.patch.object(standalone.host, "check_tpus_idle"),
            mock.patch.object(standalone.host, "check_tpu_access"),
            mock.patch.object(
                standalone.host,
                "host_path",
                side_effect=lambda p: str(self.tmp) + p,
            ),
            mock.patch.object(
                standalone.importlib.util,
                "find_spec",
                side_effect=lambda name, *args: (
                    mock.sentinel.spec
                    if name == standalone.BENCHMARKS_PACKAGE
                    else real_find_spec(name, *args)
                ),
            ),
            mock.patch.object(standalone, "NativeBackend"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _release(scheduler):
        if scheduler._lock_fd is not None:
            os.close(scheduler._lock_fd)

    def test_specs_and_scripts_name_the_vm(self):
        scheduler = standalone.StandaloneScheduler()
        self.addCleanup(self._release, scheduler)
        manager = RambleManager(
            str(self.tmp / "ws"),
            "render",
            run_id=f"{_VM}-lite-abcd",
        )
        manager.create_workspace()
        manager.add_custom_repository()
        manager.configure_workspace(
            scheduler,
            [{"name": "tpu-info", "variables": {"timeout": 150}}],
            machine_type="ct6e-standard-4t",
        )
        manager.setup_workspace()

        experiments = self.tmp / "ws/render/experiments/tpu-info"
        exp_dirs = [
            str(path.parent)
            for path in sorted(experiments.rglob(job_spec.SPEC_FILE))
        ]
        self.assertTrue(exp_dirs)
        for exp_dir in exp_dirs:
            with self.subTest(exp_dir=exp_dir):
                spec = job_spec.load_job_spec(exp_dir)
                self.assertEqual((spec.node, spec.runtime), (_VM, "tpu"))
                script = pathlib.Path(spec.script).read_text()
                self.assertIn(f'export MY_NODE_NAME="{_VM}"', script)
                self.assertNotIn("{node}", script)


if __name__ == "__main__":
    unittest.main()
