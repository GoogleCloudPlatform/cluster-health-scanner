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

"""Unit tests for the standalone workflow manager."""

import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import yaml

ramble_mock = mock.Mock()
ramble_mock.wmkit = mock.Mock()


class _MockWorkflowManagerBase:
    def __init__(self, *args, **kwargs):
        pass


ramble_mock.wmkit.WorkflowManagerBase = _MockWorkflowManagerBase
ramble_mock.wmkit.workflow_manager_variable = mock.Mock()
ramble_mock.wmkit.tags = mock.Mock()
ramble_mock.wmkit.register_template = mock.Mock()
ramble_mock.wmkit.__all__ = [
    "WorkflowManagerBase",
    "workflow_manager_variable",
    "tags",
    "register_template",
]
ramble_mock.experiment_result = mock.Mock()
with mock.patch.dict(
    sys.modules,
    {
        "ramble": ramble_mock,
        "ramble.wmkit": ramble_mock.wmkit,
        "ramble.experiment_result": ramble_mock.experiment_result,
    },
):
    from chs.repos.workflow_managers.standalone import (
        workflow_manager as standalone_wm,
    )
from chs.schedulers.standalone import job_spec  # noqa: E402

_WM_DIR = os.path.dirname(standalone_wm.__file__)

_VARS = {
    "job_name": "vm-1-lite-ab12-gemm node/1",
    "experiment_run_dir": "/ws/experiments/gemm/default/gemm-node-1",
    "command": 'cd /ws\npython3 gemm.py >> "/ws/gemm.out" 2>&1',
    "node": "vm-1",
    "container_image": "nvcr.io/nvidia/cuda:12",
    "timeout": "600",
    "experiment_index": "4",
    "privileged": "false",
    "chs_run_id": "vm-1-lite-ab12",
    "runtime": "",
}


class StandaloneWorkflowManagerTest(unittest.TestCase):
    def setUp(self):
        self.vars = dict(_VARS)
        self.wm = standalone_wm.Standalone("file_path")
        self.wm.app_inst = mock.Mock()
        self.wm.app_inst.expander.expand_var_name.side_effect = (
            lambda name: self.vars[name]
        )

    def _spec(self):
        return yaml.safe_load(self.wm._job_vars()["job_spec_yaml"])

    def test_job_spec_and_loads(self):
        spec = self._spec()
        job_name = spec.pop("job_name")
        self.assertRegex(job_name, r"^vm-1-lite-ab12-gemm-node-1-[0-9a-f]{6}$")
        self.assertEqual(
            spec,
            {
                "node": "vm-1",
                "image": "nvcr.io/nvidia/cuda:12",
                "script": (
                    "/ws/experiments/gemm/default/gemm-node-1/job_script"
                ),
                "timeout": "600",
                "order": "4",
                "privileged": "false",
                "run_id": "vm-1-lite-ab12",
                "runtime": "",
            },
        )

        self.vars["runtime"] = "tpu"
        self.vars["privileged"] = "true"
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        with open(os.path.join(tmp, job_spec.SPEC_FILE), "w") as f:
            f.write(self.wm._job_vars()["job_spec_yaml"])
        loaded = job_spec.load_job_spec(tmp)
        self.assertEqual(loaded.runtime, "tpu")
        self.assertTrue(loaded.privileged)
        self.assertEqual((loaded.timeout, loaded.order), (600, 4))

        self.vars["privileged"] = ""
        self.assertEqual(self._spec()["privileged"], "false")

    def test_truncation_and_command(self):
        job_vars = self.wm._job_vars()
        self.assertEqual(
            job_vars["standalone_command"], "cd /ws\npython3 gemm.py "
        )

        self.vars["job_name"] = "x" * 300
        job_vars = self.wm._job_vars()
        self.assertRegex(
            yaml.safe_load(job_vars["job_spec_yaml"])["job_name"],
            r"^x{100}-[0-9a-f]{6}$",
        )

    def test_templates(self):
        known = set(_VARS) | set(self.wm._job_vars())
        for template in (
            "job_spec.yaml.tpl",
            "job_script.tpl",
        ):
            with open(os.path.join(_WM_DIR, template)) as f:
                content = f.read()
            self.assertLessEqual(set(re.findall(r"\{(\w+)\}", content)), known)
            if template != "job_spec.yaml.tpl":
                self.assertNotIn("{job_spec_yaml}", content)
                self.assertNotIn("{job_name}", content)

        with open(os.path.join(_WM_DIR, "job_script.tpl")) as f:
            content = f.read()
        self.assertIn('cd "{experiment_run_dir}"', content)
        self.assertTrue(content.rstrip().endswith("{standalone_command}"))


if __name__ == "__main__":
    unittest.main()
