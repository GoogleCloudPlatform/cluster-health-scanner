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

"""Unit tests for kubernetes_job batch templates."""

import pathlib
import re
import unittest

_TEMPLATE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src/chs/repos/workflow_managers/kubernetes_job"
)

_TIMEOUT_FLAG = re.compile(r"--(request-)?timeout=\d+s")


def _kubectl_lines(template: str, needle: str = "kubectl ") -> list[str]:
    return [
        line
        for line in (_TEMPLATE_DIR / template).read_text().splitlines()
        if needle in line and not line.lstrip().startswith("#")
    ]


class TestKubernetesJobTemplates(unittest.TestCase):
    """Every kubectl call in the batch templates must be time-bounded."""

    def test_every_kubectl_call_has_timeout(self):
        for template in ["batch_submit.tpl"]:
            with self.subTest(template=template):
                lines = _kubectl_lines(template)
                self.assertTrue(lines, f"no kubectl calls found in {template}")
                for line in lines:
                    self.assertRegex(
                        line,
                        _TIMEOUT_FLAG,
                        f"unbounded kubectl in {template}",
                    )

    def test_delete_bounds_finalizer_wait(self):
        for template in ["batch_submit.tpl"]:
            with self.subTest(template=template):
                deletes = _kubectl_lines(template, "kubectl delete")
                self.assertTrue(deletes)
                for line in deletes:
                    # --request-timeout does not bound the finalizer wait.
                    self.assertRegex(line, r"(?<!request-)--timeout=\d+s")

    def test_unused_batch_templates_are_removed(self):
        """batch_wait/batch_cancel are not rendered since CHS submits Jobs."""
        self.assertFalse((_TEMPLATE_DIR / "batch_wait.tpl").exists())
        self.assertFalse((_TEMPLATE_DIR / "batch_cancel.tpl").exists())
        manager = (_TEMPLATE_DIR / "workflow_manager.py").read_text()
        self.assertNotIn("batch_wait", manager)
        self.assertNotIn("batch_cancel", manager)

    def test_submit_pre_delete_stays_non_fatal(self):
        text = (_TEMPLATE_DIR / "batch_submit.tpl").read_text()
        self.assertRegex(
            text,
            re.compile(r"kubectl delete .*--timeout=\d+s \|\| true$", re.M),
        )


if __name__ == "__main__":
    unittest.main()
