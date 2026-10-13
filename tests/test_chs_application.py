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

"""Covers the CHS runner: ``Chs.create_experiment_chain``.

The runner turns one Ramble experiment into the experiments that do the work.
It reads the tests the suite asked for, loads each one's plugin, and renders
whatever the plugin returns.

There were no tests for it. That is how ``raise ChsExecutionException`` sat in
the error path for as long as it did without anyone noticing nothing imported
the name, which made the failure a ``NameError`` instead of the intended
exception.

Ramble is stubbed. Nothing here needs a real installation: the runner's job is
to decide *what* to render, and the stub records the calls.
"""

import dataclasses
import logging
import os
import sys
import unittest
from unittest import mock
from unittest.mock import MagicMock

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)

# Must be in place before the application module is imported: it does
# `from ramble.appkit import *` at module scope, and the class body calls
# maintainers/variable/tags/executable/workload while it is being defined.
sys.modules.setdefault("ramble", MagicMock())
sys.modules.setdefault("ramble.context", MagicMock())
sys.modules.setdefault("ramble.repository", MagicMock())
_appkit = sys.modules.get("ramble.appkit") or MagicMock()
_appkit.ExecutableApplication = object
for _name in (
    "maintainers",
    "variable",
    "tags",
    "executable",
    "workload",
    "figure_of_merit",
    "figure_of_merit_context",
    "success_criteria",
    "archive_pattern",
):
    setattr(_appkit, _name, MagicMock())
sys.modules["ramble.appkit"] = _appkit
sys.modules.setdefault("spack", MagicMock())
sys.modules.setdefault("spack.util.spack_yaml", MagicMock())

from chs.exceptions import ChsExecutionException  # noqa: E402
from chs.repos.applications.chs.application import Chs  # noqa: E402


def make_runner(tests, run_id="abc123", deadline=None):
    """Builds a Chs whose expander answers with the given suite settings."""
    runner = Chs()

    answers = {
        "tests": tests,
        "chs_run_id": run_id,
        "deadline": deadline,
    }
    runner.expander = mock.Mock()
    runner.expander.expand_var_name.side_effect = lambda name, typed=False: (
        answers.get(name)
    )
    runner.expander.experiment_name = "fallback-name"

    runner.non_reserved_variables = mock.Mock(
        return_value={
            "machine_type": "a3-ultragpu-8g",
            "scheduler": "kubernetes",
        }
    )
    runner.experiment_set = mock.Mock()
    runner.workspace = workspace()
    return runner


@dataclasses.dataclass
class SuiteEntry:
    """One entry from the parsed tests configuration.

    ``build_test_request`` reads these attributes for real, so a Mock will not
    do: its auto-created attributes are Mocks, and the request tries to merge
    ``variables`` into a dict.
    """

    name: str
    variables: dict = dataclasses.field(default_factory=dict)
    workloads: list = dataclasses.field(default_factory=list)


def workspace():
    ws = mock.Mock()
    ws.root = "/tmp/does-not-need-to-exist"
    return ws


class NoTests(unittest.TestCase):
    """A suite that asks for nothing renders nothing."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_an_empty_tests_variable_renders_nothing(self):
        runner = make_runner(tests=None)
        runner.create_experiment_chain()
        runner.experiment_set.render_experiment_set.assert_not_called()


class BadTestsConfig(unittest.TestCase):
    """A suite that cannot be parsed fails with the documented exception."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_unparseable_tests_raise_chs_execution_exception(self):
        """Regression: this used to raise NameError, not the real exception.

        ``ChsExecutionException`` was raised but never imported, so the error
        path blew up with a ``NameError`` and the caller saw nothing useful.
        """
        runner = make_runner(tests="this is not a tests config")

        with mock.patch(
            "chs.repos.applications.chs.application.parse_tests_config",
            side_effect=ValueError("malformed"),
        ):
            with self.assertRaises(ChsExecutionException) as caught:
                runner.create_experiment_chain()

        self.assertIn("malformed", str(caught.exception))


class UnknownTest(unittest.TestCase):
    """A test with no plugin is skipped, not fatal."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_a_test_with_no_plugin_is_skipped(self):
        runner = make_runner(tests="ignored")
        parsed = mock.Mock()
        parsed.tests = [mock.Mock(name="entry")]
        parsed.tests[0].name = "no-such-test"

        with mock.patch(
            "chs.repos.applications.chs.application.parse_tests_config",
            return_value=parsed,
        ):
            with mock.patch(
                "chs.repos.applications.chs.application.load_plugin",
                return_value=None,
            ):
                runner.create_experiment_chain()

        runner.experiment_set.render_experiment_set.assert_not_called()


class Rendering(unittest.TestCase):
    """What a plugin returns is what gets rendered."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def _run(self, specs, run_id="abc123"):
        runner = make_runner(tests="ignored", run_id=run_id)
        parsed = mock.Mock()
        # build_test_request reads these for real, so they have to be real
        # values rather than auto-created Mock attributes.
        parsed.tests = [SuiteEntry(name="gemm")]

        plugin = mock.Mock()
        plugin.build.return_value = specs

        with mock.patch(
            "chs.repos.applications.chs.application.parse_tests_config",
            return_value=parsed,
        ):
            with mock.patch(
                "chs.repos.applications.chs.application.load_plugin",
                return_value=plugin,
            ):
                runner.create_experiment_chain()
        return runner, plugin

    @staticmethod
    def _spec(application, workload):
        spec = mock.Mock()
        spec.application = application
        spec.workload = workload
        spec.to_ramble_context.return_value = {"variables": {}}
        return spec

    def test_every_spec_a_plugin_returns_is_rendered(self):
        specs = [self._spec("gemm", "check"), self._spec("gemm", "other")]
        runner, _ = self._run(specs)

        render = runner.experiment_set.render_experiment_set
        self.assertEqual(render.call_count, 2)
        rendered = [call.args[:2] for call in render.call_args_list]
        self.assertEqual(rendered, [("gemm", "check"), ("gemm", "other")])
        self.assertTrue(
            all(call.kwargs["chained"] for call in render.call_args_list)
        )

    def test_a_plugin_that_returns_nothing_renders_nothing(self):
        runner, _ = self._run([])
        runner.experiment_set.render_experiment_set.assert_not_called()

    def test_the_run_id_gives_the_experiment_name_prefix(self):
        _, plugin = self._run([self._spec("gemm", "check")], run_id="xyz789")
        request = plugin.build.call_args.args[0]
        self.assertEqual(request.base_prefix, "chs-xyz789")

    def test_without_a_run_id_the_experiment_name_is_the_prefix(self):
        _, plugin = self._run([self._spec("gemm", "check")], run_id=None)
        request = plugin.build.call_args.args[0]
        self.assertEqual(request.base_prefix, "fallback-name")


if __name__ == "__main__":
    unittest.main()
