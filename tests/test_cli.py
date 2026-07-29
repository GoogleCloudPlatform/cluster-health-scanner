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

"""Unit tests for the Cluster Health Scanner (CHS) CLI application.

These tests verify the correctness of the CLI's argument parsing, subcommand
routing, and error handling for various valid and invalid inputs.
"""

import dataclasses
import datetime
import os
import shutil
import subprocess
import tempfile
import unittest
from typing import Any
from unittest import mock

import click
from click import testing

from chs import cli
from chs import exceptions
from chs.cli import list_formatting
from chs.utils.tests import TestsMap as _TestsMap

CliRunner = testing.CliRunner
MOCK_DATETIME = datetime.datetime(2023, 1, 1, 12, 0, 0, 436888)
MOCK_WORKSPACE_NAME = "20230101-120000_mockcluster-suite1-mockhash"
MOCK_CUSTOM_WORKSPACE_NAME = "20230101-120000_mockcluster-custom-mockhash"


@dataclasses.dataclass
class _ListActionKwargs:
    suites: bool = False
    tests: bool = False
    schedulers: bool = False
    machine_types: bool = False
    suite: str | None = None
    machine_type: str | None = None


@dataclasses.dataclass
class _ListCommandTestCase:
    name: str
    args_list: list[str]
    expected_kwargs: dict[str, Any]


@dataclasses.dataclass
class _CommandErrorTestCase:
    name: str
    args_list: list[str]
    expected_output: str


@dataclasses.dataclass
class _RunCommandTestCase:
    name: str
    args_list: list[str]
    expected_return: cli.RunExecutionResult


@dataclasses.dataclass
class _ListActionTestCase:
    name: str
    kwargs: _ListActionKwargs
    expected_result: bool
    expected_suites_calls: int
    expected_tests_call_args: tuple[str | None, bool] | None
    expected_schedulers_calls: int
    expected_machine_types_calls: int = 0


@dataclasses.dataclass
class _HistoryTestCase:
    name: str
    mock_runs: dict[str, Any]
    expected_strings: list[str]
    unexpected_strings: list[str] = dataclasses.field(default_factory=list)


class TestCli(unittest.TestCase):
    """Tests the CLI logic and command invocation."""

    def setUp(self):
        super().setUp()
        self.runner = CliRunner()
        self.mock_rm_cls = self.enterContext(
            mock.patch("chs.cli.RambleManager", autospec=True)
        )
        self.mock_rm_instance = self.mock_rm_cls.return_value
        self.mock_rm_instance.workspace_name = MOCK_WORKSPACE_NAME
        self.mock_rm_instance.path = f"/mocked/workspaces/{MOCK_WORKSPACE_NAME}"
        self.mock_rm_instance.execute_experiments.return_value = []
        self.mock_rm_instance.archive.return_value = (
            "mock_ws_path",
            "mock_json_path",
            "mock_html_path",
            True,
        )

        self.mock_dt = self.enterContext(
            mock.patch("chs.cli.datetime.datetime", autospec=True)
        )
        self.mock_dt.now.return_value = MOCK_DATETIME

        self.mock_get_scheduler = self.enterContext(
            mock.patch("chs.cli.get_scheduler", autospec=True)
        )
        self.mock_detect_scheduler = self.enterContext(
            mock.patch("chs.cli.detect_scheduler", autospec=True)
        )
        self.mock_detect_scheduler.return_value = "slurm"
        self.mock_scheduler = self.mock_get_scheduler.return_value
        self.mock_scheduler.name = "mock_scheduler"
        self.mock_scheduler.nodes_arg = None
        self.mock_scheduler.nodelists_arg = None
        self.mock_scheduler.get_nodes_under_test.return_value = ["node1"]
        self.mock_scheduler.has_no_failed_jobs.return_value = False
        self.mock_scheduler.get_active_job_names.return_value = []
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "a3-megagpu-8g"
        ]
        self.mock_scheduler.run_preflight_checks.side_effect = (
            lambda tests, *args, **kwargs: tests
        )
        self.mock_scheduler.setup_scheduler_resources.side_effect = (
            lambda workspace_config_dir=None, tests=None, *args, **kwargs: tests
        )
        self.enterContext(
            mock.patch("chs.preflights.run_platform_preflights", autospec=True)
        )
        self.enterContext(
            mock.patch("chs.cli._setup_workspace_logger", autospec=True)
        )

        self.mock_load_suite = self.enterContext(
            mock.patch("chs.cli.load_suite", autospec=True)
        )
        self.mock_load_suite.return_value = {
            "tests": [
                {"name": "nccl-cluster"},
                {"name": "nccl-pairwise"},
                {"name": "dcgm"},
            ]
        }
        self.mock_app_data_cls = self.enterContext(
            mock.patch("chs.cli.ChsAppData", autospec=True)
        )
        self.mock_app_data = self.mock_app_data_cls.return_value
        self.mock_app_data.get_workspaces_dir.return_value = (
            "/mocked/workspaces"
        )
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"
        self.mock_app_data.resolve_run_id.side_effect = lambda x: x
        self.mock_app_data.get_storage_config.return_value = {
            "type": "gcs",
            "gcs": {"bucket_name": "test_bucket"},
        }

        self.mock_get_cluster_name = self.enterContext(
            mock.patch(
                "chs.cli.utils.get_sanitized_cluster_name",
                return_value="mockcluster",
            )
        )
        self.enterContext(
            mock.patch(
                "chs.cli.utils.generate_random_suffix", return_value="mockhash"
            )
        )

        # Mock suites utilities to return test dummy values
        self.enterContext(
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=[
                    "a3-ultragpu-8g",
                    "a4-highgpu-8g",
                    "a3-megagpu-8g",
                    "a4x-highgpu-4g",
                ],
            )
        )
        self.enterContext(
            mock.patch(
                "chs.utils.suites.get_all_suites",
                return_value=["lite", "standard", "burn-in", "suite1", "s1"],
            )
        )
        self.enterContext(
            mock.patch(
                "chs.utils.tests.get_all_supported_schedulers",
                return_value=["slurm", "kubernetes"],
            )
        )
        self.enterContext(
            mock.patch(
                "chs.utils.tests.get_all_supported_tests",
                return_value=[
                    "t1",
                    "t2",
                    "t3",
                    "nccl-cluster",
                    "nccl-pairwise",
                    "dcgm",
                ],
            )
        )
        self.enterContext(
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "slurm": {
                            "a3-megagpu-8g": [
                                "t1",
                                "t2",
                                "t3",
                                "nccl-cluster",
                                "nccl-pairwise",
                                "dcgm",
                            ],
                        },
                        "kubernetes": {
                            "a3-megagpu-8g": [
                                "t1",
                                "t2",
                                "t3",
                                "nccl-cluster",
                                "nccl-pairwise",
                                "dcgm",
                            ],
                        },
                    }
                ),
            )
        )

    def test_list_tests_implementation_both_hints(self):
        """Test _get_tests_list returns both hints."""

        renderables = list_formatting.get_tests_renderables(
            suite=None, tests_flag=True
        )
        hint = (
            "  (Hint: Use --suite <suite_name> or "
            "--machine-type <machine_type> to filter tests)"
        )
        self.assertTrue(
            any(hint in r.plain for r in renderables if hasattr(r, "plain"))
        )

    @mock.patch("chs.cli.list_formatting.load_suite")
    def test_list_tests_implementation_machine_type_hint(self, mock_load_suite):
        """Test _get_tests_list returns machine-type hint."""

        mock_load_suite.return_value = {"tests": ["test1", "test2"]}
        renderables = list_formatting.get_tests_renderables(
            suite="suite1", tests_flag=True
        )
        hint = "  (Hint: Use --machine-type <machine_type> to filter tests)"
        self.assertTrue(
            any(hint in r.plain for r in renderables if hasattr(r, "plain"))
        )

    @mock.patch("chs.cli.utils.tests.get_tests_by_machine_type")
    def test_list_tests_implementation_suite_hint(self, mock_get_tests):
        """Test _get_tests_list returns suite hint."""
        mock_get_tests.return_value = ["test1", "test2"]

        renderables = list_formatting.get_tests_renderables(
            suite=None, tests_flag=True, machine_type="a3-ultragpu-8g"
        )
        hint = "  (Hint: Use --suite <suite_name> to filter tests)"
        self.assertTrue(
            any(hint in r.plain for r in renderables if hasattr(r, "plain"))
        )

    @mock.patch("chs.cli.utils.tests.get_tests_by_machine_type")
    @mock.patch("chs.cli.list_formatting.load_suite")
    def test_list_tests_implementation_no_hint(
        self, mock_load_suite, mock_get_tests
    ):
        """Test _get_tests_list returns no hint."""
        mock_get_tests.return_value = ["test1", "test2"]
        mock_load_suite.return_value = {"tests": ["test1", "test2"]}

        renderables = list_formatting.get_tests_renderables(
            suite="suite1", tests_flag=True, machine_type="a3-ultragpu-8g"
        )
        self.assertFalse(
            any("Hint:" in r.plain for r in renderables if hasattr(r, "plain"))
        )

    @mock.patch.object(cli, "list_action", autospec=True)
    def test_list_command(self, mock_list_action):
        """Test 'list' command with various flags."""
        test_cases = [
            _ListCommandTestCase(
                name="suites",
                args_list=["list", "suites"],
                expected_kwargs={"suites": True},
            ),
            _ListCommandTestCase(
                name="tests",
                args_list=["list", "tests"],
                expected_kwargs={
                    "tests": True,
                    "suite": None,
                    "machine_type": None,
                },
            ),
            _ListCommandTestCase(
                name="schedulers",
                args_list=["list", "schedulers"],
                expected_kwargs={"schedulers": True},
            ),
            _ListCommandTestCase(
                name="machine_types",
                args_list=["list", "machine-types"],
                expected_kwargs={"machine_types": True},
            ),
            _ListCommandTestCase(
                name="all",
                args_list=["list"],
                expected_kwargs={},
            ),
            _ListCommandTestCase(
                name="suite",
                args_list=["list", "tests", "--suite", "suite1"],
                expected_kwargs={
                    "tests": True,
                    "suite": "suite1",
                    "machine_type": None,
                },
            ),
            _ListCommandTestCase(
                name="suite_trailing_comma",
                args_list=["list", "tests", "--suite", "suite1,"],
                expected_kwargs={
                    "tests": True,
                    "suite": "suite1,",
                    "machine_type": None,
                },
            ),
            _ListCommandTestCase(
                name="suite_leading_comma",
                args_list=["list", "tests", "--suite", ",suite1"],
                expected_kwargs={
                    "tests": True,
                    "suite": ",suite1",
                    "machine_type": None,
                },
            ),
            _ListCommandTestCase(
                name="suite_double_comma",
                args_list=["list", "tests", "--suite", "suite1,,suite2"],
                expected_kwargs={
                    "tests": True,
                    "suite": "suite1,,suite2",
                    "machine_type": None,
                },
            ),
            _ListCommandTestCase(
                name="suite_whitespace_and_comma",
                args_list=[
                    "list",
                    "tests",
                    "--suite",
                    "suite1, suite2 ,suite3 ",
                ],
                expected_kwargs={
                    "tests": True,
                    "suite": "suite1, suite2 ,suite3",
                    "machine_type": None,
                },
            ),
            _ListCommandTestCase(
                name="machine_type",
                args_list=[
                    "list",
                    "tests",
                    "--machine-type",
                    "a3-ultragpu-8g",
                ],
                expected_kwargs={
                    "tests": True,
                    "suite": None,
                    "machine_type": "a3-ultragpu-8g",
                },
            ),
        ]
        for case in test_cases:
            with self.subTest(case.name):
                mock_list_action.reset_mock()
                result = self.runner.invoke(cli.cli, case.args_list)
                self.assertEqual(result.exit_code, 0)
                mock_list_action.assert_called_once_with(**case.expected_kwargs)

    def test_list_command_errors(self):
        """Test 'list' command with invalid flags."""
        test_cases = [
            _CommandErrorTestCase(
                name="suite_empty",
                args_list=["list", "tests", "--suite", ""],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="suite_whitespace",
                args_list=["list", "tests", "--suite", " "],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="suite_without_tests",
                args_list=["list", "--suite", "suite1"],
                expected_output="No such option",
            ),
            _CommandErrorTestCase(
                name="suite_with_suites",
                args_list=["list", "suites", "--suite", "suite1"],
                expected_output="No such option",
            ),
        ]
        for case in test_cases:
            with self.subTest(case.name):
                result = self.runner.invoke(cli.cli, case.args_list)
                self.assertEqual(result.exit_code, 2)
                self.assertIn(case.expected_output, result.output)

    @mock.patch("chs.cli._analyze_action", autospec=True)
    @mock.patch("chs.cli.get_scheduler")
    def test_run_command(self, mock_get_scheduler, mock_analyze_action):
        """Test 'run' command."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir

        # Setup mock scheduler
        mock_scheduler = mock.Mock()
        mock_scheduler.get_nodes_under_test.return_value = ["node1"]
        mock_scheduler.filter_partial_tpu_slices.return_value = ["node1"]

        def get_scheduler_side_effect(name, **kwargs):
            s = mock.Mock()
            s.name = name
            s.nodes_arg = None
            s.nodelists_arg = None
            s.get_nodes_under_test.return_value = ["node1"]
            s.has_no_failed_jobs.return_value = False
            s.get_active_job_names.return_value = []
            s.detect_nodes_hardware.return_value = ["a3-megagpu-8g"]
            s.run_preflight_checks.side_effect = (
                lambda tests=None, *args, **kwargs: tests
            )
            s.setup_scheduler_resources.side_effect = (
                lambda workspace_config_dir=None,
                tests=None,
                *args,
                **kwargs: tests
            )
            return s

        mock_get_scheduler.side_effect = get_scheduler_side_effect

        test_cases = [
            _RunCommandTestCase(
                name="success",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                ],
                expected_return=cli.RunExecutionResult(
                    suite="suite1",
                    scheduler_name="slurm",
                    tests=[
                        {"name": "nccl-cluster"},
                        {"name": "nccl-pairwise"},
                        {"name": "dcgm"},
                    ],
                    workspace_name=MOCK_WORKSPACE_NAME,
                    workspaces_base_dir=mock_default_dir,
                    job_ids=["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="with_tests",
                args_list=[
                    "-v",
                    "run",
                    "--tests",
                    "t1,t2",
                ],
                expected_return=cli.RunExecutionResult(
                    suite="custom",
                    scheduler_name="mock_scheduler",
                    tests=[{"name": "t1"}, {"name": "t2"}],
                    workspace_name=MOCK_CUSTOM_WORKSPACE_NAME,
                    workspaces_base_dir=mock_default_dir,
                    job_ids=["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="with_single_test",
                args_list=[
                    "-v",
                    "run",
                    "--tests",
                    "t1",
                ],
                expected_return=cli.RunExecutionResult(
                    suite="custom",
                    scheduler_name="mock_scheduler",
                    tests=[{"name": "t1"}],
                    workspace_name=MOCK_CUSTOM_WORKSPACE_NAME,
                    workspaces_base_dir=mock_default_dir,
                    job_ids=["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="trailing_comma",
                args_list=[
                    "-v",
                    "run",
                    "--tests",
                    "t1,",
                ],
                expected_return=cli.RunExecutionResult(
                    suite="custom",
                    scheduler_name="mock_scheduler",
                    tests=[{"name": "t1"}],
                    workspace_name=MOCK_CUSTOM_WORKSPACE_NAME,
                    workspaces_base_dir=mock_default_dir,
                    job_ids=["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="leading_comma",
                args_list=[
                    "-v",
                    "run",
                    "--tests",
                    ",t1",
                ],
                expected_return=cli.RunExecutionResult(
                    suite="custom",
                    scheduler_name="mock_scheduler",
                    tests=[{"name": "t1"}],
                    workspace_name=MOCK_CUSTOM_WORKSPACE_NAME,
                    workspaces_base_dir=mock_default_dir,
                    job_ids=["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="double_comma",
                args_list=[
                    "-v",
                    "run",
                    "--tests",
                    "t1,,t2",
                ],
                expected_return=cli.RunExecutionResult(
                    suite="custom",
                    scheduler_name="mock_scheduler",
                    tests=[{"name": "t1"}, {"name": "t2"}],
                    workspace_name=MOCK_CUSTOM_WORKSPACE_NAME,
                    workspaces_base_dir=mock_default_dir,
                    job_ids=["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="whitespace",
                args_list=[
                    "-v",
                    "run",
                    "--tests",
                    "t1, t2 ",
                ],
                expected_return=cli.RunExecutionResult(
                    suite="custom",
                    scheduler_name="mock_scheduler",
                    tests=[{"name": "t1"}, {"name": "t2"}],
                    workspace_name=MOCK_CUSTOM_WORKSPACE_NAME,
                    workspaces_base_dir=mock_default_dir,
                    job_ids=["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="whitespace_and_comma",
                args_list=[
                    "-v",
                    "run",
                    "--tests",
                    "t1, t2 ,t3 ",
                ],
                expected_return=cli.RunExecutionResult(
                    suite="custom",
                    scheduler_name="mock_scheduler",
                    tests=[{"name": "t1"}, {"name": "t2"}, {"name": "t3"}],
                    workspace_name=MOCK_CUSTOM_WORKSPACE_NAME,
                    workspaces_base_dir=mock_default_dir,
                    job_ids=["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="with_nodes",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                    "--nodes",
                    "node1,node2",
                ],
                expected_return=cli.RunExecutionResult(
                    suite="suite1",
                    scheduler_name="mock_scheduler",
                    tests=[
                        {"name": "nccl-cluster"},
                        {"name": "nccl-pairwise"},
                        {"name": "dcgm"},
                    ],
                    workspace_name=MOCK_WORKSPACE_NAME,
                    workspaces_base_dir=mock_default_dir,
                    job_ids=["job1"],
                ),
            ),
        ]
        for case in test_cases:
            with self.subTest(case.name):
                self.mock_rm_cls.reset_mock()
                self.mock_rm_instance.reset_mock()
                self.mock_get_scheduler.reset_mock()
                self.mock_app_data.get_workspaces_dir.reset_mock()
                mock_analyze_action.reset_mock()
                self.mock_rm_instance.execute_experiments.return_value = [
                    "job1"
                ]

                self.mock_scheduler.name = "slurm"

                case.expected_return.scheduler_name = "slurm"

                result = self.runner.invoke(
                    cli.cli,
                    case.args_list + ["--auto-approve"],
                    standalone_mode=False,
                )
                self.assertEqual(
                    result.exit_code,
                    0,
                    f"Subtest {case.name} failed: {result.output}\n"
                    f"Exception: {result.exception}",
                )
                self.assertEqual(result.return_value, case.expected_return)

                self.mock_app_data.get_workspaces_dir.assert_called()
                self.mock_rm_cls.assert_any_call(
                    mock_default_dir,
                    case.expected_return.workspace_name,
                    run_id=f"mockcluster-{case.expected_return.suite}-mockhash",
                )
                self.mock_rm_instance.create_workspace.assert_called_once()
                self.mock_rm_instance.configure_workspace.assert_called_once()
                args, kwargs = (
                    self.mock_rm_instance.configure_workspace.call_args
                )
                self.assertEqual(
                    args[0].name, case.expected_return.scheduler_name
                )
                self.assertEqual(args[1], case.expected_return.tests)
                self.assertEqual(kwargs["machine_type"], "a3-megagpu-8g")
                self.mock_rm_instance.setup_workspace.assert_called_once()
                self.mock_rm_instance.execute_experiments.assert_called_once()
                mock_analyze_action.assert_called_once_with(
                    self.mock_app_data,
                    f"mockcluster-{case.expected_return.suite}-mockhash",
                )

    @mock.patch("chs.cli._analyze_action")
    def test_run_command_active_tier_tests_filtered_by_supported_tests(
        self, mock_analyze_action
    ):
        """Test active_tier_tests only contains tests for given machine type."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "tpu7x-standard-4t"
        ]
        self.mock_load_suite.return_value = {
            "tests": [
                {"name": "nvidia-smi"},
                {"name": "dcgm"},
                {"name": "tpu-info"},
                {"name": "collectives"},
            ]
        }

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["tpu7x-standard-4t", "a3-megagpu-8g"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "tpu7x-standard-4t": [
                                "tpu-info",
                                "collectives",
                            ],
                        }
                    }
                ),
            ),
        ):
            result = self.runner.invoke(
                cli.cli,
                ["run", "--suite", "standard", "--auto-approve"],
                standalone_mode=False,
            )
            self.assertEqual(
                result.exit_code,
                0,
                f"Command failed: {result.output}\n"
                f"Exception: {result.exception}",
            )

            self.mock_app_data.update_run.assert_called_once()
            _, run_details = self.mock_app_data.update_run.call_args[0]
            self.assertEqual(
                run_details["active_tier_tests"],
                ["tpu-info", "collectives"],
            )

    @mock.patch("chs.cli._analyze_action")
    def test_run_command_raises_validation_error_when_initial_tier_tests_empty(
        self, mock_analyze_action
    ):
        """Test Exception raised when initial tier has no supported tests."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "tpu7x-standard-4t"
        ]
        self.mock_load_suite.return_value = {
            "tests": [
                {"name": "tier1-gpu-test"},
                {
                    "name": "tpu-info",
                    "trigger_condition": {"DEGRADED": ["tier1-gpu-test"]},
                },
            ]
        }

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["tpu7x-standard-4t"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "tpu7x-standard-4t": [
                                "tpu-info",
                            ],
                        }
                    }
                ),
            ),
        ):
            result = self.runner.invoke(
                cli.cli,
                ["run", "--suite", "standard", "--auto-approve"],
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIn(
                "No supported initial health checks found in suite",
                result.output,
            )

    def test_run_action_called_process_error_includes_stderr(self):
        """Test run_action formats CalledProcessError with stderr."""
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "kubernetes"
        mock_scheduler.setup_scheduler_resources.side_effect = (
            lambda workspace_config_dir=None, tests=None, **kwargs: tests
        )
        err = subprocess.CalledProcessError(
            1,
            ["kubectl", "create", "namespace", "chs"],
            stderr="Error from server (Forbidden): namespaces 'chs' forbidden",
        )
        mock_scheduler.apply_scheduler_resources.side_effect = err

        with self.assertRaises(exceptions.ChsExecutionException) as ctx:
            cli.run_action(
                suite="custom",
                scheduler_obj=mock_scheduler,
                tests=[{"name": "tpu-info"}],
                workspace_name="test_ws",
                workspaces_base_dir="/tmp/test_ws",
                machine_type="tpu7x-standard-4t",
            )
        self.assertIn(
            "CHS execution failed: Command '['kubectl', 'create', 'namespace',"
            " 'chs']' returned non-zero exit status 1: Error from server"
            " (Forbidden): namespaces 'chs' forbidden",
            str(ctx.exception),
        )

    def test_format_subprocess_error_helper(self):
        """Test format_subprocess_error formats command & output."""
        err_with_stderr = subprocess.CalledProcessError(
            1,
            ["kubectl", "get", "crd"],
            stderr="Error from server (Forbidden)",
        )
        self.assertEqual(
            exceptions.format_subprocess_error(err_with_stderr),
            "Command '['kubectl', 'get', 'crd']' returned non-zero exit status"
            " 1: Error from server (Forbidden)",
        )

        err_with_stdout = subprocess.CalledProcessError(
            1,
            ["gcloud", "auth"],
            output="Invalid credentials",
        )
        self.assertEqual(
            exceptions.format_subprocess_error(err_with_stdout),
            "Command '['gcloud', 'auth']' returned non-zero exit status 1:"
            " Invalid credentials",
        )

        err_without_output = subprocess.CalledProcessError(
            1,
            ["ls", "/nonexistent"],
        )
        self.assertEqual(
            exceptions.format_subprocess_error(err_without_output),
            "Command '['ls', '/nonexistent']' returned non-zero exit status 1.",
        )

        err_with_bytes_stderr = subprocess.CalledProcessError(
            1,
            ["kubectl", "apply"],
            stderr=b"Permission denied error\n",
        )
        self.assertEqual(
            exceptions.format_subprocess_error(err_with_bytes_stderr),
            "Command '['kubectl', 'apply']' returned non-zero exit status 1:"
            " Permission denied error",
        )

        generic_err = ValueError("Something went wrong")
        self.assertEqual(
            exceptions.format_subprocess_error(generic_err),
            "Something went wrong",
        )

    @mock.patch("chs.cli._analyze_action")
    def test_run_command_preserves_suite_test_variables(
        self, mock_analyze_action
    ):
        """Test suite variables and workloads are preserved in tests config."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "a3-megagpu-8g"
        ]
        self.mock_load_suite.return_value = {
            "tests": [
                {
                    "name": "dcgm",
                    "variables": {"diag_level": [3]},
                },
                {
                    "name": "nccl-cluster",
                    "workloads": ["all-gather"],
                    "variables": {"begin_message_size": "256M"},
                },
            ]
        }

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["a3-megagpu-8g"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "a3-megagpu-8g": [
                                "dcgm",
                                "nccl-cluster",
                            ],
                        }
                    }
                ),
            ),
        ):
            result = self.runner.invoke(
                cli.cli,
                ["run", "--suite", "standard", "--auto-approve"],
                standalone_mode=False,
            )
            self.assertEqual(
                result.exit_code,
                0,
                f"Command failed: {result.output}\n"
                f"Exception: {result.exception}",
            )

            self.mock_app_data.update_run.assert_called_once()
            _, run_details = self.mock_app_data.update_run.call_args[0]
            self.assertEqual(
                run_details["tests"],
                [
                    {
                        "name": "dcgm",
                        "variables": {"diag_level": [3]},
                    },
                    {
                        "name": "nccl-cluster",
                        "workloads": ["all-gather"],
                        "variables": {"begin_message_size": "256M"},
                    },
                ],
            )

    @mock.patch("chs.cli._analyze_action")
    def test_run_command_custom_run_id(self, mock_analyze_action):
        """Test 'run' command with a custom run ID."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.name = "slurm"

        result = self.runner.invoke(
            cli.cli,
            [
                "-v",
                "run",
                "--suite",
                "suite1",
                "--run-id",
                "customrun",
                "--auto-approve",
            ],
            standalone_mode=False,
        )
        self.assertEqual(result.exit_code, 0)

        expected_run_id = "customrun-mockhash"
        expected_workspace_name = f"20230101-120000_{expected_run_id}"

        self.mock_rm_cls.assert_any_call(
            mock_default_dir, expected_workspace_name, run_id=expected_run_id
        )
        mock_analyze_action.assert_called_once_with(
            self.mock_app_data,
            expected_run_id,
        )

    @mock.patch("chs.cli._analyze_action")
    def test_run_command_auto_attach_success(self, mock_analyze_action):
        """Test 'run' command automatically attaches to an active run."""
        self.mock_app_data.get_latest_run_id.side_effect = None
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"
        self.mock_app_data.get_run.return_value = {"submission_complete": True}
        self.mock_scheduler.has_no_failed_jobs.return_value = True

        result = self.runner.invoke(
            cli.cli,
            ["-v", "run", "--suite", "suite1"],
            standalone_mode=False,
        )
        self.assertEqual(result.exit_code, 0)

        mock_analyze_action.assert_called_once_with(
            self.mock_app_data,
            "latest_run",
        )
        self.assertEqual(result.return_value.workspace_name, "latest_run")
        self.assertEqual(result.return_value.suite, "suite1")

    @mock.patch("chs.cli.cancel_action")
    @mock.patch("chs.cli._analyze_action")
    def test_run_command_auto_attach_died_mid_submission(
        self, mock_analyze_action, mock_cancel_action
    ):
        """Test 'run' command auto-cancels and restarts if incomplete."""
        self.mock_app_data.get_latest_run_id.side_effect = None
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"
        self.mock_app_data.get_run.return_value = {"submission_complete": False}
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir

        result = self.runner.invoke(
            cli.cli,
            ["-v", "run", "--suite", "suite1", "--auto-approve"],
            standalone_mode=False,
        )
        self.assertEqual(result.exit_code, 0)

        mock_cancel_action.assert_called_once_with("slurm")
        self.mock_rm_instance.execute_experiments.assert_called_once()

    @mock.patch("chs.cli.cancel_action")
    @mock.patch("chs.cli._analyze_action")
    def test_run_command_auto_attach_completed_run(
        self, mock_analyze_action, mock_cancel_action
    ):
        """Test 'run' auto-cancels and restarts if previous run is completed."""
        self.mock_app_data.get_latest_run_id.side_effect = None
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"
        self.mock_app_data.get_run.return_value = {
            "submission_complete": True,
            "is_boh_final": True,
        }
        self.mock_scheduler.has_no_failed_jobs.return_value = True
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir

        result = self.runner.invoke(
            cli.cli,
            ["-v", "run", "--suite", "suite1", "--auto-approve"],
            standalone_mode=False,
        )
        self.assertEqual(result.exit_code, 0)

        mock_cancel_action.assert_called_once_with("slurm")
        self.mock_rm_instance.execute_experiments.assert_called_once()

    def test_run_command_aborts_on_foreign_active_jobs(self):
        """Test 'run' aborts if foreign jobs are detected."""
        self.mock_app_data.get_latest_runs.return_value = {
            "ws1": {"run_id": "my-old-run-1"},
            "ws2": {"run_id": "my-old-run-2"},
        }

        self.mock_scheduler.get_active_job_names.return_value = [
            "chs-someone-elses-run-job-1",
            "chs-someone-elses-run-job-2",
        ]

        result = self.runner.invoke(
            cli.cli,
            ["run", "--suite", "suite1", "--auto-approve"],
            standalone_mode=False,
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIsInstance(result.exception, click.ClickException)
        self.assertIn(
            "A CHS run is already in progress on this cluster by another user",
            str(result.exception),
        )
        self.mock_app_data.get_latest_run_id.assert_not_called()

    def test_cancel_action_exception_handling(self):
        """Test that cancel_action wraps exceptions in ClickException."""
        self.mock_app_data.get_latest_run_id.return_value = "some_run_id"
        self.mock_scheduler.has_no_failed_jobs.return_value = True
        self.mock_scheduler.cancel_jobs_by_prefix.side_effect = (
            exceptions.ChsSchedulerException("Cancellation failed")
        )
        result = self.runner.invoke(cli.cli, ["cancel", "--auto-approve"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Cancellation failed", result.output)

    @mock.patch("chs.cli.RambleManager")
    @mock.patch("chs.cli._show_results")
    @mock.patch("chs.cli.StorageProviderFactory.get_provider_for_run")
    def test_analyze_action_archival_edge_cases(
        self, mock_get_provider, mock_show_results, mock_rm_class
    ):
        """Test _analyze_action archival shortcutting across storage cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_dir = os.path.join(tmpdir, "run1")
            os.makedirs(os.path.join(ws_dir, ".ramble-workspace"))
            boh_path = os.path.join(ws_dir, "bill_of_health.json")
            with open(boh_path, "w") as f:
                f.write("{}")

            self.mock_app_data.get_workspaces_dir.return_value = tmpdir
            self.mock_app_data.resolve_run_id.return_value = "run1"

            mock_rm = mock.MagicMock()
            mock_rm.path = ws_dir
            mock_rm.workspace_name = "run1"
            mock_rm.is_workspace_created = True
            mock_rm_class.return_value = mock_rm

            # Case 1: Local BoH exists + Storage NOT configured -> Shortcut
            mock_get_provider.return_value = None
            mock_rm.is_archived.return_value = True

            cli._analyze_action(self.mock_app_data, "run1")
            mock_show_results.assert_called_once_with(
                self.mock_app_data, "run1"
            )
            mock_rm.archive.assert_not_called()

            # Reset mocks
            mock_show_results.reset_mock()

            # Case 2: Local BoH + GCS Storage configured + is_archived=True
            mock_provider = mock.Mock()
            mock_get_provider.return_value = mock_provider
            mock_rm.is_archived.return_value = True

            cli._analyze_action(self.mock_app_data, "run1")
            mock_show_results.assert_called_once_with(
                self.mock_app_data, "run1"
            )
            mock_rm.archive.assert_not_called()

            # Reset mocks
            mock_show_results.reset_mock()

            # Case 3: Local BoH + GCS configured + is_archived=False -> Archive
            mock_rm.is_archived.return_value = False
            mock_rm.has_bill_of_health.return_value = True
            mock_rm.archive.return_value = (
                "gs://bucket/ws.tar.gz",
                "gs://bucket/boh.json",
                "gs://bucket/boh.html",
                True,
            )

            cli._analyze_action(self.mock_app_data, "run1")
            mock_rm.archive.assert_called_once()

    @mock.patch("chs.storage.StorageProviderFactory.get_provider_for_run")
    def test_results_command_no_boh_no_storage(self, mock_get_provider_for_run):
        """Test results command when BoH is missing and no storage set."""
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"
        self.mock_app_data.get_run.return_value = None
        mock_get_provider_for_run.return_value = None

        with (
            mock.patch(
                "chs.storage.StorageProviderFactory.get_active_provider",
                side_effect=Exception("No storage configured"),
            ),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            self.mock_app_data.get_workspaces_dir.return_value = tmpdir
            os.makedirs(os.path.join(tmpdir, "latest_run", "chs-results"))
            result = self.runner.invoke(cli.cli, ["results"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("failed to resolve storage provider", result.output)

    @mock.patch("chs.storage.StorageProviderFactory.get_provider_for_run")
    @mock.patch("chs.cli.ResultsViewer", autospec=True)
    def test_results_command_download_success(
        self, mock_viewer, mock_get_provider_for_run
    ):
        """Test results command downloads BoH from storage if missing."""
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"
        self.mock_app_data.get_run.return_value = None

        mock_provider = mock.Mock()
        mock_get_provider_for_run.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_app_data.get_workspaces_dir.return_value = tmpdir
            os.makedirs(os.path.join(tmpdir, "latest_run", "chs-results"))

            result = self.runner.invoke(cli.cli, ["results"])
            self.assertEqual(result.exit_code, 0)
            mock_provider.download_file.assert_called_once()
            args, _ = mock_provider.download_file.call_args
            self.assertEqual(args[0], "bills_of_health/latest_run.json")
            mock_viewer.assert_called_once()

    def test_archive_workspace_context_manager_success(self):
        """Test archive_workspace calling archival on success."""
        mock_rm = mock.Mock()
        mock_rm.archive.return_value = (
            "/mock/workspace.tar.gz",
            "/mock/boh.json",
            "/mock/boh.html",
            True,
        )
        with cli.archive_workspace(mock_rm):
            pass
        mock_rm.archive.assert_called_once()

    def test_archive_workspace_context_manager_failure(self):
        """Test archive_workspace calling archival on failure."""
        mock_rm = mock.Mock()
        mock_rm.archive.return_value = (
            "/mock/workspace.tar.gz",
            "/mock/boh.json",
            "/mock/boh.html",
            True,
        )
        with self.assertRaises(ValueError):
            with cli.archive_workspace(mock_rm):
                raise ValueError("test error")
        mock_rm.archive.assert_called_once()

    def test_archive_workspace_context_manager_keyboard_interrupt(self):
        """Test archive_workspace on BaseException (e.g. KeyboardInterrupt)."""
        mock_rm = mock.Mock()
        mock_rm.archive.return_value = (
            "/mock/workspace.tar.gz",
            "/mock/boh.json",
            "/mock/boh.html",
            True,
        )
        with self.assertRaises(KeyboardInterrupt):
            with cli.archive_workspace(mock_rm):
                raise KeyboardInterrupt()
        mock_rm.archive.assert_called_once()

    def test_archive_workspace_context_manager_detached(self):
        """Test archive_workspace bypassing archival on ChsDetachedException."""
        mock_rm = mock.Mock()
        with self.assertRaises(exceptions.ChsDetachedException):
            with cli.archive_workspace(mock_rm):
                raise exceptions.ChsDetachedException()
        mock_rm.archive.assert_not_called()

    def test_archive_workspace_context_manager_archive_failure(self):
        """Test archive_workspace raising ChsArchiveException when BOH fails."""
        mock_rm = mock.Mock()
        mock_rm.archive.return_value = (
            "/mock/workspace.tar.gz",
            "/mock/boh.json",
            None,
            False,
        )
        mock_rm.archive.side_effect = exceptions.ChsArchiveException("boh fail")
        with self.assertRaises(exceptions.ChsArchiveException):
            with cli.archive_workspace(mock_rm):
                pass
        mock_rm.archive.assert_called_once()

    def test_run_action_chs_exception(self):
        """Test that run_action handles ChsException and archives."""
        self.mock_app_data.get_latest_run_id.side_effect = (
            exceptions.NoPreviousRunsFoundException()
        )
        self.mock_scheduler.name = "slurm"
        self.mock_rm_instance.setup_workspace.side_effect = (
            exceptions.ChsException("Mock failure")
        )
        result = self.runner.invoke(
            cli.cli,
            ["run", "--auto-approve"],
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Mock failure", result.output)
        self.mock_rm_instance.archive.assert_called_once()

    def test_run_action_exception_handling(self):
        """Test that run_action wraps exceptions in ClickException."""
        self.mock_app_data.get_latest_run_id.side_effect = (
            exceptions.NoPreviousRunsFoundException()
        )
        self.mock_app_data.get_workspaces_dir.return_value = (
            "/mocked/workspaces"
        )

        self.mock_scheduler.name = "slurm"

        self.mock_rm_instance.setup_workspace.side_effect = OSError(
            "Permission denied"
        )
        result = self.runner.invoke(
            cli.cli,
            ["run", "-s", "s1", "--auto-approve"],
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("CHS execution failed", result.output)
        self.assertIn("Permission denied", result.output)
        self.mock_rm_instance.archive.assert_called_once()

    @mock.patch("chs.cli.detect_scheduler")
    @mock.patch("chs.cli.get_scheduler")
    @mock.patch("chs.cli._analyze_action", autospec=True)
    def test_run_command_passes_nodepool_to_scheduler(
        self, mock_analyze_action, mock_get_scheduler, mock_detect_scheduler
    ):
        """Test that nodepools argument is passed to get_scheduler."""
        mock_detect_scheduler.return_value = "kubernetes"
        self.mock_app_data.get_workspaces_dir.return_value = (
            "/mocked/workspaces"
        )
        self.mock_app_data.get_latest_run_id.return_value = None
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "kubernetes"
        mock_scheduler.get_nodes_under_test.return_value = ["node1"]
        mock_scheduler.filter_partial_tpu_slices.return_value = ["node1"]
        mock_scheduler.has_no_failed_jobs.return_value = False
        mock_scheduler.get_active_job_names.return_value = []
        mock_scheduler.detect_nodes_hardware.return_value = ["a3-megagpu-8g"]
        mock_scheduler.run_preflight_checks.side_effect = (
            lambda tests=None, *args, **kwargs: tests
        )
        mock_scheduler.setup_scheduler_resources.side_effect = (
            lambda workspace_config_dir=None, tests=None, *args, **kwargs: tests
        )
        mock_get_scheduler.return_value = mock_scheduler
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        result = self.runner.invoke(
            cli.cli,
            [
                "run",
                "--nodepools",
                "pool1",
                "--auto-approve",
            ],
        )

        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed: {result.output}\nException: {result.exception}",
        )
        self.assertEqual(mock_get_scheduler.call_count, 3)
        calls_with_nodepool = [
            c
            for c in mock_get_scheduler.call_args_list
            if c[1].get("nodepools") == "pool1"
        ]
        self.assertEqual(len(calls_with_nodepool), 1)

    def test_run_command_heterogeneous_cluster_raises(self):
        """Test `run` command when a heterogeneous cluster is detected."""
        self.mock_scheduler.name = "slurm"
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "a3-megagpu-8g",
            "a3-ultragpu-8g",
        ]
        result = self.runner.invoke(cli.cli, ["run", "--auto-approve"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Heterogeneous cluster detected", result.output)
        self.assertIn("--partitions", result.output)

    @mock.patch("chs.cli._handle_previous_run")
    def test_validations_occur_before_preparing_cluster(
        self, mock_handle_previous_run
    ):
        """Test validation failures prevent cluster preparation for execution"""
        mock_handle_previous_run.return_value = None
        self.mock_scheduler.name = "slurm"
        self.mock_scheduler.get_nodes_under_test.return_value = [
            "node1",
            "node2",
        ]
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "a3-megagpu-8g",
            "a3-ultragpu-8g",
        ]
        result = self.runner.invoke(cli.cli, ["run", "--auto-approve"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Heterogeneous cluster detected", result.output)

    def test_validate_run_args_empty_machine_types(self):
        """Test _validate_run_args raises validation error on empty types."""
        with self.assertRaisesRegex(
            exceptions.ChsValidationError, "No valid machine types"
        ):
            cli._validate_run_args("slurm", self.mock_scheduler, [], "test1")

    def test_run_command_errors(self):
        """Test 'run' command with invalid flags."""
        ws_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, ws_dir)
        self.mock_app_data.get_workspaces_dir.return_value = ws_dir

        test_cases = [
            _CommandErrorTestCase(
                name="empty_tests",
                args_list=[
                    "run",
                    "--suite",
                    "suite1",
                    "--tests",
                    "",
                ],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="empty_suite",
                args_list=[
                    "run",
                    "--suite",
                    "",
                ],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="run_id_empty",
                args_list=[
                    "run",
                    "--run-id",
                    "",
                ],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="run_id_uppercase",
                args_list=[
                    "run",
                    "--run-id",
                    "Uppercase",
                ],
                expected_output="Run ID must consist of lowercase alphanumeric",
            ),
            _CommandErrorTestCase(
                name="run_id_too_long",
                args_list=[
                    "run",
                    "--run-id",
                    "a" * 21,
                ],
                expected_output="Run ID must be 20 characters or less",
            ),
        ]
        for case in test_cases:
            with self.subTest(case.name):
                self.mock_scheduler.name = "slurm"

                if case.name == "no_nodes_provided":
                    self.mock_scheduler.get_nodes_under_test.return_value = []
                    mock_filter = self.mock_scheduler.filter_partial_tpu_slices
                    mock_filter.return_value = []
                else:
                    self.mock_scheduler.get_nodes_under_test.return_value = [
                        "node1"
                    ]
                    mock_filter = self.mock_scheduler.filter_partial_tpu_slices
                    mock_filter.return_value = ["node1"]

                result = self.runner.invoke(cli.cli, case.args_list)
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn(case.expected_output, result.output)

    @mock.patch("chs.cli.ResultsViewer", autospec=True)
    def test_results_command(self, mock_results_viewer):
        """Test 'results' command."""
        # Create temp workspace
        workspaces_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workspaces_dir)
        self.mock_app_data.get_workspaces_dir.return_value = workspaces_dir

        # Create BOH file
        run_dir = os.path.join(workspaces_dir, "my_run_id")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "bill_of_health.json"), "w") as f:
            f.write("{}")

        # Test valid run_id
        result = self.runner.invoke(
            cli.cli,
            ["results", "my_run_id"],
        )
        self.assertEqual(result.exit_code, 0)
        mock_results_viewer.assert_called_once()
        viewer_mock = mock_results_viewer.return_value
        viewer_mock.show_summary_and_details.assert_called_once_with(
            tests=None, resources=None, show_details=False
        )

        # Test with tests flag
        mock_results_viewer.reset_mock()
        result = self.runner.invoke(
            cli.cli,
            ["results", "my_run_id", "--tests", "t1,t2"],
        )
        self.assertEqual(result.exit_code, 0)
        viewer_mock.show_summary_and_details.assert_called_once_with(
            tests=["t1", "t2"], resources=None, show_details=False
        )

        # Test with resources flag
        mock_results_viewer.reset_mock()
        result = self.runner.invoke(
            cli.cli,
            ["results", "my_run_id", "--resources", "r1"],
        )
        self.assertEqual(result.exit_code, 0)
        viewer_mock.show_summary_and_details.assert_called_once_with(
            tests=None, resources=["r1"], show_details=False
        )

    def test_results_command_no_run_id(self):
        """Test 'results' command without run_id."""
        workspaces_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workspaces_dir)
        self.mock_app_data.get_workspaces_dir.return_value = workspaces_dir
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"

        # Create BOH file
        run_dir = os.path.join(workspaces_dir, "latest_run")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "bill_of_health.json"), "w") as f:
            f.write("{}")

        with mock.patch("chs.cli.ResultsViewer") as mock_viewer:
            result = self.runner.invoke(cli.cli, ["results"])
            self.assertEqual(result.exit_code, 0)
            self.mock_app_data.get_latest_run_id.assert_called_once()
            mock_viewer.assert_called_once()

    def test_ensure_object_called(self):
        """Test that ctx.obj is initialized."""
        self.mock_app_data_cls.reset_mock()
        result = self.runner.invoke(cli.cli, ["list"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.mock_app_data_cls.assert_called_once()

    @mock.patch(
        "chs.cli.list_formatting.get_machine_types_renderable", autospec=True
    )
    @mock.patch(
        "chs.cli.list_formatting.get_schedulers_renderable", autospec=True
    )
    @mock.patch("chs.cli.list_formatting.get_tests_renderables", autospec=True)
    @mock.patch("chs.cli.list_formatting.get_suites_renderable", autospec=True)
    def test_list_action(
        self,
        mock_suites,
        mock_tests,
        mock_schedulers,
        mock_machine_types,
    ):
        """Test list_action with various flags."""
        test_cases = [
            _ListActionTestCase(
                name="all",
                kwargs=_ListActionKwargs(),
                expected_result=True,
                expected_suites_calls=1,
                expected_tests_call_args=(None, False, None, None),
                expected_schedulers_calls=1,
                expected_machine_types_calls=1,
            ),
            _ListActionTestCase(
                name="suites_only",
                kwargs=_ListActionKwargs(suites=True),
                expected_result=False,
                expected_suites_calls=1,
                expected_tests_call_args=None,
                expected_schedulers_calls=0,
                expected_machine_types_calls=0,
            ),
            _ListActionTestCase(
                name="tests_only",
                kwargs=_ListActionKwargs(
                    tests=True,
                    suite="s1",
                ),
                expected_result=False,
                expected_suites_calls=0,
                expected_tests_call_args=("s1", True, None, None),
                expected_schedulers_calls=0,
                expected_machine_types_calls=0,
            ),
            _ListActionTestCase(
                name="schedulers_only",
                kwargs=_ListActionKwargs(schedulers=True),
                expected_result=False,
                expected_suites_calls=0,
                expected_tests_call_args=None,
                expected_schedulers_calls=1,
                expected_machine_types_calls=1,
            ),
            _ListActionTestCase(
                name="suites_and_tests",
                kwargs=_ListActionKwargs(
                    suites=True,
                    tests=True,
                ),
                expected_result=False,
                expected_suites_calls=1,
                expected_tests_call_args=(None, True, None, None),
                expected_schedulers_calls=0,
                expected_machine_types_calls=0,
            ),
            _ListActionTestCase(
                name="tests_and_schedulers",
                kwargs=_ListActionKwargs(
                    tests=True,
                    schedulers=True,
                ),
                expected_result=False,
                expected_suites_calls=0,
                expected_tests_call_args=(None, True, None, None),
                expected_schedulers_calls=1,
                expected_machine_types_calls=1,
            ),
            _ListActionTestCase(
                name="all_flags",
                kwargs=_ListActionKwargs(
                    suites=True,
                    tests=True,
                    schedulers=True,
                ),
                expected_result=False,
                expected_suites_calls=1,
                expected_tests_call_args=(None, True, None, None),
                expected_schedulers_calls=1,
                expected_machine_types_calls=1,
            ),
        ]
        mock_suites.return_value = ["suite1"]
        mock_tests.return_value = ["test1"]
        mock_schedulers.return_value = ["sch1"]
        for case in test_cases:
            with self.subTest(case.name):
                mock_suites.reset_mock()
                mock_tests.reset_mock()
                mock_schedulers.reset_mock()
                result = cli.list_action(**dataclasses.asdict(case.kwargs))
                self.assertEqual(result, case.expected_result)
                self.assertEqual(
                    mock_suites.call_count, case.expected_suites_calls
                )
                if case.expected_tests_call_args:
                    mock_tests.assert_called_once_with(
                        *case.expected_tests_call_args
                    )
                else:
                    mock_tests.assert_not_called()
                self.assertEqual(
                    mock_schedulers.call_count, case.expected_schedulers_calls
                )

    @mock.patch.object(cli, "cli", autospec=True)
    def test_main(self, mock_cli):
        """Test main entry point."""
        cli.main()
        mock_cli.assert_called_once_with(args=None, obj={})

    def test_results_command_no_previous_runs(self):
        """Test results command when no previous runs exist."""
        self.mock_app_data.get_latest_run_id.side_effect = (
            exceptions.NoPreviousRunsFoundException("No runs")
        )
        result = self.runner.invoke(cli.cli, ["results"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("No runs", result.output)

    @mock.patch("chs.storage.StorageProviderFactory.get_provider")
    def test_configure_storage_gcs(self, mock_get_provider):
        """Test 'configure storage gcs' command."""
        mock_provider = mock.Mock()
        mock_get_provider.return_value = mock_provider

        result = self.runner.invoke(
            cli.cli, ["configure", "storage", "gcs", "my-bucket"]
        )
        self.assertEqual(result.exit_code, 0)
        mock_get_provider.assert_called_once_with(
            {"type": "gcs", "gcs": {"bucket_name": "my-bucket"}}
        )
        mock_provider.validate.assert_called_once()
        self.mock_app_data.set_storage_config.assert_called_once_with(
            {"type": "gcs", "gcs": {"bucket_name": "my-bucket"}}
        )
        self.assertIn(
            "Storage configured and validated successfully.", result.output
        )

    @unittest.skip("Local path storage configuration is temporarily disabled")
    @mock.patch("chs.storage.StorageProviderFactory.get_provider")
    def test_configure_storage_path(self, mock_get_provider):
        """Test 'configure storage path' command."""
        mock_provider = mock.Mock()
        mock_get_provider.return_value = mock_provider

        result = self.runner.invoke(
            cli.cli, ["configure", "storage", "path", "/my/path"]
        )
        self.assertEqual(result.exit_code, 0)
        mock_get_provider.assert_called_once_with(
            {"type": "path", "path": {"path": "/my/path"}}
        )
        mock_provider.validate.assert_called_once()
        self.mock_app_data.set_storage_config.assert_called_once_with(
            {"type": "path", "path": {"path": "/my/path"}}
        )

    @mock.patch("chs.storage.StorageProviderFactory.get_provider")
    def test_configure_storage_validation_failure(self, mock_get_provider):
        """Test 'configure storage' fails if validation fails."""
        mock_provider = mock.Mock()
        mock_provider.validate.side_effect = Exception("Validation failed")
        mock_get_provider.return_value = mock_provider

        result = self.runner.invoke(
            cli.cli, ["configure", "storage", "gcs", "my-bucket"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.mock_app_data.set_storage_config.assert_not_called()
        self.assertIn("Storage validation failed", result.output)

    @mock.patch("chs.storage.StorageProviderFactory.get_provider")
    def test_configure_storage_interactive_gcs(self, mock_get_provider):
        """Test interactive 'configure storage' for GCS."""
        mock_provider = mock.Mock()
        mock_get_provider.return_value = mock_provider

        # Input: my-bucket
        result = self.runner.invoke(
            cli.cli, ["configure", "storage"], input="my-bucket\n"
        )
        self.assertEqual(result.exit_code, 0)
        self.mock_app_data.set_storage_config.assert_called_once_with(
            {"type": "gcs", "gcs": {"bucket_name": "my-bucket"}}
        )


class TestCliHistory(unittest.TestCase):
    """Tests the 'chs history' command."""

    def setUp(self):
        super().setUp()
        self.runner = testing.CliRunner()
        self.mock_app_data_cls = self.enterContext(
            mock.patch("chs.cli.ChsAppData", autospec=True)
        )
        self.mock_app_data = self.mock_app_data_cls.return_value

    def test_history_command(self):
        """Test history command with various scenarios."""
        test_cases = [
            _HistoryTestCase(
                name="no_runs",
                mock_runs={},
                expected_strings=["No runs found."],
            ),
            _HistoryTestCase(
                name="with_runs",
                mock_runs={
                    "run1": {
                        "run_id": "run1",
                        "submission_time": "2023-01-01T12:00:00Z",
                        "scheduler": "slurm",
                    },
                    "run2": {
                        "run_id": "run2",
                        "submission_time": "2023-01-02T12:00:00Z",
                        "command_statement": "chs run kubernetes",
                        "storage": {
                            "type": "path",
                            "path": {"path": "/tmp/test"},
                        },
                    },
                },
                expected_strings=[
                    "Run ID",
                    "Submitted",
                    "Storage",
                    "Age",
                    "Command",
                    "run1",
                    "run2",
                    "chs run",
                    "chs run kubernetes",
                    "/tmp/test",
                ],
                unexpected_strings=[
                    "Alias",
                    "my_alias",
                    "alias not set",
                    "Status",
                    "Details",
                    "BoH",
                ],
            ),
        ]
        for case in test_cases:
            with self.subTest(case.name):
                self.mock_app_data.get_latest_runs.return_value = case.mock_runs
                result = self.runner.invoke(cli.cli, ["history"])
                self.assertEqual(result.exit_code, 0)
                self.mock_app_data.get_latest_runs.assert_called_once_with(5)
                self.mock_app_data.get_latest_runs.reset_mock()
                for expected in case.expected_strings:
                    self.assertIn(expected, result.output)
                for unexpected in case.unexpected_strings:
                    self.assertNotIn(unexpected, result.output)


if __name__ == "__main__":
    unittest.main()
