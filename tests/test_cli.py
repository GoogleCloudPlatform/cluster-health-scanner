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
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import unittest
from typing import Any
from unittest import mock

import click
from click import testing

from chs import cli
from chs import constants
from chs import exceptions
from chs import telemetry
from chs.cli import background_cleanup
from chs.cli import list_formatting
from chs.protos import bill_of_health_pb2 as pb
from chs.ramble_manager import RambleManager
from chs.utils.k8s_client import KubeError
from chs.utils.tests import TestsMap as _TestsMap

from tests import k8s_fakes  # noqa: F401

CliRunner = testing.CliRunner
MOCK_DATETIME = datetime.datetime(2023, 1, 1, 12, 0, 0, 436888)
MOCK_WORKSPACE_NAME = "20230101-120000_mockcluster-suite1-mockhash"
MOCK_CUSTOM_WORKSPACE_NAME = "20230101-120000_mockcluster-custom-mockhash"
# A workspace name with a real 4-char suffix, and its telemetry run ID.
MOCK_TELEMETRY_WORKSPACE_NAME = "20230101-120000_mockcluster-lite-a3f9"
MOCK_TELEMETRY_RUN_ID = "20230101-120000-a3f9"


def _analyze_in_guard(app_data, run_id, scheduler):
    """Runs `_analyze_action` under the teardown guard, as `chs run` does."""
    with cli.cancel_jobs_on_failure(scheduler, run_id):
        return cli._analyze_action(app_data, run_id)


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

        def patched_logger_error(msg, *args, **kwargs):
            click.echo(msg, err=True)

        self.mock_logger_error = mock.patch(
            "chs.exceptions.logger.error", side_effect=patched_logger_error
        )
        self.mock_logger_error.start()
        self.addCleanup(self.mock_logger_error.stop)

        self.mock_rm_cls = self.enterContext(
            mock.patch("chs.cli.RambleManager", autospec=True)
        )
        self.mock_rm_instance = self.mock_rm_cls.return_value
        self.mock_rm_instance.workspace_name = MOCK_WORKSPACE_NAME
        self.mock_rm_instance.path = f"/mocked/workspaces/{MOCK_WORKSPACE_NAME}"
        self.mock_rm_instance.export_logs_dir = None
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
        self.mock_scheduler.nodepools_arg = None
        self.mock_scheduler.partitions_arg = None
        self.mock_scheduler.require_whole_slices = False
        self.mock_scheduler.partially_busy_slices = {}
        self.mock_scheduler.get_tpu_slice_summary.return_value = None
        self.mock_scheduler.get_nodes_under_test.return_value = ["node1"]
        self.mock_scheduler.find_runs_targeting_nodes.return_value = {}
        self.mock_scheduler.cancel_run.return_value = True
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
        self.mock_app_data.get_all_runs.return_value = {
            "latest_run": {"run_id": "latest_run"}
        }
        self.mock_app_data.resolve_run_id.side_effect = lambda x: x
        self.mock_app_data.get_run.return_value = None
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
        # Keep telemetry run tagging from leaking between tests.
        self.enterContext(
            mock.patch.object(telemetry, "_state", telemetry._State())
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
                return_value=["lite", "standard", "suite1", "s1"],
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
                expected_kwargs={"suites": True, "accelerator_type": None},
            ),
            _ListCommandTestCase(
                name="tests",
                args_list=["list", "tests"],
                expected_kwargs={
                    "tests": True,
                    "suite": None,
                    "machine_type": None,
                    "accelerator_type": None,
                },
            ),
            _ListCommandTestCase(
                name="schedulers",
                args_list=["list", "schedulers"],
                expected_kwargs={"schedulers": True, "accelerator_type": None},
            ),
            _ListCommandTestCase(
                name="machine_types",
                args_list=["list", "machine-types"],
                expected_kwargs={
                    "machine_types": True,
                    "accelerator_type": None,
                },
            ),
            _ListCommandTestCase(
                name="tpu_suites",
                args_list=["list", "suites", "--tpu"],
                expected_kwargs={"suites": True, "accelerator_type": "tpu"},
            ),
            _ListCommandTestCase(
                name="gpu_tests",
                args_list=["list", "tests", "--gpu"],
                expected_kwargs={
                    "tests": True,
                    "suite": None,
                    "machine_type": None,
                    "accelerator_type": "gpu",
                },
            ),
            _ListCommandTestCase(
                name="all_tpu",
                args_list=["list", "--tpu"],
                expected_kwargs={"accelerator_type": "tpu"},
            ),
            _ListCommandTestCase(
                name="all",
                args_list=["list"],
                expected_kwargs={"accelerator_type": None},
            ),
            _ListCommandTestCase(
                name="suite",
                args_list=["list", "tests", "--suite", "suite1"],
                expected_kwargs={
                    "tests": True,
                    "suite": "suite1",
                    "machine_type": None,
                    "accelerator_type": None,
                },
            ),
            _ListCommandTestCase(
                name="suite_trailing_comma",
                args_list=["list", "tests", "--suite", "suite1,"],
                expected_kwargs={
                    "tests": True,
                    "suite": "suite1,",
                    "machine_type": None,
                    "accelerator_type": None,
                },
            ),
            _ListCommandTestCase(
                name="suite_leading_comma",
                args_list=["list", "tests", "--suite", ",suite1"],
                expected_kwargs={
                    "tests": True,
                    "suite": ",suite1",
                    "machine_type": None,
                    "accelerator_type": None,
                },
            ),
            _ListCommandTestCase(
                name="suite_double_comma",
                args_list=["list", "tests", "--suite", "suite1,,suite2"],
                expected_kwargs={
                    "tests": True,
                    "suite": "suite1,,suite2",
                    "machine_type": None,
                    "accelerator_type": None,
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
                    "accelerator_type": None,
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
                    "accelerator_type": None,
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
            s.find_runs_targeting_nodes.return_value = {}
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
            mock_manager.attach_mock(s, "scheduler")
            return s

        mock_manager = mock.Mock()
        mock_manager.attach_mock(self.mock_rm_instance, "rm")
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

                apply_idx = mock_manager.mock_calls.index(
                    mock.call.scheduler.apply_scheduler_resources()
                )
                create_idx = mock_manager.mock_calls.index(
                    mock.call.rm.create_workspace()
                )
                config_idx = -1
                for i, c in enumerate(mock_manager.mock_calls):
                    if c[0] == "rm.configure_workspace":
                        config_idx = i
                        break
                self.assertLess(apply_idx, create_idx)
                self.assertLess(create_idx, config_idx)
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
                mock_manager.reset_mock()

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

    @mock.patch("chs.cli._analyze_action")
    def test_run_tpu_skips_multi_host_when_no_nodepool(
        self, mock_analyze_action
    ):
        """Test multi-host TPU tests are skipped when no nodepool is passed."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.nodes_arg = "node1"
        self.mock_scheduler.nodepools_arg = None
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "tpu7x-standard-4t"
        ]

        with (
            mock.patch("chs.cli.logger.warning") as mock_warning,
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["tpu7x-standard-4t"],
            ),
            mock.patch(
                "chs.utils.tests.get_all_supported_tests",
                return_value=["tpu-info", "tpu-collectives"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "tpu7x-standard-4t": [
                                "tpu-info",
                                "tpu-collectives",
                            ],
                        }
                    }
                ),
            ),
        ):
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--tests",
                    "tpu-info,tpu-collectives",
                    "--nodes",
                    "node1",
                    "--auto-approve",
                ],
                standalone_mode=False,
            )
            self.assertEqual(result.exit_code, 0)
            mock_warning.assert_any_call(
                "Multi-host TPU test(s) %s need entire slice, skipping "
                "them (please use --nodepools flag to run them).",
                "tpu-collectives",
            )
            self.mock_rm_instance.configure_workspace.assert_called_once()
            configured_tests = (
                self.mock_rm_instance.configure_workspace.call_args[0][1]
            )
            test_names = [t.get("name") for t in configured_tests]
            self.assertIn("tpu-info", test_names)
            self.assertNotIn("tpu-collectives", test_names)

    @mock.patch("chs.cli._analyze_action")
    def test_run_tpu_full_cluster_runs_multi_host(self, mock_analyze_action):
        """Test full-cluster TPU run schedules multi-host tests w/o flags."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.nodes_arg = None
        self.mock_scheduler.nodepools_arg = None
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "tpu7x-standard-4t"
        ]

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["tpu7x-standard-4t"],
            ),
            mock.patch(
                "chs.utils.tests.get_all_supported_tests",
                return_value=["tpu-info", "tpu-collectives"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "tpu7x-standard-4t": [
                                "tpu-info",
                                "tpu-collectives",
                            ],
                        }
                    }
                ),
            ),
        ):
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--tests",
                    "tpu-info,tpu-collectives",
                    "--auto-approve",
                ],
                standalone_mode=False,
            )
            self.assertEqual(result.exit_code, 0)
            self.mock_rm_instance.configure_workspace.assert_called_once()
            configured_tests = (
                self.mock_rm_instance.configure_workspace.call_args[0][1]
            )
            test_names = [t.get("name") for t in configured_tests]
            self.assertIn("tpu-info", test_names)
            self.assertIn("tpu-collectives", test_names)

    @mock.patch("chs.cli._analyze_action")
    def test_run_tpu_on_slurm_is_not_supported(self, mock_analyze_action):
        """A TPU run on Slurm is rejected, even when targeting partitions.

        The catalog mirrors the shipped platform files: TPU tests are listed
        for Kubernetes only, never for Slurm.
        """
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_scheduler.name = "slurm"
        self.mock_scheduler.partitions_arg = "v6e"
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "ct6e-standard-4t"
        ]

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="slurm"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["a3-megagpu-8g", "ct6e-standard-4t"],
            ),
            mock.patch(
                "chs.utils.tests.get_all_supported_tests",
                return_value=["dcgm", "tpu-info"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "slurm": {"a3-megagpu-8g": ["dcgm"]},
                        "kubernetes": {"ct6e-standard-4t": ["tpu-info"]},
                    }
                ),
            ),
        ):
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--tests",
                    "tpu-info",
                    "--partitions",
                    "v6e",
                    "--auto-approve",
                ],
                standalone_mode=False,
            )
        self.assertEqual(result.exit_code, 1)
        self.assertIsInstance(result.exception, exceptions.ChsValidationError)
        self.assertIn(
            "not supported for scheduler 'slurm' and machine type "
            "'ct6e-standard-4t'",
            str(result.exception),
        )
        self.assertNotIn("--partitions", str(result.exception))
        self.mock_rm_instance.execute_experiments.assert_not_called()

    @mock.patch("chs.cli._analyze_action")
    def test_run_tpu_fails_when_only_multi_host_and_no_nodepool(
        self, mock_analyze_action
    ):
        """Test error raised when only multi-host TPU tests without nodepool."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.nodes_arg = "node1"
        self.mock_scheduler.nodepools_arg = None
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "tpu7x-standard-4t"
        ]

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["tpu7x-standard-4t"],
            ),
            mock.patch(
                "chs.utils.tests.get_all_supported_tests",
                return_value=["tpu-collectives"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "tpu7x-standard-4t": [
                                "tpu-collectives",
                            ],
                        }
                    }
                ),
            ),
        ):
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--tests",
                    "tpu-collectives",
                    "--nodes",
                    "node1",
                    "--auto-approve",
                ],
                standalone_mode=False,
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIsInstance(
                result.exception, exceptions.ChsValidationError
            )
            self.assertIn(
                "TPU multi-host test(s) 'tpu-collectives' require an entire "
                "slice (use --nodepools).",
                str(result.exception),
            )

    @mock.patch("chs.cli._analyze_action")
    def test_run_tpu_skips_multi_host_when_slice_partially_busy(
        self, mock_analyze_action
    ):
        """Test multi-host tests skipped on partially busy mixed run."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.nodes_arg = None
        self.mock_scheduler.nodepools_arg = "pool1"
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "tpu7x-standard-4t"
        ]

        def mock_get_nodes():
            self.mock_scheduler.partially_busy_slices = {"pool1": ["node2"]}
            if self.mock_scheduler.require_whole_slices:
                return []
            return ["node1"]

        self.mock_scheduler.get_nodes_under_test.side_effect = mock_get_nodes

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["tpu7x-standard-4t"],
            ),
            mock.patch(
                "chs.utils.tests.get_all_supported_tests",
                return_value=["tpu-info", "tpu-collectives"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "tpu7x-standard-4t": [
                                "tpu-info",
                                "tpu-collectives",
                            ],
                        }
                    }
                ),
            ),
            mock.patch("chs.cli.logger") as mock_cli_logger,
        ):
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--tests",
                    "tpu-info,tpu-collectives",
                    "--nodepools",
                    "pool1",
                    "--auto-approve",
                ],
                standalone_mode=False,
            )
            self.assertEqual(result.exit_code, 0)
            self.mock_scheduler.get_nodes_under_test.assert_called_once_with()
            mock_cli_logger.warning.assert_any_call(
                "Slice '%s' is partially busy. Multi-host test(s) %s "
                "require full slices and were omitted from this run.",
                "pool1",
                "'tpu-collectives'",
            )
            self.mock_rm_instance.configure_workspace.assert_called_once()
            configured_tests = (
                self.mock_rm_instance.configure_workspace.call_args[0][1]
            )
            test_names = [t.get("name") for t in configured_tests]
            self.assertIn("tpu-info", test_names)
            self.assertNotIn("tpu-collectives", test_names)

    @mock.patch("chs.cli._analyze_action")
    def test_run_tpu_skips_multi_host_in_tiered_suite_when_slice_partially_busy(
        self, mock_analyze_action
    ):
        """Test later-tier multi-host tests skipped on partially busy slice."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.nodes_arg = None
        self.mock_scheduler.nodepools_arg = "pool1"
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "tpu7x-standard-4t"
        ]

        def mock_get_nodes():
            self.mock_scheduler.partially_busy_slices = {"pool1": ["node2"]}
            if self.mock_scheduler.require_whole_slices:
                return []
            return ["node1"]

        self.mock_scheduler.get_nodes_under_test.side_effect = mock_get_nodes

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["tpu7x-standard-4t"],
            ),
            mock.patch(
                "chs.utils.tests.get_all_supported_tests",
                return_value=["tpu-info", "tpu-collectives"],
            ),
            mock.patch(
                "chs.cli.load_suite",
                return_value={
                    "name": "tiered-tpu-suite",
                    "tests": [
                        {"name": "tpu-info"},
                        {
                            "name": "tpu-collectives",
                            "trigger_condition": {"HEALTHY": ["tpu-info"]},
                        },
                    ],
                },
            ),
            mock.patch(
                "chs.utils.suites.get_all_suites",
                return_value=["tiered-tpu-suite"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "tpu7x-standard-4t": [
                                "tpu-info",
                                "tpu-collectives",
                            ],
                        }
                    }
                ),
            ),
            mock.patch("chs.cli.logger") as mock_cli_logger,
        ):
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--suite",
                    "tiered-tpu-suite",
                    "--nodepools",
                    "pool1",
                    "--auto-approve",
                ],
                standalone_mode=False,
            )
            self.assertEqual(result.exit_code, 0)
            mock_cli_logger.warning.assert_any_call(
                "Slice '%s' is partially busy. Multi-host test(s) %s "
                "require full slices and were omitted from this run.",
                "pool1",
                "'tpu-collectives'",
            )
            self.mock_rm_instance.configure_workspace.assert_called_once()
            configured_tests = (
                self.mock_rm_instance.configure_workspace.call_args[0][1]
            )
            test_names = [t.get("name") for t in configured_tests]
            self.assertIn("tpu-info", test_names)
            self.assertNotIn("tpu-collectives", test_names)

    @mock.patch("chs.cli._analyze_action")
    def test_run_tpu_tiered_suite_multi_host_first_tier_rejects_nodes(
        self, mock_analyze_action
    ):
        """Test --nodes fails fast when the initial tier is multi-host only.

        Without the guard, pruning empties active_tier_tests and Ramble
        would treat `[]` as "no filter", running every tier at once.
        """
        self.mock_app_data.get_workspaces_dir.return_value = "/mocked/ws"
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.nodes_arg = "node1"
        self.mock_scheduler.nodepools_arg = None
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "tpu7x-standard-4t"
        ]

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["tpu7x-standard-4t"],
            ),
            mock.patch(
                "chs.utils.tests.get_all_supported_tests",
                return_value=["tpu-info", "tpu-collectives"],
            ),
            mock.patch(
                "chs.cli.load_suite",
                return_value={
                    "name": "tiered-multi-host-first",
                    "tests": [
                        {"name": "tpu-collectives"},
                        {
                            "name": "tpu-info",
                            "trigger_condition": {
                                "UNHEALTHY": ["tpu-collectives"]
                            },
                        },
                    ],
                },
            ),
            mock.patch(
                "chs.utils.suites.get_all_suites",
                return_value=["tiered-multi-host-first"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "tpu7x-standard-4t": [
                                "tpu-info",
                                "tpu-collectives",
                            ],
                        }
                    }
                ),
            ),
        ):
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--suite",
                    "tiered-multi-host-first",
                    "--nodes",
                    "node1",
                    "--auto-approve",
                ],
                standalone_mode=False,
            )

        self.assertIsInstance(result.exception, exceptions.ChsValidationError)
        self.assertIn("Initial tier", str(result.exception))
        self.assertIn("--nodepools", str(result.exception))
        self.mock_rm_instance.configure_workspace.assert_not_called()
        self.mock_rm_instance.execute_experiments.assert_not_called()

    @mock.patch("chs.cli._analyze_action")
    def test_run_tpu_does_not_log_multi_host_skip_on_dynamic_pool(
        self, mock_analyze_action
    ):
        """Test no multi-host fallback log for partially busy dynamic pool."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_app_data.get_latest_run_id.return_value = None
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.nodes_arg = None
        self.mock_scheduler.nodepools_arg = "dynpool"
        self.mock_scheduler.partially_busy_slices = {"dynpool": ["node2"]}
        self.mock_scheduler.get_nodes_under_test.return_value = []
        self.mock_scheduler.detect_nodes_hardware.return_value = [
            "tpu7x-standard-4t"
        ]

        with (
            mock.patch("chs.cli.detect_scheduler", return_value="kubernetes"),
            mock.patch(
                "chs.utils.tests.get_all_supported_machine_types",
                return_value=["tpu7x-standard-4t"],
            ),
            mock.patch(
                "chs.utils.tests.get_all_supported_tests",
                return_value=["tpu-info", "tpu-collectives"],
            ),
            mock.patch(
                "chs.utils.tests.get_tests_map",
                return_value=_TestsMap(
                    data={
                        "kubernetes": {
                            "tpu7x-standard-4t": [
                                "tpu-info",
                                "tpu-collectives",
                            ],
                        }
                    }
                ),
            ),
            mock.patch("chs.cli.logger") as mock_cli_logger,
        ):
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--tests",
                    "tpu-info,tpu-collectives",
                    "--nodepools",
                    "dynpool",
                    "--auto-approve",
                ],
                standalone_mode=False,
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIsInstance(
                result.exception, exceptions.NoNodesProvidedException
            )
            for call in mock_cli_logger.warning.call_args_list:
                self.assertNotIn(
                    "require full slices and were omitted from this run",
                    str(call),
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

    def test_run_action_kube_error_raises_execution_exception(self):
        """Test run_action wraps a Kubernetes API error like a CPE."""
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "kubernetes"
        mock_scheduler.setup_scheduler_resources.side_effect = (
            lambda workspace_config_dir=None, tests=None, **kwargs: tests
        )
        err = KubeError(
            'Error from server (Forbidden): namespaces "chs" is forbidden: '
            'User "chs" cannot create resource "namespaces"',
            status=403,
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
        self.assertEqual(str(ctx.exception), f"CHS execution failed: {err}")
        self.assertIs(ctx.exception.__cause__, err)
        self.mock_rm_instance.create_workspace.assert_not_called()

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

    def test_run_command_uses_the_given_scheduler(self):
        """Test 'run --scheduler' skips detection."""
        self.mock_get_scheduler.side_effect = exceptions.ChsSchedulerException(
            "stop here"
        )
        self.runner.invoke(
            cli.cli,
            [
                "run",
                "--suite",
                "suite1",
                "--auto-approve",
                "--scheduler",
                "standalone",
            ],
        )
        self.assertEqual(
            self.mock_get_scheduler.call_args.args, ("standalone",)
        )
        self.mock_detect_scheduler.assert_not_called()

    def test_run_command_rejects_unknown_scheduler(self):
        result = self.runner.invoke(
            cli.cli, ["run", "--scheduler", "pbs", "--auto-approve"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.mock_get_scheduler.assert_not_called()

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
            mock_default_dir,
            expected_workspace_name,
            run_id=expected_run_id,
        )
        mock_analyze_action.assert_called_once_with(
            self.mock_app_data, expected_run_id
        )

    @mock.patch("chs.cli._analyze_action")
    def test_run_command_storage_path(self, mock_analyze_action):
        """Test --storage-path flag for local and cloud destinations."""
        self.mock_app_data.get_workspaces_dir.return_value = "/mocked/ws"
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.name = "slurm"
        workspace_name = "20230101-120000_customrun-mockhash"

        with tempfile.TemporaryDirectory() as cwd:
            # 1. Default run: saves to ./<DEFAULT_OUTPUT_DIR>/<workspace_name>
            with mock.patch(
                "chs.cli.pathlib.Path.cwd", return_value=pathlib.Path(cwd)
            ):
                result = self.runner.invoke(
                    cli.cli,
                    [
                        "run",
                        "--suite",
                        "suite1",
                        "--run-id",
                        "customrun",
                        "--auto-approve",
                    ],
                    standalone_mode=False,
                )
                self.assertEqual(result.exit_code, 0, result.output)
                _, run_details = self.mock_app_data.update_run.call_args[0]
                expected_default_dir = os.path.join(
                    cwd, constants.DEFAULT_OUTPUT_DIR, workspace_name
                )
                self.assertEqual(
                    run_details["export_logs_dir"], expected_default_dir
                )
                self.assertEqual(
                    run_details["storage"],
                    {"type": "gcs", "gcs": {"bucket_name": "test_bucket"}},
                )
                self.assertIn(
                    "Results will be saved locally under "
                    f"./{constants.DEFAULT_OUTPUT_DIR}/ and uploaded to "
                    "gs://test_bucket",
                    result.output,
                )

            # 2. Custom local --storage-path: exports to a per-run subfolder
            # and still uploads to the globally configured GCS storage.
            custom_out = os.path.join(cwd, "custom_output")
            with mock.patch(
                "chs.cli.pathlib.Path.cwd", return_value=pathlib.Path(cwd)
            ):
                result = self.runner.invoke(
                    cli.cli,
                    [
                        "run",
                        "--suite",
                        "suite1",
                        "--run-id",
                        "customrun",
                        "--auto-approve",
                        "--storage-path",
                        custom_out,
                    ],
                    standalone_mode=False,
                )
                self.assertEqual(result.exit_code, 0, result.output)
                _, run_details = self.mock_app_data.update_run.call_args[0]
                self.assertEqual(
                    run_details["export_logs_dir"],
                    os.path.join(custom_out, workspace_name),
                )
                self.assertEqual(
                    run_details["storage"],
                    {"type": "gcs", "gcs": {"bucket_name": "test_bucket"}},
                )
                self.assertIn(
                    f"Results will be saved locally under {custom_out}/ "
                    "and uploaded to gs://test_bucket",
                    result.output,
                )

            # 3. Unwritable local --storage-path fails validation upfront
            self.mock_rm_instance.execute_experiments.reset_mock()
            with mock.patch(
                "chs.cli.StorageProviderFactory.get_provider"
            ) as mock_get_provider:
                mock_get_provider.return_value.validate.side_effect = (
                    exceptions.ChsValidationError(
                        "Directory '/forbidden' is not writable."
                    )
                )
                result = self.runner.invoke(
                    cli.cli,
                    [
                        "run",
                        "--suite",
                        "suite1",
                        "--run-id",
                        "customrun",
                        "--auto-approve",
                        "--storage-path",
                        "/forbidden",
                    ],
                )
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn(
                    "Storage path validation failed: Directory '/forbidden' "
                    "is not writable.",
                    result.output,
                )
                mock_get_provider.assert_called_once_with(
                    {"type": "path", "path": {"path": "/forbidden"}}
                )
                self.mock_rm_instance.execute_experiments.assert_not_called()

            # 4. Custom cloud --storage-path (gs:// URIs)
            storage_cases = [
                (
                    "gs://my-bucket/custom/prefix",
                    {"bucket_name": "my-bucket", "prefix": "custom/prefix"},
                ),
                (
                    "gs://my-bucket",
                    {"bucket_name": "my-bucket"},
                ),
            ]
            for storage_input, expected_gcs in storage_cases:
                with (
                    self.subTest(storage_input=storage_input),
                    mock.patch(
                        "chs.cli.pathlib.Path.cwd",
                        return_value=pathlib.Path(cwd),
                    ),
                    mock.patch(
                        "chs.cli.StorageProviderFactory.get_provider"
                    ) as mock_get_provider,
                ):
                    mock_provider = mock.Mock()
                    mock_get_provider.return_value = mock_provider
                    result = self.runner.invoke(
                        cli.cli,
                        [
                            "run",
                            "--suite",
                            "suite1",
                            "--run-id",
                            "customrun",
                            "--auto-approve",
                            "--storage-path",
                            storage_input,
                        ],
                        standalone_mode=False,
                    )
                    self.assertEqual(result.exit_code, 0, result.output)
                    mock_provider.validate.assert_called_once()
                    _, run_details = self.mock_app_data.update_run.call_args[0]
                    self.assertEqual(
                        run_details["storage"],
                        {
                            "type": "gcs",
                            "gcs": expected_gcs,
                        },
                    )
                    expected_default_dir = os.path.join(
                        cwd, constants.DEFAULT_OUTPUT_DIR, workspace_name
                    )
                    self.assertEqual(
                        run_details["export_logs_dir"], expected_default_dir
                    )
                    self.assertIn(
                        "Results will be saved locally under "
                        f"./{constants.DEFAULT_OUTPUT_DIR}/ and uploaded to "
                        f"{storage_input}",
                        result.output,
                    )

            # 5. Invalid --storage-path with empty bucket
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--suite",
                    "suite1",
                    "--run-id",
                    "customrun",
                    "--auto-approve",
                    "--storage-path",
                    "gs://",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn(
                "Storage validation failed: GCS bucket name cannot be "
                "empty.",
                result.output,
            )

            # 6. Unsupported URI scheme rejected upfront
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--suite",
                    "suite1",
                    "--run-id",
                    "customrun",
                    "--auto-approve",
                    "--storage-path",
                    "s3://my-bucket/path",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn(
                "Unsupported storage URI scheme 's3://'",
                result.output,
            )

            # 6b. Malformed gs: destination without // rejected upfront
            for bad_gs in ("gs:/my-bucket", "gs:my-bucket", "GS:/bucket"):
                result = self.runner.invoke(
                    cli.cli,
                    [
                        "run",
                        "--suite",
                        "suite1",
                        "--run-id",
                        "customrun",
                        "--auto-approve",
                        "--storage-path",
                        bad_gs,
                    ],
                )
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn(
                    f"Invalid GCS destination format '{bad_gs}'. "
                    "Did you mean 'gs://'?",
                    result.output,
                )

            # 7. No storage configured (local-only mode)
            self.mock_app_data.get_storage_config.return_value = None
            with mock.patch(
                "chs.cli.pathlib.Path.cwd", return_value=pathlib.Path(cwd)
            ):
                result = self.runner.invoke(
                    cli.cli,
                    [
                        "run",
                        "--suite",
                        "suite1",
                        "--run-id",
                        "customrun",
                        "--auto-approve",
                    ],
                    standalone_mode=False,
                )
                self.assertEqual(result.exit_code, 0, result.output)
                _, run_details = self.mock_app_data.update_run.call_args[0]
                self.assertIsNone(run_details["storage"])
                self.assertIn("No GCS storage configured", result.output)
                self.assertIn(
                    f"./{constants.DEFAULT_OUTPUT_DIR}/", result.output
                )

            # 8. No storage configured + local --storage-path
            result = self.runner.invoke(
                cli.cli,
                [
                    "run",
                    "--suite",
                    "suite1",
                    "--run-id",
                    "customrun",
                    "--auto-approve",
                    "--storage-path",
                    custom_out,
                ],
                standalone_mode=False,
            )
            self.assertEqual(result.exit_code, 0, result.output)
            _, run_details = self.mock_app_data.update_run.call_args[0]
            self.assertIsNone(run_details["storage"])
            self.assertEqual(
                run_details["export_logs_dir"],
                os.path.join(custom_out, workspace_name),
            )
            self.assertIn(
                "No GCS storage configured. Results will be saved locally "
                f"under {custom_out}/",
                result.output,
            )

    def test_resolve_and_validate_storage(self):
        """Only gs:// is remote; every other value is a local directory."""
        global_config = {"type": "gcs", "gcs": {"bucket_name": "global"}}
        app_data = mock.Mock()
        app_data.get_storage_config.return_value = global_config

        with tempfile.TemporaryDirectory() as tmp:
            tmp = os.path.realpath(tmp)
            home = os.path.join(tmp, "home")
            os.makedirs(home)
            work = os.path.join(tmp, "work")
            os.makedirs(work)
            old_cwd = os.getcwd()
            os.chdir(work)
            self.addCleanup(os.chdir, old_cwd)

            # Omitted or blank: global config, default export directory.
            for value in (None, "", "   "):
                with self.subTest(storage_path=value):
                    self.assertEqual(
                        cli._resolve_and_validate_storage(value, app_data),
                        (global_config, None),
                    )

            # Local paths: keep the global config, export to the directory.
            with mock.patch.dict(os.environ, {"HOME": home}):
                local_cases = [
                    ("my-bucket", os.path.join(work, "my-bucket")),
                    (
                        "my-bucket/team-a",
                        os.path.join(work, "my-bucket", "team-a"),
                    ),
                    ("./custom-out", os.path.join(work, "custom-out")),
                    ("../sibling", os.path.join(tmp, "sibling")),
                    ("~/reports", os.path.join(home, "reports")),
                    (os.path.join(tmp, "abs"), os.path.join(tmp, "abs")),
                ]
                for value, expected_dir in local_cases:
                    with self.subTest(storage_path=value):
                        self.assertEqual(
                            cli._resolve_and_validate_storage(value, app_data),
                            (global_config, expected_dir),
                        )
                        self.assertTrue(os.path.isdir(expected_dir))
                        # The write-test file is removed after validation.
                        self.assertEqual(os.listdir(expected_dir), [])
            os.chdir(old_cwd)

        # gs:// URIs (case-insensitive scheme) replace the global config.
        with mock.patch(
            "chs.cli.StorageProviderFactory.get_provider"
        ) as mock_get_provider:
            self.assertEqual(
                cli._resolve_and_validate_storage("GS://b/p/", app_data),
                (
                    {"type": "gcs", "gcs": {"bucket_name": "b", "prefix": "p"}},
                    None,
                ),
            )
            mock_get_provider.return_value.validate.assert_called_once()

        # Validation errors keep the underlying exception as the cause.
        with mock.patch(
            "chs.cli.StorageProviderFactory.get_provider"
        ) as mock_get_provider:
            cause = exceptions.ChsValidationError("not writable")
            mock_get_provider.return_value.validate.side_effect = cause
            for value in ("/some/dir", "gs://b"):
                with self.subTest(storage_path=value):
                    with self.assertRaises(click.ClickException) as ctx:
                        cli._resolve_and_validate_storage(value, app_data)
                    self.assertIs(ctx.exception.__cause__, cause)

        # Non-gs:// URI schemes are rejected.
        with self.assertRaisesRegex(
            click.ClickException, "Unsupported storage URI scheme 's3://'"
        ):
            cli._resolve_and_validate_storage("s3://b", app_data)

        # Malformed gs: prefixes are rejected with a helpful suggestion.
        for bad_gs in (
            "gs:/bucket",
            "gs:bucket",
            "GS:/bucket",
            "gs:",
            "gs:foo/bar",
        ):
            with self.subTest(storage_path=bad_gs):
                with self.assertRaisesRegex(
                    click.ClickException,
                    rf"Invalid GCS destination format '{re.escape(bad_gs)}'\. "
                    r"Did you mean 'gs://'\?",
                ):
                    cli._resolve_and_validate_storage(bad_gs, app_data)

    def test_resolve_export_logs_dir(self):
        """Resolver always nests results in a per-run subfolder."""
        with tempfile.TemporaryDirectory() as cwd:
            with mock.patch(
                "chs.cli.pathlib.Path.cwd", return_value=pathlib.Path(cwd)
            ):
                # Default when custom_export_dir is None
                self.assertEqual(
                    cli._resolve_export_logs_dir(None, "ws1"),
                    os.path.join(cwd, constants.DEFAULT_OUTPUT_DIR, "ws1"),
                )

                # Custom absolute path gets a per-run subfolder too
                custom_abs = os.path.join(cwd, "my_custom_dir")
                self.assertEqual(
                    cli._resolve_export_logs_dir(custom_abs, "ws1"),
                    os.path.join(custom_abs, "ws1"),
                )

    @mock.patch("chs.cli.logger", autospec=True)
    def test_export_logs_hook(self, mock_logger):
        """_export_logs exports, or falls back to workspace on OSError."""
        rm = mock.create_autospec(RambleManager, instance=True)
        rm.workspace_name = "ws1"
        rm.path = "/mocked/workspaces/ws1"

        rm.export_logs_dir = None
        cli._export_logs(rm)
        rm.export_logs.assert_not_called()
        mock_logger.debug.assert_not_called()

        export_dir = f"/custom/{constants.DEFAULT_OUTPUT_DIR}/ws1"
        rm.export_logs_dir = export_dir
        rm.export_logs.return_value = [f"{export_dir}/a.log"]
        cli._export_logs(rm)
        rm.export_logs.assert_called_once_with(export_dir)
        rm.update_run_entry.assert_not_called()

        # OSError: warn, do NOT copy into the workspace, and keep the
        # configured export location so later retries can still export there.
        rm.export_logs.reset_mock()
        rm.export_logs.side_effect = PermissionError(13, "Permission denied")
        with mock.patch("chs.cli.Console", autospec=True) as mock_console:
            cli._export_logs(rm)
        rm.export_logs.assert_called_once_with(export_dir)
        rm.update_run_entry.assert_not_called()
        warning = mock_console.return_value.print.call_args[0][0]
        self.assertIn(f"Could not export results to '{export_dir}'", warning)
        self.assertIn("Permission denied", warning)
        self.assertIn(
            "Results preserved in workspace: /mocked/workspaces/ws1", warning
        )

    @mock.patch("chs.cli._export_logs", autospec=True)
    def test_archive_and_log_exports_after_archiving(self, mock_export):
        """_archive_and_log triggers the export hook after archiving."""
        calls = []
        rm = mock.create_autospec(RambleManager, instance=True)
        rm.workspace_name = "ws1"
        rm.path = "/nonexistent/ws1"
        rm.export_logs_dir = None
        rm.has_bill_of_health.return_value = False
        mock_export.side_effect = lambda _: calls.append("export")
        rm.archive.side_effect = lambda: calls.append("archive") or (
            None,
            None,
            None,
            False,
        )
        with mock.patch(
            "chs.cli.StorageProviderFactory.get_provider_for_run",
            return_value=None,
        ):
            cli._archive_and_log(rm)
        self.assertEqual(calls, ["archive", "export"])
        mock_export.assert_called_once_with(rm)
        rm.update_run_entry.assert_not_called()

    def test_archive_and_log_still_archives_when_export_crashes(self):
        """An unexpected export error never blocks the GCS upload."""
        rm = mock.create_autospec(RambleManager, instance=True)
        rm.workspace_name = "ws1"
        rm.path = "/nonexistent/ws1"
        rm.export_logs_dir = "/out/ws1"
        rm.has_bill_of_health.return_value = False
        rm.export_logs.side_effect = AttributeError("bad run metadata")
        rm.archive.return_value = (None, None, None, False)
        with (
            mock.patch(
                "chs.cli.StorageProviderFactory.get_provider_for_run",
                return_value=None,
            ),
            mock.patch("chs.cli.Console", autospec=True) as mock_console,
        ):
            cli._archive_and_log(rm)
        rm.archive.assert_called_once()
        printed = [
            c.args[0] for c in mock_console.return_value.print.call_args_list
        ]
        self.assertTrue(
            any("Could not export results to '/out/ws1'" in p for p in printed)
        )

    @mock.patch("chs.cli._export_logs", autospec=True)
    def test_archive_and_log_gcs_upload_failure_warns(self, mock_export):
        """GCS upload failure names the destination and shows local paths."""
        with tempfile.TemporaryDirectory() as export_dir:
            for name in ("bill_of_health.html", "bill_of_health.json"):
                pathlib.Path(export_dir, name).write_text("{}")
            rm = mock.create_autospec(RambleManager, instance=True)
            rm.workspace_name = "ws1"
            rm.path = "/nonexistent/ws1"
            rm.export_logs_dir = export_dir
            rm.has_bill_of_health.return_value = True
            rm.archive.return_value = (None, None, None, False)
            provider = mock.Mock()
            provider.get_uri.return_value = "gs://my-bucket/team-a/"
            with (
                mock.patch(
                    "chs.cli.StorageProviderFactory.get_provider_for_run",
                    return_value=provider,
                ),
                mock.patch("chs.cli.Console", autospec=True) as mock_console,
            ):
                cli._archive_and_log(rm)

        printed = [
            c.args[0] for c in mock_console.return_value.print.call_args_list
        ]
        self.assertIn(
            "[Warning] Failed to upload results to GCS (gs://my-bucket/team-a)."
            " Results are preserved locally:",
            printed[0],
        )
        self.assertIn(
            os.path.join(export_dir, "bill_of_health.html"), printed[1]
        )
        self.assertIn(
            os.path.join(export_dir, "bill_of_health.json"), printed[2]
        )
        rm.update_run_entry.assert_called_once_with(
            {
                "boh_html_uri": os.path.join(export_dir, "bill_of_health.html"),
                "boh_uri": os.path.join(export_dir, "bill_of_health.json"),
            }
        )

    def test_format_local_display_path(self):
        """_format_local_display_path formats paths relative to cwd."""
        cwd = pathlib.Path.cwd().resolve()
        subpath = str(cwd / "chs-output" / "ws1" / "bill_of_health.html")
        self.assertEqual(
            cli._format_local_display_path(subpath),
            "./chs-output/ws1/bill_of_health.html",
        )

        # Path outside cwd remains unchanged
        outside = "/tmp/somewhere/outside/bill_of_health.html"
        self.assertEqual(cli._format_local_display_path(outside), outside)

        # Remote URIs remain unchanged
        gcs_uri = "gs://bucket/path/bill_of_health.html"
        self.assertEqual(cli._format_local_display_path(gcs_uri), gcs_uri)

        # None remains None
        self.assertIsNone(cli._format_local_display_path(None))

    @mock.patch("chs.cli._export_logs", autospec=True)
    def test_archive_and_log_formats_local_paths_relative_to_cwd(
        self, mock_export
    ):
        """_archive_and_log prints local paths as ./chs-output/..."""
        cwd = pathlib.Path.cwd().resolve()
        rel_export_dir = "chs-output/ws1"
        export_dir = str(cwd / rel_export_dir)
        rm = mock.create_autospec(RambleManager, instance=True)
        rm.workspace_name = "ws1"
        rm.path = "/workspace/ws1"
        rm.export_logs_dir = export_dir
        rm.has_bill_of_health.return_value = True
        rm.archive.return_value = (None, None, None, False)

        with (
            mock.patch(
                "chs.cli.StorageProviderFactory.get_provider_for_run",
                return_value=None,
            ),
            mock.patch("chs.cli.Console", autospec=True) as mock_console,
            mock.patch(
                "chs.cli._find_local_artifact",
                side_effect=lambda name, *dirs: os.path.join(export_dir, name),
            ),
        ):
            cli._archive_and_log(rm)

        printed = [
            c.args[0] for c in mock_console.return_value.print.call_args_list
        ]
        expected_html = f"./{rel_export_dir}/bill_of_health.html"
        expected_json = f"./{rel_export_dir}/bill_of_health.json"
        self.assertIn(expected_html, printed[0])
        self.assertIn(expected_json, printed[1])
        # Local paths must not be OSC 8 hyperlinks (no file-scheme handler).
        self.assertNotIn("[link=", printed[0])
        self.assertNotIn("[link=", printed[1])
        rm.update_run_entry.assert_called_once_with(
            {
                "boh_html_uri": os.path.join(export_dir, "bill_of_health.html"),
                "boh_uri": os.path.join(export_dir, "bill_of_health.json"),
            }
        )

    @mock.patch("chs.cli._export_logs", autospec=True)
    def test_archive_and_log_links_gcs_locations(self, mock_export):
        """_archive_and_log hyperlinks GCS locations to the Cloud console."""
        rm = mock.create_autospec(RambleManager, instance=True)
        rm.workspace_name = "ws1"
        rm.path = "/nonexistent/ws1"
        rm.export_logs_dir = None
        rm.has_bill_of_health.return_value = True
        rm.archive.return_value = (
            "gs://bucket/p/workspaces/ws1.tar.gz",
            "gs://bucket/p/bills_of_health/ws1.json",
            "gs://bucket/p/bills_of_health/ws1.html",
            True,
        )
        with (
            mock.patch(
                "chs.cli.StorageProviderFactory.get_provider_for_run",
                return_value=mock.Mock(),
            ),
            mock.patch("chs.cli.Console", autospec=True) as mock_console,
        ):
            cli._archive_and_log(rm)

        calls = mock_console.return_value.print.call_args_list
        printed = [c.args[0] for c in calls]
        self.assertIn(
            "[link=https://storage.cloud.google.com/bucket/p/bills_of_health/"
            "ws1.html]gs://bucket/p/bills_of_health/ws1.html[/link]",
            printed[0],
        )
        self.assertIn(
            "[link=https://storage.cloud.google.com/bucket/p/bills_of_health/"
            "ws1.json]gs://bucket/p/bills_of_health/ws1.json[/link]",
            printed[1],
        )
        for call in calls:
            self.assertTrue(call.kwargs.get("soft_wrap"))

    def test_show_results_when_boh_missing_and_no_storage_configured(self):
        """When BoH is missing and storage unconfigured, raise clear error."""
        app_data = mock.Mock()
        app_data.get_workspaces_dir.return_value = "/tmp/nonexistent-workspaces"
        app_data.resolve_run_id.return_value = "ws_test"
        app_data.get_run.return_value = {}
        app_data.get_storage_config.return_value = {}

        with mock.patch(
            "chs.cli.StorageProviderFactory.get_provider_for_run",
            return_value=None,
        ):
            with self.assertRaises(
                exceptions.BillOfHealthNotFoundException
            ) as ctx:
                cli._show_results(app_data, "ws_test")

        msg = str(ctx.exception)
        self.assertIn("Results not found for run 'ws_test'.", msg)
        self.assertNotIn("No storage provider configured", msg)
        self.assertNotIn("chs configure storage", msg)

    @mock.patch("chs.cli.ResultsViewer", autospec=True)
    def test_show_results_local_run_passes_uris_to_viewer(
        self, mock_viewer_cls
    ):
        """Test _show_results passes local html and json URIs."""
        mock_viewer = mock_viewer_cls.return_value
        mock_viewer.boh = pb.BillOfHealth()

        with tempfile.TemporaryDirectory() as tmp_dir:
            ws_name = "test-ws-local"
            ws_path = pathlib.Path(tmp_dir) / "workspaces" / ws_name
            ws_path.mkdir(parents=True)
            boh_json = ws_path / "bill_of_health.json"
            boh_html = ws_path / "bill_of_health.html"
            boh_json.write_text("{}")
            boh_html.write_text("<html></html>")

            app_data = mock.Mock()
            app_data.get_workspaces_dir.return_value = str(
                pathlib.Path(tmp_dir) / "workspaces"
            )
            app_data.resolve_run_id.return_value = ws_name
            app_data.get_run.return_value = {
                "workspace_path": str(ws_path),
                "boh_html_uri": str(boh_html),
                "boh_uri": str(boh_json),
            }

            with mock.patch(
                "chs.cli.StorageProviderFactory.get_provider_for_run",
                return_value=None,
            ):
                cli._show_results(app_data, ws_name)

            expected_html = cli._format_local_display_path(str(boh_html))
            expected_json = cli._format_local_display_path(str(boh_json))

            mock_viewer_cls.assert_called_once_with(
                boh_json,
                boh_html_uri=expected_html,
                boh_json_uri=expected_json,
            )
            mock_viewer.show_summary_and_details.assert_called_once()

    @mock.patch("chs.cli.ResultsViewer", autospec=True)
    def test_show_results_gcs_run_passes_uris_to_viewer(self, mock_viewer_cls):
        """Test _show_results passes GCS html and json URIs."""
        mock_viewer = mock_viewer_cls.return_value
        mock_viewer.boh = pb.BillOfHealth()

        with tempfile.TemporaryDirectory() as tmp_dir:
            ws_name = "test-ws-gcs"
            ws_path = pathlib.Path(tmp_dir) / "workspaces" / ws_name
            ws_path.mkdir(parents=True)
            boh_json = ws_path / "bill_of_health.json"
            boh_json.write_text("{}")

            app_data = mock.Mock()
            app_data.get_workspaces_dir.return_value = str(
                pathlib.Path(tmp_dir) / "workspaces"
            )
            app_data.resolve_run_id.return_value = ws_name
            app_data.get_run.return_value = {
                "boh_html_uri": f"gs://my-bucket/prefix/bills_of_health/{ws_name}.html",
                "boh_uri": f"gs://my-bucket/prefix/bills_of_health/{ws_name}.json",
            }

            with mock.patch(
                "chs.cli.StorageProviderFactory.get_provider_for_run",
                return_value=None,
            ):
                cli._show_results(app_data, ws_name)

            mock_viewer_cls.assert_called_once_with(
                boh_json,
                boh_html_uri=f"gs://my-bucket/prefix/bills_of_health/{ws_name}.html",
                boh_json_uri=f"gs://my-bucket/prefix/bills_of_health/{ws_name}.json",
            )
            mock_viewer.show_summary_and_details.assert_called_once()

    @mock.patch("chs.cli.ResultsViewer", autospec=True)
    def test_show_results_skips_stale_local_paths(self, mock_viewer_cls):
        """Deleted export paths fall back to the internal workspace."""
        mock_viewer = mock_viewer_cls.return_value
        mock_viewer.boh = pb.BillOfHealth()

        with tempfile.TemporaryDirectory() as tmp_dir:
            ws_name = "test-ws-stale"
            workspaces = pathlib.Path(tmp_dir) / "workspaces"
            boh_json = workspaces / ws_name / "bill_of_health.json"
            boh_json.parent.mkdir(parents=True)
            boh_json.write_text("{}")
            deleted_export = pathlib.Path(tmp_dir) / "chs-output" / ws_name

            app_data = mock.Mock()
            app_data.get_workspaces_dir.return_value = str(workspaces)
            app_data.resolve_run_id.return_value = ws_name
            app_data.get_run.return_value = {
                "boh_html_uri": str(deleted_export / "bill_of_health.html"),
                "boh_uri": str(deleted_export / "bill_of_health.json"),
            }

            with mock.patch(
                "chs.cli.StorageProviderFactory.get_provider_for_run",
                return_value=None,
            ):
                cli._show_results(app_data, ws_name)
            mock_viewer_cls.assert_called_once_with(
                boh_json,
                boh_html_uri=None,
                boh_json_uri=cli._format_local_display_path(str(boh_json)),
            )

    @mock.patch("chs.cli._analyze_action", autospec=True)
    def test_run_command_rechecks_targeted_nodes_before_submitting(
        self, mock_analyze_action
    ):
        """Test 'run' fails if a confirmed node got targeted by another run."""
        targeting = {"other-run-ab12": ["node2"]}
        self.mock_scheduler.get_nodes_under_test.return_value = [
            "node1",
            "node2",
        ]
        self.mock_scheduler.find_runs_targeting_nodes.return_value = targeting

        result = self.runner.invoke(
            cli.cli,
            ["run", "--suite", "suite1", "--auto-approve"],
            standalone_mode=False,
        )

        self.assertIsInstance(
            result.exception, exceptions.NodesTargetedByOtherRunsException
        )
        self.assertEqual(result.exception.conflicts, targeting)
        self.mock_scheduler.find_runs_targeting_nodes.assert_called_once_with(
            ["node1", "node2"]
        )
        self.mock_app_data.update_run.assert_not_called()
        self.mock_rm_instance.execute_experiments.assert_not_called()
        mock_analyze_action.assert_not_called()

    def test_nodes_targeted_by_other_runs_message_lists_only_run_ids(self):
        """Test the conflict message names the runs but not their nodes."""
        error = exceptions.NodesTargetedByOtherRunsException(
            {"run-b": ["node3", "node4"], "run-a": ["node1"]}
        )

        self.assertEqual(
            str(error),
            "Requested nodes are targeted by other active CHS runs: "
            "run-a, run-b.\n"
            "Target a disjoint set of nodes, or wait for those runs to "
            "finish. If a run is stale, terminate it with "
            "`chs cancel <run_id>`.",
        )
        self.assertEqual(
            error.conflicts, {"run-b": ["node3", "node4"], "run-a": ["node1"]}
        )

    @mock.patch("chs.cli._analyze_action", autospec=True)
    def test_run_command_registers_submits_and_analyzes_under_guard(
        self, mock_analyze_action
    ):
        """Test the run is registered, submitted and analyzed in order."""
        run_id = "mockcluster-suite1-mockhash"
        calls = []
        self.mock_app_data.update_run.side_effect = lambda *args, **kwargs: (
            calls.append("register_run")
        )
        self.mock_rm_instance.execute_experiments.side_effect = (
            lambda **kwargs: calls.append("submit") or ["job1"]
        )
        mock_analyze_action.side_effect = lambda *args: calls.append("analyze")

        result = self.runner.invoke(
            cli.cli,
            ["run", "--suite", "suite1", "--auto-approve"],
            standalone_mode=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(calls, ["register_run", "submit", "analyze"])
        self.assertEqual(
            self.mock_scheduler.setup_scheduler_resources.call_args.kwargs[
                "run_id"
            ],
            run_id,
        )
        mock_analyze_action.assert_called_once_with(self.mock_app_data, run_id)

    def test_run_command_failure_tears_down_only_its_own_run(self):
        """Test a failed submission tears down nothing but its own run."""
        run_id = "mockcluster-suite1-mockhash"
        self.mock_rm_instance.setup_workspace.side_effect = OSError(
            "Permission denied"
        )
        manager = mock.Mock()
        for name in ("cancel_run", "cancel_all_runs"):
            manager.attach_mock(getattr(self.mock_scheduler, name), name)

        result = self.runner.invoke(
            cli.cli, ["run", "--suite", "suite1", "--auto-approve"]
        )

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(
            manager.mock_calls,
            [
                mock.call.cancel_run(run_id),
            ],
        )

    def _attach_run_teardown_calls(self) -> mock.Mock:
        manager = mock.Mock()
        for name in ("cancel_run", "cancel_all_runs"):
            manager.attach_mock(getattr(self.mock_scheduler, name), name)
        return manager

    def _assert_torn_down_once(self, manager: mock.Mock, run_id: str) -> None:
        self.assertEqual(
            manager.mock_calls,
            [
                mock.call.cancel_run(run_id),
            ],
        )

    def test_run_command_early_analysis_failure_tears_down_once(self):
        """Test a failure before analysis starts waiting still tears down."""
        run_id = "mockcluster-suite1-mockhash"
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_app_data.resolve_run_id.side_effect = (
            exceptions.WorkspaceNotFoundException("gone")
        )
        manager = self._attach_run_teardown_calls()

        with mock.patch("chs.cli.theme.get_progress_spinner") as spinner:
            result = self.runner.invoke(
                cli.cli, ["run", "--suite", "suite1", "--auto-approve"]
            )

        self.assertEqual(result.exit_code, 1)
        self._assert_torn_down_once(manager, run_id)
        teardown_spinners = [
            c
            for c in spinner.call_args_list
            if "Cleaning up CHS resources" in c.args[0]
        ]
        self.assertEqual(len(teardown_spinners), 1)

    def test_run_command_failure_while_waiting_tears_down_once(self):
        """Test a failure while waiting for jobs tears the run down once."""
        run_id = "mockcluster-suite1-mockhash"
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_app_data.get_run.return_value = {}
        self.mock_rm_instance.wait_for_jobs.side_effect = KeyboardInterrupt()
        manager = self._attach_run_teardown_calls()
        workspace = self.enterContext(tempfile.TemporaryDirectory())
        self.mock_rm_instance.path = workspace

        result = self.runner.invoke(
            cli.cli, ["run", "--suite", "suite1", "--auto-approve"]
        )

        self.assertEqual(result.exit_code, 130)
        self._assert_torn_down_once(manager, run_id)
        self.mock_rm_instance.cancel_experiments.assert_not_called()

    @mock.patch("chs.cli._analyze_action")
    def test_run_command_keyboard_interrupt_analysis_exits_130(
        self, mock_analyze_action
    ):
        """Test Ctrl+C during analysis exits 130 after one teardown."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.name = "slurm"
        mock_analyze_action.side_effect = KeyboardInterrupt()
        manager = self._attach_run_teardown_calls()

        result = self.runner.invoke(
            cli.cli,
            ["-v", "run", "--suite", "suite1", "--auto-approve"],
        )
        self.assertEqual(result.exit_code, 130)
        self._assert_torn_down_once(manager, "mockcluster-suite1-mockhash")

    def test_cancel_action_exception_handling(self):
        """Test that cancel_action wraps exceptions in ClickException."""
        self.mock_app_data.get_latest_run_id.return_value = "some-run-id"
        self.mock_scheduler.cancel_run.side_effect = (
            exceptions.ChsSchedulerException("Cancellation failed")
        )
        result = self.runner.invoke(cli.cli, ["cancel", "--auto-approve"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Cancellation failed", result.output)

    def test_cancel_uses_the_scheduler_the_run_ran_on(self):
        """A local run is cancelled on its recorded scheduler."""
        self.mock_app_data.resolve_run_id.side_effect = None
        self.mock_app_data.resolve_run_id.return_value = "ws-run1"
        self.mock_app_data.get_run.return_value = {"scheduler": "standalone"}

        result = self.runner.invoke(cli.cli, ["cancel", "run1", "-y"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.mock_app_data.get_run.assert_called_once_with("ws-run1")
        self.mock_get_scheduler.assert_called_once_with("standalone")
        self.mock_detect_scheduler.assert_not_called()
        self.mock_scheduler.cancel_run.assert_called_once_with("run1")

    def test_cancel_auto_detects_without_a_recorded_scheduler(self):
        """Unknown runs and runs without a recorded scheduler auto-detect."""
        for resolve, entry in (
            (exceptions.WorkspaceNotFoundException("missing"), None),
            (None, {}),
            (None, {"scheduler": "bogus"}),
        ):
            with self.subTest(entry=entry):
                self.mock_get_scheduler.reset_mock()
                self.mock_detect_scheduler.reset_mock()
                self.mock_app_data.resolve_run_id.side_effect = resolve
                self.mock_app_data.resolve_run_id.return_value = "ws-run1"
                self.mock_app_data.get_run.return_value = entry

                result = self.runner.invoke(cli.cli, ["cancel", "run1", "-y"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.mock_detect_scheduler.assert_called_once_with()
                self.mock_get_scheduler.assert_called_once_with("slurm")

    def test_cancel_action_kube_error_raises_cancellation_exception(self):
        """Test cancel_action wraps a Kubernetes API error of any step."""
        err = KubeError(
            "Error from server (Forbidden): jobs.batch is " "forbidden",
            status=403,
        )
        for cancel, step in (
            (lambda: cli.cancel_action("run1"), "cancel_run"),
            (cli.cancel_all_action, "cancel_all_runs"),
        ):
            with self.subTest(step=step):
                scheduler = mock.Mock()
                getattr(scheduler, step).side_effect = err
                self.mock_get_scheduler.return_value = scheduler

                with self.assertRaises(
                    exceptions.ChsCancellationException
                ) as ctx:
                    cancel()

                self.assertEqual(
                    str(ctx.exception),
                    "CHS cancellation failed: KubeError('Error from server "
                    "(Forbidden): jobs.batch is forbidden')",
                )
                self.assertIs(ctx.exception.__cause__, err)
                self.mock_get_scheduler.assert_called_with("slurm")

    def _attach_teardown_calls(self) -> mock.Mock:
        manager = mock.Mock()
        for name in ("cancel_run", "cancel_all_runs"):
            manager.attach_mock(getattr(self.mock_scheduler, name), name)
        return manager

    def test_cancel_without_run_id_cancels_latest_local_run(self):
        """Test 'cancel' without arguments tears down the latest local run."""
        self.mock_app_data.get_latest_run_id.return_value = "latest-run-ab12"
        manager = self._attach_teardown_calls()

        result = self.runner.invoke(cli.cli, ["cancel", "-y"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            manager.mock_calls,
            [
                mock.call.cancel_run("latest-run-ab12"),
            ],
        )

    def test_cancel_run_id_not_known_locally_cancels_on_cluster(self):
        """Test 'cancel RUN_ID' works for another user's run."""
        self.mock_app_data.resolve_run_id.side_effect = (
            exceptions.WorkspaceNotFoundException("not found")
        )
        manager = self._attach_teardown_calls()

        result = self.runner.invoke(cli.cli, ["cancel", "other-run-cd34", "-y"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            manager.mock_calls,
            [
                mock.call.cancel_run("other-run-cd34"),
            ],
        )
        self.mock_app_data.get_latest_run_id.assert_not_called()
        self.mock_rm_cls.assert_not_called()

    def test_cancel_all_wipes_every_chs_run_on_the_cluster(self):
        """Test 'cancel --all' keeps the cluster-wide teardown."""
        manager = self._attach_teardown_calls()

        result = self.runner.invoke(cli.cli, ["cancel", "--all", "-y"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            manager.mock_calls,
            [
                mock.call.cancel_all_runs(),
            ],
        )

    def test_cancel_run_id_and_all_are_mutually_exclusive(self):
        """Test 'cancel' rejects RUN_ID combined with --all."""
        result = self.runner.invoke(
            cli.cli, ["cancel", "other-run-cd34", "--all", "-y"]
        )

        self.assertEqual(result.exit_code, 2)
        self.assertIn("mutually exclusive", result.output)
        self.mock_get_scheduler.assert_not_called()

    def test_cancel_without_local_runs_suggests_run_id_or_all(self):
        """Test 'cancel' without arguments or local runs explains options."""
        no_runs = exceptions.NoPreviousRunsFoundException("No previous runs")
        self.mock_app_data.get_latest_run_id.side_effect = no_runs

        result = self.runner.invoke(
            cli.cli, ["cancel", "-y"], standalone_mode=False
        )

        self.assertIsInstance(
            result.exception, exceptions.NoPreviousRunsFoundException
        )
        self.assertIn("Pass a RUN_ID", str(result.exception))
        self.assertIn("--all", str(result.exception))
        self.assertIs(result.exception.__cause__, no_runs)
        self.mock_app_data.get_all_runs.assert_not_called()
        self.mock_get_scheduler.assert_not_called()

    def test_cancel_rejects_invalid_run_id_before_touching_cluster(self):
        """Test 'cancel' rejects run IDs that cannot be CHS run IDs."""
        for run_id in (
            "Upper-Case",
            "under_score",
            "-leading-dash",
            "trailing-dash-",
            "20261014-084121_cluster-lite-ab12",
            "a" * 64,
        ):
            with self.subTest(run_id=run_id):
                result = self.runner.invoke(
                    cli.cli, ["cancel", "-y", "--", run_id]
                )

                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("not a valid CHS run ID", result.output)
                self.mock_app_data.get_run.assert_not_called()
                self.mock_get_scheduler.assert_not_called()

    def test_cancel_accepts_longest_valid_run_id(self):
        """Test 'cancel' accepts a 63-character run ID."""
        run_id = "a" * 63

        result = self.runner.invoke(cli.cli, ["cancel", run_id, "-y"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.mock_scheduler.cancel_run.assert_called_once_with(run_id)

    def test_cancel_of_run_absent_from_cluster_warns(self):
        """Test cancel warns instead of reporting success for unknown runs."""
        self.mock_scheduler.cancel_run.return_value = False
        self.mock_app_data.get_latest_run_id.return_value = "latest-run-ab12"
        for args, run_id in (
            (["cancel", "typo-run", "-y"], "typo-run"),
            (["cancel", "-y"], "latest-run-ab12"),
        ):
            with self.subTest(args=args):
                result = self.runner.invoke(cli.cli, args)

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn(
                    f"No active CHS run '{run_id}' was found on the cluster.",
                    " ".join(result.output.split()),
                )
                self.assertNotIn("Successfully canceled", result.output)

    def test_cancel_reports_failed_deletions_as_failure(self):
        """Test cancel fails without success if some objects survived."""
        cases = [
            (
                ["cancel", "run-ab12", "-y"],
                "cancel_run",
                "Failed to delete the Jobs of CHS run 'run-ab12'. Run "
                "`chs cancel run-ab12` again to retry.",
            ),
            (
                ["cancel", "--all", "-y"],
                "cancel_all_runs",
                "Failed to delete the CHS chs namespace. Run "
                "`chs cancel --all` again to retry.",
            ),
        ]
        for args, step, message in cases:
            with self.subTest(step=step):
                error = exceptions.ChsCancellationException(message)
                getattr(self.mock_scheduler, step).side_effect = error

                result = self.runner.invoke(cli.cli, args)

                self.assertNotEqual(result.exit_code, 0, result.output)
                self.assertIn(message, " ".join(result.output.split()))
                self.assertNotIn("Successfully canceled", result.output)

    def test_cancel_prompts_state_their_blast_radius(self):
        """Test cancel prompts distinguish one run from the whole cluster."""
        cases = [
            (lambda: cli.safe_cancel_action("run1"), ["run 'run1'"]),
            (
                cli.safe_cancel_all_action,
                ["ALL CHS runs on the cluster", "other users"],
            ),
        ]
        for cancel, expected_fragments in cases:
            with self.subTest(expected_fragments=expected_fragments):
                with (
                    mock.patch("chs.cli.sys.stdin") as mock_stdin,
                    mock.patch(
                        "chs.cli.click.confirm", return_value=False
                    ) as mock_confirm,
                    mock.patch(
                        "chs.cli.utils.is_in_cluster", return_value=False
                    ),
                ):
                    mock_stdin.isatty.return_value = True
                    with self.assertRaises(click.Abort):
                        cancel()

                prompt = mock_confirm.call_args.args[0]
                for fragment in expected_fragments:
                    self.assertIn(fragment, prompt)
                self.mock_get_scheduler.assert_not_called()

    @mock.patch("chs.cli._archive_and_log", autospec=True)
    def test_cancel_only_cleans_up_the_cluster(self, mock_archive_and_log):
        """Test 'cancel' neither archives nor changes the local registry."""
        self.mock_app_data.get_latest_run_id.return_value = "latest-run-ab12"
        for args in (["cancel"], ["cancel", "run-cd34"], ["cancel", "--all"]):
            with self.subTest(args=args):
                result = self.runner.invoke(cli.cli, [*args, "-y"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.mock_app_data.update_run.assert_not_called()
                self.mock_rm_cls.assert_not_called()
                mock_archive_and_log.assert_not_called()

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

            _analyze_in_guard(self.mock_app_data, "run1", self.mock_scheduler)
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

            _analyze_in_guard(self.mock_app_data, "run1", self.mock_scheduler)
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

            _analyze_in_guard(self.mock_app_data, "run1", self.mock_scheduler)
            mock_rm.archive.assert_called_once()

    def test_analyze_action_kube_error_raises_analysis_exception(self):
        """Test _analyze_action wraps a Kubernetes API error like a CPE."""
        err = KubeError(
            "Error from server (InternalError): etcdserver: request timed out",
            status=500,
        )
        self.mock_app_data.get_run.return_value = {}
        self.mock_rm_instance.wait_for_jobs.side_effect = err

        with tempfile.TemporaryDirectory() as ws_dir:
            self.mock_rm_instance.path = ws_dir
            with self.assertRaises(exceptions.ChsAnalysisException) as ctx:
                _analyze_in_guard(
                    self.mock_app_data, "run1", self.mock_scheduler
                )

        self.assertEqual(
            str(ctx.exception), f"CHS analysis failed for run_id=run1: {err}"
        )
        self.assertIs(ctx.exception.__cause__, err)
        self.mock_rm_instance.analyze_experiments.assert_not_called()
        self.mock_scheduler.cancel_run.assert_called_once_with("run1")

    def test_analyze_action_keyboard_interrupt_while_waiting_cancels_run(self):
        """Test Ctrl+C while waiting for jobs cancels the run."""
        self.mock_app_data.get_run.return_value = {}
        self.mock_rm_instance.wait_for_jobs.side_effect = KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as ws_dir:
            self.mock_rm_instance.path = ws_dir
            with self.assertRaises(KeyboardInterrupt):
                _analyze_in_guard(
                    self.mock_app_data, "run1", self.mock_scheduler
                )

        self.mock_scheduler.cancel_run.assert_called_once_with("run1")

    @mock.patch("chs.cli._archive_and_log", side_effect=KeyboardInterrupt())
    @mock.patch("chs.cli._show_results")
    def test_analyze_action_keyboard_interrupt_after_cleanup_tears_down_again(
        self, mock_show_results, mock_archive_and_log
    ):
        """Test Ctrl+C after the cleanup repeats the idempotent teardown."""
        self.mock_app_data.get_run.return_value = {}

        with tempfile.TemporaryDirectory() as ws_dir:
            self.mock_rm_instance.path = ws_dir
            with self.assertRaises(KeyboardInterrupt):
                _analyze_in_guard(
                    self.mock_app_data, "run1", self.mock_scheduler
                )

        mock_show_results.assert_called_once()
        mock_archive_and_log.assert_called_once()
        self.mock_rm_instance.cancel_experiments.assert_called_once()
        self.mock_scheduler.cancel_run.assert_called_once_with("run1")

    def test_analyze_action_interrupt_after_final_boh_cleans_up_once(self):
        """Test Ctrl+C after the final BoH doesn't start a second cleanup."""
        self.mock_app_data.get_run.return_value = {}
        self.mock_app_data.update_run.side_effect = KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as ws_dir:
            self.mock_rm_instance.path = ws_dir
            with self.assertRaises(KeyboardInterrupt):
                _analyze_in_guard(
                    self.mock_app_data, "run1", self.mock_scheduler
                )

        self.mock_scheduler.cancel_run.assert_called_once_with("run1")
        self.mock_rm_instance.cancel_experiments.assert_not_called()

    def test_background_cleanup_logs_kube_error_as_warning(self):
        """Test a Kubernetes API error of the cleanup is only logged."""
        self.enterContext(mock.patch.object(cli.theme, "get_progress_spinner"))
        err = KubeError(
            'Error from server (Forbidden): namespaces "chs" is forbidden',
            status=403,
        )
        ramble_manager = mock.Mock()
        ramble_manager.get_scheduler_type.return_value = "kubernetes"
        ramble_manager.cancel_experiments.side_effect = err

        with mock.patch.object(background_cleanup, "logger") as mock_logger:
            with background_cleanup.BackgroundCleanup("run1") as cleanup:
                cleanup.start(ramble_manager)

        mock_logger.warning.assert_called_once_with(
            "Failed to clean CHS resources. Run 'chs cancel %s' to finish "
            "cleaning up. Error: %s",
            "run1",
            err,
        )
        ramble_manager.cancel_experiments.assert_called_once_with(
            scheduler_name="kubernetes"
        )

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
            os.makedirs(os.path.join(tmpdir, "latest_run", "test_results"))
            result = self.runner.invoke(cli.cli, ["results"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("failed to resolve storage provider", result.output)

    @mock.patch("chs.storage.StorageProviderFactory.get_active_provider")
    @mock.patch("chs.storage.StorageProviderFactory.get_provider_for_run")
    @mock.patch("chs.cli.ResultsViewer", autospec=True)
    def test_results_command_download_success(
        self, mock_viewer, mock_get_provider_for_run, mock_get_active_provider
    ):
        """Test results command downloads BoH from storage if missing."""
        mock_viewer.return_value.boh = pb.BillOfHealth()
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"
        self.mock_app_data.get_run.return_value = None

        # Simulate get_provider_for_run returning None (because get_run is None)
        # and falling back to get_active_provider()
        mock_get_provider_for_run.return_value = None
        mock_provider = mock.Mock()
        mock_provider.get_uri.side_effect = lambda p: f"gs://bucket/{p}"
        mock_provider.exists.side_effect = lambda p: True
        mock_get_active_provider.return_value = mock_provider

        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_app_data.get_workspaces_dir.return_value = tmpdir
            os.makedirs(os.path.join(tmpdir, "latest_run", "test_results"))

            result = self.runner.invoke(cli.cli, ["results"])
            self.assertEqual(result.exit_code, 0)
            mock_provider.download_file.assert_called_once()
            args, _ = mock_provider.download_file.call_args
            self.assertEqual(args[0], "bills_of_health/latest_run.json")
            mock_viewer.assert_called_once()
            self.assertEqual(
                mock_viewer.call_args.kwargs["boh_html_uri"],
                "gs://bucket/bills_of_health/latest_run.html",
            )
            self.assertEqual(
                mock_viewer.call_args.kwargs["boh_json_uri"],
                "gs://bucket/bills_of_health/latest_run.json",
            )

    @mock.patch("chs.cli._analyze_action", autospec=True)
    def test_run_command_sets_telemetry_run(self, mock_analyze_action):
        """Test 'run' tags telemetry with the new run."""
        self.enterContext(
            mock.patch(
                "chs.cli.utils.generate_random_suffix", return_value="a3f9"
            )
        )
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]

        result = self.runner.invoke(
            cli.cli,
            ["run", "--suite", "suite1", "--auto-approve"],
            standalone_mode=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_analyze_action.assert_called_once()
        self.assertEqual(telemetry._state.run_id, MOCK_TELEMETRY_RUN_ID)

    @mock.patch("chs.cli._analyze_action", autospec=True)
    def test_run_command_sends_run_metrics(self, mock_analyze_action):
        """Test 'run' reports the command, cluster and scheduled tests."""
        send = self.enterContext(
            mock.patch.object(telemetry, "send_metric", autospec=True)
        )
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        self.mock_scheduler.get_nodes_under_test.return_value = ["n1", "n2"]
        self.mock_scheduler.name = "kubernetes"
        self.mock_scheduler.project_id = "my-project"
        self.mock_scheduler.node_to_zone = {
            "n1": "us-central1-a",
            "n2": "us-central1-a",
        }

        result = self.runner.invoke(
            cli.cli,
            ["run", "--suite", "suite1", "--auto-approve"],
            standalone_mode=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_analyze_action.assert_called_once()
        sent = {c.args[0]: c.args[1] for c in send.call_args_list}
        M = telemetry.MetricName
        self.assertEqual(sent[M.PARENT_COMMAND], "run")
        self.assertEqual(sent[M.COMMAND_FLAGS], "auto-approve,suite")
        self.assertEqual(sent[M.CLUSTER_TYPE], "GKE")
        self.assertEqual(sent[M.STORAGE_TYPE], "GCS")
        self.assertEqual(sent[M.MACHINE_TYPE], "a3-megagpu-8g")
        self.assertEqual(sent[M.NODE_COUNT], 2)
        self.assertEqual(sent[M.ZONE], "us-central1-a")
        self.assertEqual(sent[M.IS_TEST_DATA], "false")
        self.assertEqual(sent[M.TEST_SUITE], "suite1")
        test_names = sent[M.TEST_NAME].split(",")
        self.assertTrue(test_names)
        self.assertEqual(test_names, sorted(test_names))
        self.assertEqual(telemetry._state.project_id, "my-project")

    @mock.patch("chs.cli._analyze_action", autospec=True)
    def test_run_command_records_run_after_run_id_is_set(
        self, mock_analyze_action
    ):
        """Test 'run' reports its metrics only once the run has an ID."""
        run_ids_when_recorded = []
        self.enterContext(
            mock.patch.object(
                cli.telemetry_reporting,
                "record_run",
                side_effect=lambda *_: run_ids_when_recorded.append(
                    telemetry._state.run_id
                ),
            )
        )
        self.enterContext(
            mock.patch(
                "chs.cli.utils.generate_random_suffix", return_value="a3f9"
            )
        )
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]

        result = self.runner.invoke(
            cli.cli,
            ["run", "--suite", "suite1", "--auto-approve"],
            standalone_mode=False,
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run_ids_when_recorded, [MOCK_TELEMETRY_RUN_ID])

    @mock.patch("chs.cli._analyze_action", autospec=True)
    def test_run_command_declined_at_prompt_sends_no_run_metrics(
        self, mock_analyze_action
    ):
        """Test a run the user declines at the prompt reports no run."""
        record_run = self.enterContext(
            mock.patch.object(cli.telemetry_reporting, "record_run")
        )
        # A mock sys makes stdin look like a terminal, so the prompts show.
        self.enterContext(mock.patch.object(cli, "sys"))
        self.enterContext(
            mock.patch.object(
                cli.click,
                "confirm",
                side_effect=lambda prompt, **_: (
                    prompt != "Proceed with execution?"
                ),
            )
        )

        result = self.runner.invoke(cli.cli, ["run", "--suite", "suite1"])

        self.assertEqual(result.exit_code, 0, result.output)
        mock_analyze_action.assert_not_called()
        record_run.assert_not_called()
        self.assertIsNone(telemetry._state.run_id)

    def test_run_command_reports_ctrl_c(self):
        """Test Ctrl+C in 'run' is reported as an interrupt, exit code 130."""
        send = self.enterContext(
            mock.patch.object(telemetry, "send_metric", autospec=True)
        )
        self.mock_detect_scheduler.side_effect = KeyboardInterrupt

        result = self.runner.invoke(cli.cli, ["run", "--auto-approve"])

        self.assertEqual(result.exit_code, 130)
        send.assert_any_call(
            telemetry.MetricName.ERROR_CATEGORY, "KeyboardInterrupt"
        )
        send.assert_called_with(telemetry.MetricName.EXIT_CODE, 130)

    def test_run_command_reports_scheduler_error(self):
        """Test 'run' reports the error when scheduler detection fails."""
        send = self.enterContext(
            mock.patch.object(telemetry, "send_metric", autospec=True)
        )
        self.mock_detect_scheduler.side_effect = (
            exceptions.ChsSchedulerException("no scheduler")
        )

        result = self.runner.invoke(cli.cli, ["run", "--auto-approve"])

        self.assertNotEqual(result.exit_code, 0)
        send.assert_any_call(
            telemetry.MetricName.ERROR_CATEGORY, "ChsSchedulerException"
        )

    def test_analyze_action_sets_telemetry_run(self):
        """Test _analyze_action tags telemetry, even if analysis fails."""
        self.mock_app_data.resolve_run_id.side_effect = None
        self.mock_app_data.resolve_run_id.return_value = (
            MOCK_TELEMETRY_WORKSPACE_NAME
        )

        # The mocked workspace is not on disk, so analysis fails after tagging.
        with self.assertRaises(exceptions.WorkspaceNotFoundException):
            cli._analyze_action(self.mock_app_data, "mockcluster-lite-a3f9")

        self.assertEqual(telemetry._state.run_id, MOCK_TELEMETRY_RUN_ID)

    def test_show_results_sets_telemetry_run(self):
        """Test _show_results tags telemetry with a registered run."""
        self.enterContext(
            mock.patch("chs.cli.StorageProviderFactory", autospec=True)
        )
        mock_viewer = self.enterContext(
            mock.patch("chs.cli.ResultsViewer", autospec=True)
        )
        mock_viewer.return_value.boh = pb.BillOfHealth()
        self.mock_app_data.resolve_run_id.side_effect = None
        self.mock_app_data.resolve_run_id.return_value = (
            MOCK_TELEMETRY_WORKSPACE_NAME
        )

        cli._show_results(self.mock_app_data, "mockcluster-lite-a3f9")

        self.assertEqual(telemetry._state.run_id, MOCK_TELEMETRY_RUN_ID)

    def test_show_results_skips_telemetry_run_for_unregistered_run(self):
        """Test _show_results skips runs missing from the local registry."""
        self.enterContext(
            mock.patch("chs.cli.StorageProviderFactory", autospec=True)
        )
        mock_viewer = self.enterContext(
            mock.patch("chs.cli.ResultsViewer", autospec=True)
        )
        mock_viewer.return_value.boh = pb.BillOfHealth()
        self.mock_app_data.resolve_run_id.side_effect = (
            exceptions.WorkspaceNotFoundException("not found")
        )

        # Not tagged even when the raw run_id looks like a workspace name.
        cli._show_results(self.mock_app_data, MOCK_TELEMETRY_WORKSPACE_NAME)

        self.assertIsNone(telemetry._state.run_id)

    def test_cancel_command_sets_telemetry_run(self):
        """Test 'cancel' tags telemetry with the latest run."""
        self.mock_app_data.resolve_run_id.side_effect = None
        self.mock_app_data.resolve_run_id.return_value = (
            MOCK_TELEMETRY_WORKSPACE_NAME
        )

        result = self.runner.invoke(cli.cli, ["cancel", "--auto-approve"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(telemetry._state.run_id, MOCK_TELEMETRY_RUN_ID)

    def test_cancel_command_skips_telemetry_run_for_unregistered_run(self):
        """Test 'cancel' of a run not in the local registry sets no run."""
        self.mock_app_data.resolve_run_id.side_effect = (
            exceptions.WorkspaceNotFoundException("not found")
        )

        result = self.runner.invoke(
            cli.cli, ["cancel", "other-user-run-ab12", "--auto-approve"]
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(telemetry._state.run_id)

    def test_archive_workspace_context_manager_success(self):
        """Test archive_workspace calling archival on success."""
        mock_rm = mock.Mock()
        mock_rm.path = "/mock/workspace"
        mock_rm.export_logs_dir = None
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
        mock_rm.path = "/mock/workspace"
        mock_rm.export_logs_dir = None
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
        """Test archive_workspace bypassing archival on KeyboardInterrupt."""
        mock_rm = mock.Mock()
        mock_rm.is_workspace_created = True
        with self.assertRaises(KeyboardInterrupt):
            with cli.archive_workspace(mock_rm):
                raise KeyboardInterrupt()
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

    @mock.patch("chs.cli.Console")
    def test_cancel_jobs_on_failure_cleanup_keyboard_interrupt(
        self, mock_console_cls
    ):
        """Test Ctrl+C during cleanup prints warning and propagates."""
        mock_console = mock_console_cls.return_value
        mock_scheduler = mock.Mock()
        mock_scheduler.cancel_run.side_effect = KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            with cli.cancel_jobs_on_failure(mock_scheduler, "run1"):
                raise RuntimeError("original failure")

        mock_console.print.assert_called_with(
            "[bold red]WARNING: Cleanup interrupted. "
            "Please run `chs cancel run1` to finish cleaning up."
            "[/bold red]"
        )

    @mock.patch("chs.cli.Console")
    @mock.patch("chs.cli.logger")
    def test_cancel_jobs_on_failure_cleanup_exception(
        self, mock_logger, mock_console_cls
    ):
        """Test Exception during cleanup logs error and prints warning."""
        mock_console = mock_console_cls.return_value
        mock_scheduler = mock.Mock()
        cleanup_err = RuntimeError("cleanup failed")
        mock_scheduler.cancel_run.side_effect = cleanup_err

        with self.assertRaises(ValueError):
            with cli.cancel_jobs_on_failure(mock_scheduler, "run1"):
                raise ValueError("original failure")

        mock_logger.error.assert_called_with(
            "Failed to clean CHS resources. Please run "
            "`chs cancel %s`. Error: %s",
            "run1",
            cleanup_err,
        )
        mock_console.print.assert_called_once()

    @mock.patch("chs.cli.Console")
    @mock.patch("chs.cli.logger")
    def test_cancel_jobs_on_failure_interrupt_reports_cleanup_outcome(
        self, mock_logger, mock_console_cls
    ):
        """Test Ctrl+C reports a finished cleanup only if no cleanup raised."""
        mock_console = mock_console_cls.return_value
        cases = [
            (None, "CHS cleanup finished."),
            ("cancel_run", "Automatic cleanup failed."),
        ]
        for failing_step, expected_message in cases:
            with self.subTest(failing_step=failing_step):
                mock_console.print.reset_mock()
                mock_scheduler = mock.Mock()
                if failing_step:
                    failing_method = getattr(mock_scheduler, failing_step)
                    failing_method.side_effect = RuntimeError("cleanup failed")

                with self.assertRaises(KeyboardInterrupt):
                    with cli.cancel_jobs_on_failure(mock_scheduler, "run1"):
                        raise KeyboardInterrupt()

                mock_console.print.assert_called_once()
                self.assertIn(
                    expected_message, mock_console.print.call_args.args[0]
                )

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
        mock_scheduler.find_runs_targeting_nodes.return_value = {}
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
        mock_get_scheduler.assert_called_once()
        self.assertEqual(
            mock_get_scheduler.call_args.kwargs["nodepools"], "pool1"
        )

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

    def test_validations_occur_before_preparing_cluster(self):
        """Test validation failures prevent cluster preparation for execution"""
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
        self.mock_scheduler.setup_scheduler_resources.assert_not_called()

    @mock.patch("chs.cli._analyze_action", autospec=True)
    @mock.patch("chs.cli.get_scheduler")
    def test_run_command_execution_preview_shows_full_cluster_name(
        self, mock_get_scheduler, mock_analyze_action
    ):
        """Test that the full cluster name is printed in execution preview."""
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "kubernetes"
        mock_scheduler.get_cluster_name.return_value = (
            "full-cluster-name-123456"
        )
        mock_scheduler.get_nodes_under_test.return_value = ["node1"]
        mock_scheduler.filter_partial_tpu_slices.return_value = ["node1"]
        mock_scheduler.detect_nodes_hardware.return_value = ["a3-megagpu-8g"]
        mock_scheduler.find_runs_targeting_nodes.return_value = {}
        mock_scheduler.run_preflight_checks.return_value = [{"name": "test1"}]
        mock_scheduler.setup_scheduler_resources.return_value = [
            {"name": "test1"}
        ]
        mock_get_scheduler.return_value = mock_scheduler

        result = self.runner.invoke(
            cli.cli, ["run", "--suite", "suite1", "--auto-approve"]
        )

        self.assertIn("full-cluster-name-123456", result.output)
        self.assertIn("Target Cluster", result.output)

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
        mock_results_viewer.return_value.boh = pb.BillOfHealth()
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
        viewer_mock.show_summary_and_details.assert_called_once_with()

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

    @mock.patch.object(cli, "cli", autospec=True)
    def test_main_keyboard_interrupt_exits_130(self, mock_cli):
        """Test main catches KeyboardInterrupt and exits with code 130."""
        mock_cli.side_effect = KeyboardInterrupt()
        with self.assertRaises(SystemExit) as cm:
            cli.main()
        self.assertEqual(cm.exception.code, 130)

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

    def test_configure_telemetry_enable(self):
        self.mock_app_data.telemetry_env_override.return_value = None

        result = self.runner.invoke(
            cli.cli, ["configure", "telemetry", "enable"]
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.mock_app_data.set_telemetry_enabled.assert_called_once_with(True)
        self.assertEqual(result.output, "Telemetry enabled.\n")

    def test_configure_telemetry_enable_says_env_var_keeps_it_off(self):
        self.mock_app_data.telemetry_env_override.return_value = (
            "CHS_TELEMETRY=0"
        )

        result = self.runner.invoke(
            cli.cli, ["configure", "telemetry", "enable"]
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.mock_app_data.set_telemetry_enabled.assert_called_once_with(True)
        self.assertEqual(
            result.output,
            "Telemetry enabled, but it stays off while CHS_TELEMETRY=0 is "
            "set.\n",
        )

    def test_configure_telemetry_disable(self):
        result = self.runner.invoke(
            cli.cli, ["configure", "telemetry", "disable"]
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.mock_app_data.set_telemetry_enabled.assert_called_once_with(False)
        self.assertEqual(result.output, "Telemetry disabled.\n")

    def test_configure_telemetry_status(self):
        for env_override, enabled, expected in (
            (None, True, "Telemetry is enabled."),
            (None, False, "Telemetry is disabled."),
            (
                "DO_NOT_TRACK=1",
                False,
                "Telemetry is disabled because DO_NOT_TRACK=1 is set.",
            ),
        ):
            with self.subTest(env_override=env_override, enabled=enabled):
                self.mock_app_data.telemetry_env_override.return_value = (
                    env_override
                )
                self.mock_app_data.is_telemetry_enabled.return_value = enabled

                result = self.runner.invoke(
                    cli.cli, ["configure", "telemetry", "status"]
                )

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(result.output, f"{expected}\n")


class TestCliHistory(unittest.TestCase):
    """Tests the 'chs history' command."""

    def setUp(self):
        super().setUp()
        self.runner = testing.CliRunner()

        def patched_logger_error(msg, *args, **kwargs):
            click.echo(msg, err=True)

        self.mock_logger_error = mock.patch(
            "chs.exceptions.logger.error", side_effect=patched_logger_error
        )
        self.mock_logger_error.start()
        self.addCleanup(self.mock_logger_error.stop)
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


class TestInterruptOnTerminationSignals(unittest.TestCase):
    """Tests for handling SIGTERM and SIGHUP during a run."""

    def setUp(self):
        super().setUp()
        for signum in (signal.SIGTERM, signal.SIGHUP):
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))
            signal.signal(signum, signal.SIG_DFL)

    @mock.patch("chs.cli.Console")
    def test_sigterm_cleans_up_like_ctrl_c(self, _mock_console_cls):
        """Test SIGTERM runs the cleanup and a second SIGTERM can't stop it."""
        mock_scheduler = mock.Mock()
        mock_scheduler.cancel_run.side_effect = lambda run_id: (
            os.kill(os.getpid(), signal.SIGTERM)
        )

        with self.assertRaises(KeyboardInterrupt):
            with cli._handle_termination_signals(cli._interrupt_run):
                with cli.cancel_jobs_on_failure(mock_scheduler, "run1"):
                    os.kill(os.getpid(), signal.SIGTERM)

        mock_scheduler.cancel_run.assert_called_once_with("run1")

    @mock.patch("chs.cli.Console")
    def test_signals_after_ctrl_c_do_not_stop_cleanup(self, _mock_console_cls):
        """Test SIGTERM and SIGHUP during a Ctrl+C cleanup let it finish."""
        mock_scheduler = mock.Mock()

        def send_termination_signals(run_id):
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGHUP)

        mock_scheduler.cancel_run.side_effect = send_termination_signals

        with (
            mock.patch.object(
                cli, "_discard_output_after_hangup"
            ) as mock_discard_output,
            self.assertRaises(KeyboardInterrupt),
        ):
            with cli._handle_termination_signals(cli._interrupt_run):
                with cli.cancel_jobs_on_failure(mock_scheduler, "run1"):
                    raise KeyboardInterrupt
        mock_discard_output.assert_called_once_with()

    def test_previous_handlers_are_restored(self):
        """Test the default handlers are back after the run."""
        with cli._handle_termination_signals(cli._interrupt_run):
            self.assertIs(signal.getsignal(signal.SIGTERM), cli._interrupt_run)

        self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
        self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_DFL)

    def test_ignored_sighup_stays_ignored(self):
        """Test a run started under nohup still survives a hangup."""
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

        with cli._handle_termination_signals(cli._interrupt_run):
            self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)

        self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)

    def test_hangup_discards_terminal_output(self):
        """Test a hangup sends terminal output to /dev/null and interrupts."""
        with (
            mock.patch.object(cli.os, "open", return_value=99),
            mock.patch.object(
                cli.os, "fstat", return_value=mock.Mock(st_mode=stat.S_IFCHR)
            ),
            mock.patch.object(cli.os, "dup2") as mock_dup2,
            mock.patch.object(cli.os, "close") as mock_close,
            cli._handle_termination_signals(cli._interrupt_run),
        ):
            with self.assertRaises(KeyboardInterrupt):
                signal.getsignal(signal.SIGHUP)(signal.SIGHUP, None)
            for signum in (signal.SIGTERM, signal.SIGHUP):
                self.assertIs(signal.getsignal(signum), cli._continue_cleanup)

        self.assertEqual(
            mock_dup2.call_args_list, [mock.call(99, 1), mock.call(99, 2)]
        )
        mock_close.assert_called_once_with(99)

    def test_hangup_keeps_output_to_files(self):
        """Test a hangup leaves output that goes to a log file alone."""
        modes = {1: stat.S_IFREG, 2: stat.S_IFCHR}
        with (
            mock.patch.object(cli.os, "open", return_value=99),
            mock.patch.object(
                cli.os,
                "fstat",
                side_effect=lambda fd: mock.Mock(st_mode=modes[fd]),
            ),
            mock.patch.object(cli.os, "dup2") as mock_dup2,
            mock.patch.object(cli.os, "close"),
        ):
            cli._discard_output_after_hangup()

        mock_dup2.assert_called_once_with(99, 2)

    def test_hangup_keeps_devnull_opened_as_standard_fd(self):
        """Test /dev/null stays open when it is opened as stdout."""
        with (
            mock.patch.object(cli.os, "open", return_value=1),
            mock.patch.object(
                cli.os, "fstat", return_value=mock.Mock(st_mode=stat.S_IFCHR)
            ),
            mock.patch.object(cli.os, "dup2"),
            mock.patch.object(cli.os, "close") as mock_close,
        ):
            cli._discard_output_after_hangup()

        mock_close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
