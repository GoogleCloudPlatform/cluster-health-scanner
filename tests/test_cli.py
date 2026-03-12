"""Unit tests for the Cluster Health Scanner (CHS) CLI application.

These tests verify the correctness of the CLI's argument parsing, subcommand
routing, and error handling for various valid and invalid inputs.
"""

import dataclasses
import datetime
import unittest
from unittest import mock

from chs import cli
from chs import exceptions
import click
from click import testing
import shutil
import tempfile
import os
import pathlib

CliRunner = testing.CliRunner
MOCK_DATETIME = datetime.datetime(2023, 1, 1, 12, 0, 0, 436888)
MOCK_WORKSPACE_NAME = MOCK_DATETIME.strftime("%Y-%m-%d_%H.%M.%S.%f")


@dataclasses.dataclass
class _ListActionKwargs:
    suites: bool = False
    tests: bool = False
    schedulers: bool = False
    from_suite: str | None = None


@dataclasses.dataclass
class _ListCommandTestCase:
    name: str
    args_list: list[str]
    expected_kwargs: _ListActionKwargs


@dataclasses.dataclass
class _CommandErrorTestCase:
    name: str
    args_list: list[str]
    expected_output: str


@dataclasses.dataclass
class _RunCommandTestCase:
    name: str
    args_list: list[str]
    expected_return: tuple[str, str, list[str] | None, str, str, list[str]]


@dataclasses.dataclass
class _ListActionTestCase:
    name: str
    kwargs: _ListActionKwargs
    expected_result: bool
    expected_suites_calls: int
    expected_tests_call_args: tuple[str | None, bool] | None
    expected_schedulers_calls: int


class TestCli(unittest.TestCase):
    """Tests the CLI logic and command invocation."""

    def setUp(self):
        super().setUp()
        self.runner = CliRunner()
        self.mock_rm_cls = self.enterContext(
            mock.patch("chs.cli.RambleManager", autospec=True)
        )
        self.mock_rm_instance = self.mock_rm_cls.return_value
        self.mock_rm_instance.execute_experiments.return_value = []
        self.mock_rm_instance.check_jobs_running.return_value = False

        self.mock_dt = self.enterContext(
            mock.patch("chs.cli.datetime.datetime", autospec=True)
        )
        self.mock_dt.now.return_value = MOCK_DATETIME

        self.mock_get_scheduler = self.enterContext(
            mock.patch("chs.cli.get_scheduler", autospec=True)
        )
        self.mock_scheduler = self.mock_get_scheduler.return_value
        self.mock_scheduler.name = "mock_scheduler"
        self.mock_scheduler.get_nodes_under_test.return_value = ["node1"]

        self.mock_load_suite_tests = self.enterContext(
            mock.patch("chs.cli.load_suite_tests", autospec=True)
        )
        self.mock_load_suite_tests.return_value = [
            "nccl-cluster",
            "nccl-pairwise",
            "dcgm",
        ]
        self.mock_app_data_cls = self.enterContext(
            mock.patch("chs.cli.ChsAppData", autospec=True)
        )
        self.mock_app_data = self.mock_app_data_cls.return_value
        self.mock_app_data.get_workspaces_dir.return_value = (
            "/mocked/workspaces"
        )
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"

    def test_non_empty_str_callback(self):
        """Test non_empty_str_callback raises BadParameter on empty string."""
        ctx = mock.Mock(spec=click.Context)
        ctx.command = mock.Mock()
        param = mock.Mock(spec=click.Parameter)
        with self.assertRaisesRegex(
            click.BadParameter, "cannot be an empty string"
        ):
            cli.non_empty_str_callback(ctx, param, "")

        with self.assertRaisesRegex(
            click.BadParameter, "cannot be an empty string"
        ):
            cli.non_empty_str_callback(ctx, param, " ")

        self.assertEqual(
            cli.non_empty_str_callback(ctx, param, "  valid  "), "valid"
        )

        self.assertEqual(
            cli.non_empty_str_callback(ctx, param, "valid"), "valid"
        )
        self.assertIsNone(cli.non_empty_str_callback(ctx, param, None))

    @mock.patch("chs.cli.logger")
    def test_list_tests_implementation_hint(self, mock_logger):
        """Test _list_tests prints hint when listing all tests."""
        cli._list_tests(from_suite=None, tests_flag=True)
        mock_logger.info.assert_any_call(
            "  (Hint: Use --from-suite <suite_name> to filter tests)"
        )

    @mock.patch("chs.cli.logger")
    def test_list_tests_implementation_no_hint(self, mock_logger):
        """Test _list_tests does not print hint when filtering by suite."""
        cli._list_tests(from_suite="suite1", tests_flag=True)
        hint_message = "  (Hint: Use --from-suite <suite_name> to filter tests)"
        self.assertTrue(
            all(
                hint_message not in call.args
                for call in mock_logger.info.call_args_list
            )
        )

    @mock.patch.object(cli, "list_action", autospec=True)
    def test_list_command(self, mock_list_action):
        """Test 'list' command with various flags."""
        test_cases = [
            _ListCommandTestCase(
                name="suites",
                args_list=["list", "--suites"],
                expected_kwargs=_ListActionKwargs(suites=True),
            ),
            _ListCommandTestCase(
                name="tests",
                args_list=["list", "--tests"],
                expected_kwargs=_ListActionKwargs(tests=True),
            ),
            _ListCommandTestCase(
                name="schedulers",
                args_list=["list", "--schedulers"],
                expected_kwargs=_ListActionKwargs(schedulers=True),
            ),
            _ListCommandTestCase(
                name="all",
                args_list=["list"],
                expected_kwargs=_ListActionKwargs(),
            ),
            _ListCommandTestCase(
                name="from_suite",
                args_list=["list", "--tests", "--from-suite", "suite1"],
                expected_kwargs=_ListActionKwargs(
                    tests=True,
                    from_suite="suite1",
                ),
            ),
            _ListCommandTestCase(
                name="suites_and_tests",
                args_list=["list", "--suites", "--tests"],
                expected_kwargs=_ListActionKwargs(
                    suites=True,
                    tests=True,
                ),
            ),
            _ListCommandTestCase(
                name="suites_and_schedulers",
                args_list=["list", "--suites", "--schedulers"],
                expected_kwargs=_ListActionKwargs(
                    suites=True,
                    schedulers=True,
                ),
            ),
            _ListCommandTestCase(
                name="tests_and_schedulers",
                args_list=["list", "--tests", "--schedulers"],
                expected_kwargs=_ListActionKwargs(
                    tests=True,
                    schedulers=True,
                ),
            ),
            _ListCommandTestCase(
                name="all_boolean_flags",
                args_list=["list", "--suites", "--tests", "--schedulers"],
                expected_kwargs=_ListActionKwargs(
                    suites=True,
                    tests=True,
                    schedulers=True,
                ),
            ),
            _ListCommandTestCase(
                name="from_suite_trailing_comma",
                args_list=["list", "--tests", "--from-suite", "suite1,"],
                expected_kwargs=_ListActionKwargs(
                    tests=True,
                    from_suite="suite1,",
                ),
            ),
            _ListCommandTestCase(
                name="from_suite_leading_comma",
                args_list=["list", "--tests", "--from-suite", ",suite1"],
                expected_kwargs=_ListActionKwargs(
                    tests=True,
                    from_suite=",suite1",
                ),
            ),
            _ListCommandTestCase(
                name="from_suite_double_comma",
                args_list=["list", "--tests", "--from-suite", "suite1,,suite2"],
                expected_kwargs=_ListActionKwargs(
                    tests=True,
                    from_suite="suite1,,suite2",
                ),
            ),
            _ListCommandTestCase(
                name="from_suite_whitespace_and_comma",
                args_list=[
                    "list",
                    "--tests",
                    "--from-suite",
                    "suite1, suite2 ,suite3 ",
                ],
                expected_kwargs=_ListActionKwargs(
                    tests=True,
                    from_suite="suite1, suite2 ,suite3",
                ),
            ),
        ]
        for case in test_cases:
            with self.subTest(case.name):
                mock_list_action.reset_mock()
                result = self.runner.invoke(cli.cli, case.args_list)
                self.assertEqual(result.exit_code, 0)
                mock_list_action.assert_called_once_with(
                    **dataclasses.asdict(case.expected_kwargs)
                )

    def test_list_command_errors(self):
        """Test 'list' command with invalid flags."""
        test_cases = [
            _CommandErrorTestCase(
                name="from_suite_empty",
                args_list=["list", "--tests", "--from-suite", ""],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="from_suite_whitespace",
                args_list=["list", "--tests", "--from-suite", " "],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="from_suite_without_tests",
                args_list=["list", "--from-suite", "suite1"],
                expected_output=(
                    "Error: Option '--from-suite' requires option '--tests'."
                ),
            ),
            _CommandErrorTestCase(
                name="from_suite_with_suites",
                args_list=["list", "--suites", "--from-suite", "suite1"],
                expected_output=(
                    "Error: Option '--from-suite' requires option '--tests'."
                ),
            ),
        ]
        for case in test_cases:
            with self.subTest(case.name):
                result = self.runner.invoke(cli.cli, case.args_list)
                self.assertEqual(result.exit_code, 2)
                self.assertIn(case.expected_output, result.output)

    @mock.patch("chs.cli.analyze_action", autospec=True)
    @mock.patch("chs.cli.get_scheduler")
    def test_run_command(self, mock_get_scheduler, mock_analyze_action):
        """Test 'run' command."""
        mock_default_dir = "/mocked/workspaces"
        self.mock_app_data.get_workspaces_dir.return_value = mock_default_dir

        # Setup mock scheduler
        mock_scheduler = mock.Mock()
        mock_scheduler.get_nodes_under_test.return_value = ["node1"]

        def get_scheduler_side_effect(name, **kwargs):
            s = mock.Mock()
            s.name = name
            s.get_nodes_under_test.return_value = ["node1"]
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
                    "--scheduler",
                    "slurm",
                    "--machine-type",
                    "a3-high",
                ],
                expected_return=(
                    "suite1",
                    "slurm",
                    ["nccl-cluster", "nccl-pairwise", "dcgm"],
                    MOCK_WORKSPACE_NAME,
                    mock_default_dir,
                    ["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="with_tests",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "slurm",
                    "--tests",
                    "t1,t2",
                    "--machine-type",
                    "a3-high",
                ],
                expected_return=(
                    "suite1",
                    "mock_scheduler",
                    ["t1", "t2"],
                    MOCK_WORKSPACE_NAME,
                    mock_default_dir,
                    ["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="with_single_test",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "slurm",
                    "--tests",
                    "t1",
                    "--machine-type",
                    "a3-high",
                ],
                expected_return=(
                    "suite1",
                    "mock_scheduler",
                    ["t1"],
                    MOCK_WORKSPACE_NAME,
                    mock_default_dir,
                    ["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="trailing_comma",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "slurm",
                    "--tests",
                    "t1,",
                    "--machine-type",
                    "a3-high",
                ],
                expected_return=(
                    "suite1",
                    "mock_scheduler",
                    ["t1"],
                    MOCK_WORKSPACE_NAME,
                    mock_default_dir,
                    ["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="leading_comma",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "slurm",
                    "--tests",
                    ",t1",
                    "--machine-type",
                    "a3-high",
                ],
                expected_return=(
                    "suite1",
                    "mock_scheduler",
                    ["t1"],
                    MOCK_WORKSPACE_NAME,
                    mock_default_dir,
                    ["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="double_comma",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "slurm",
                    "--tests",
                    "t1,,t2",
                    "--machine-type",
                    "a3-high",
                ],
                expected_return=(
                    "suite1",
                    "mock_scheduler",
                    ["t1", "t2"],
                    MOCK_WORKSPACE_NAME,
                    mock_default_dir,
                    ["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="whitespace",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "slurm",
                    "--tests",
                    "t1, t2 ",
                    "--machine-type",
                    "a3-high",
                ],
                expected_return=(
                    "suite1",
                    "mock_scheduler",
                    ["t1", "t2"],
                    MOCK_WORKSPACE_NAME,
                    mock_default_dir,
                    ["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="whitespace_and_comma",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "slurm",
                    "--tests",
                    "t1, t2 ,t3 ",
                    "--machine-type",
                    "a3-high",
                ],
                expected_return=(
                    "suite1",
                    "mock_scheduler",
                    ["t1", "t2", "t3"],
                    MOCK_WORKSPACE_NAME,
                    mock_default_dir,
                    ["job1"],
                ),
            ),
            _RunCommandTestCase(
                name="with_nodes",
                args_list=[
                    "-v",
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "slurm",
                    "--nodes",
                    "node1,node2",
                    "--machine-type",
                    "a3-high",
                ],
                expected_return=(
                    "suite1",
                    "mock_scheduler",
                    ["nccl-cluster", "nccl-pairwise", "dcgm"],
                    MOCK_WORKSPACE_NAME,
                    mock_default_dir,
                    ["job1"],
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

                expected_return_list = list(case.expected_return)
                expected_return_list[1] = "slurm"
                case.expected_return = tuple(expected_return_list)

                result = self.runner.invoke(
                    cli.cli,
                    case.args_list,
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
                    mock_default_dir, MOCK_WORKSPACE_NAME
                )
                self.mock_rm_instance.create_workspace.assert_called_once()
                self.mock_rm_instance.configure_workspace.assert_called_once()
                args, kwargs = (
                    self.mock_rm_instance.configure_workspace.call_args
                )
                self.assertEqual(args[0].name, case.expected_return[1])
                self.assertEqual(args[1], case.expected_return[2])
                self.assertEqual(kwargs["machine_type"], "a3-high")
                self.mock_rm_instance.setup_workspace.assert_called_once()
                self.mock_rm_instance.execute_experiments.assert_called_once()
                mock_analyze_action.assert_called_once_with(
                    self.mock_app_data, MOCK_WORKSPACE_NAME
                )

    def test_analyze_action_exception_handling(self):
        """Test that analyze_action wraps exceptions in ClickException."""
        self.mock_rm_instance.wait_for_jobs.side_effect = (
            exceptions.ChsAnalysisException("Analysis failed")
        )
        result = self.runner.invoke(cli.cli, ["analyze", "some_run_id"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Analysis failed", result.output)

    def test_cancel_action_exception_handling(self):
        """Test that cancel_action wraps exceptions in ClickException."""
        self.mock_rm_instance.check_jobs_running.return_value = True
        self.mock_rm_instance.cancel_experiments.side_effect = (
            exceptions.ChsCancellationException("Cancellation failed")
        )
        result = self.runner.invoke(cli.cli, ["cancel", "some_run_id"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Cancellation failed", result.output)

    def test_results_command_no_boh(self):
        """Test `results` command when bill_of_health.json is not found."""
        self.mock_app_data.get_latest_run_id.return_value = "latest_run"
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_app_data.get_workspaces_dir.return_value = tmpdir
            os.makedirs(os.path.join(tmpdir, "latest_run", "chs-results"))
            result = self.runner.invoke(cli.cli, ["results"])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Could not find bill of health", result.output)

    def test_run_action_exception_handling(self):
        """Test that run_action wraps exceptions in ClickException."""
        self.mock_app_data.get_workspaces_dir.return_value = (
            "/mocked/workspaces"
        )

        self.mock_scheduler.name = "slurm"

        self.mock_rm_instance.create_workspace.side_effect = OSError(
            "Permission denied"
        )
        result = self.runner.invoke(
            cli.cli,
            ["run", "-s", "s1", "-sch", "gke", "--machine-type", "a3-high"],
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("CHS execution failed", result.output)
        self.assertIn("Permission denied", result.output)

    @mock.patch("chs.cli.get_scheduler")
    @mock.patch("chs.cli.analyze_action", autospec=True)
    def test_run_command_passes_nodepool_to_scheduler(
        self, mock_analyze_action, mock_get_scheduler
    ):
        """Test that nodepool argument is passed to get_scheduler."""
        self.mock_app_data.get_workspaces_dir.return_value = (
            "/mocked/workspaces"
        )
        mock_scheduler = mock.Mock()
        mock_scheduler.name = "gke"
        mock_scheduler.get_nodes_under_test.return_value = ["node1"]
        mock_get_scheduler.return_value = mock_scheduler
        self.mock_rm_instance.execute_experiments.return_value = ["job1"]
        result = self.runner.invoke(
            cli.cli,
            [
                "run",
                "--scheduler",
                "gke",
                "--nodepool",
                "pool1",
                "--machine-type",
                "a3-high",
            ],
        )

        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed: {result.output}\nException: {result.exception}",
        )
        mock_get_scheduler.assert_called_once()
        _, kwargs = mock_get_scheduler.call_args
        self.assertEqual(kwargs.get("nodepool"), "pool1")

    def test_run_command_latest_run_active(self):
        """Test `run` command when the latest run is active."""
        self.mock_app_data.get_latest_run_id.return_value = (
            "2023-01-01_00.00.00.000000"
        )
        self.mock_rm_instance.check_jobs_running.return_value = True
        result = self.runner.invoke(
            cli.cli, ["run", "-sch", "slurm", "--machine-type", "a3-high"]
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("The latest CHS run is still in progress", result.output)

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
                    "--scheduler",
                    "slurm",
                    "--tests",
                    "",
                    "--machine-type",
                    "a3-high",
                ],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="unknown_scheduler",
                args_list=[
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "unknown",
                    "--machine-type",
                    "a3-high",
                ],
                expected_output="Only Slurm and GKE schedulers are supported",
            ),
            _CommandErrorTestCase(
                name="missing_scheduler",
                args_list=[
                    "run",
                    "--suite",
                    "suite1",
                    "--machine-type",
                    "a3-high",
                ],
                expected_output="Missing option '--scheduler'",
            ),
            _CommandErrorTestCase(
                name="missing_machine_type",
                args_list=["run", "--suite", "suite1", "--scheduler", "slurm"],
                expected_output="Missing option '--machine-type'",
            ),
            _CommandErrorTestCase(
                name="suite_missing_value",
                args_list=[
                    "run",
                    "--suite",
                    "--scheduler",
                    "gke",
                    "--machine-type",
                    "a3-high",
                ],
                expected_output="Missing option '--scheduler' / '-sch'",
            ),
            _CommandErrorTestCase(
                name="scheduler_missing_value",
                args_list=[
                    "run",
                    "--suite",
                    "s1",
                    "--scheduler",
                    "--tests",
                    "t1",
                    "--machine-type",
                    "a3-high",
                ],
                expected_output="Got unexpected extra argument",
            ),
            _CommandErrorTestCase(
                name="tests_missing_value",
                args_list=[
                    "run",
                    "--tests",
                    "--scheduler",  # Values for tests consumes scheduler flag
                    "slurm",
                    "--suite",
                    "s1",
                    "--machine-type",
                    "a3-high",
                ],
                expected_output="Missing option '--scheduler'",
            ),
            _CommandErrorTestCase(
                name="empty_suite",
                args_list=[
                    "run",
                    "--suite",
                    "",
                    "--scheduler",
                    "slurm",
                    "--machine-type",
                    "a3-high",
                ],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="empty_scheduler",
                args_list=[
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "",
                    "--machine-type",
                    "a3-high",
                ],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="empty_machine_type",
                args_list=[
                    "run",
                    "--suite",
                    "suite1",
                    "--scheduler",
                    "slurm",
                    "--machine-type",
                    "",
                ],
                expected_output="cannot be an empty string",
            ),
        ]
        for case in test_cases:
            with self.subTest(case.name):
                if "slurm" in case.args_list:
                    self.mock_scheduler.name = "slurm"
                elif "gke" in case.args_list:
                    self.mock_scheduler.name = "gke"
                else:
                    self.mock_scheduler.name = "mock_scheduler"

                if case.name == "no_nodes_provided":
                    self.mock_scheduler.get_nodes_under_test.return_value = []
                else:
                    self.mock_scheduler.get_nodes_under_test.return_value = [
                        "node1"
                    ]

                result = self.runner.invoke(cli.cli, case.args_list)
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn(case.expected_output, result.output)

    @mock.patch("chs.cli.ResultsViewer", autospec=True)
    def test_analyze_command_shortcut_boh(self, mock_results_viewer):
        """Test 'analyze' command shortcut when BoH already exists."""
        # Create temp workspace
        workspaces_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workspaces_dir)
        self.mock_app_data.get_workspaces_dir.return_value = workspaces_dir

        # Create BOH file
        run_dir = os.path.join(workspaces_dir, "my_run_id")
        os.makedirs(run_dir)
        boh_path = os.path.join(run_dir, "bill_of_health.json")
        with open(boh_path, "w") as f:
            f.write("{}")

        # Configure RambleManager mock instance
        self.mock_rm_instance.path = os.path.join(workspaces_dir, "my_run_id")

        result = self.runner.invoke(
            cli.cli,
            ["analyze", "my_run_id"],
            standalone_mode=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.mock_rm_instance.wait_for_jobs.assert_not_called()
        self.mock_rm_instance.generate_boh.assert_not_called()
        mock_results_viewer.assert_called_once_with(pathlib.Path(boh_path))

    @mock.patch("chs.cli.ResultsViewer", autospec=True)
    def test_analyze_command_shortcut_results(self, mock_results_viewer):
        """Test 'analyze' command shortcut when results.latest.json exists."""
        # Create temp workspace
        workspaces_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workspaces_dir)
        self.mock_app_data.get_workspaces_dir.return_value = workspaces_dir

        # Create results.latest.json file
        workspace_dir = os.path.join(workspaces_dir, "my_run_id")
        os.makedirs(workspace_dir)
        results_path = os.path.join(workspace_dir, "results.latest.json")
        with open(results_path, "w") as f:
            f.write("{}")

        # Configure RambleManager mock instance
        self.mock_rm_instance.path = workspace_dir

        def mock_generate_boh(results_path):
            boh_file = os.path.join(workspace_dir, "bill_of_health.json")
            with open(boh_file, "w") as f:
                f.write("{}")
            return workspace_dir

        self.mock_rm_instance.generate_boh.side_effect = mock_generate_boh

        result = self.runner.invoke(
            cli.cli,
            ["analyze", "my_run_id"],
            standalone_mode=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.mock_rm_instance.wait_for_jobs.assert_not_called()
        self.mock_rm_instance.generate_boh.assert_called_once_with(results_path)
        mock_results_viewer.assert_called_once()

    @mock.patch("chs.cli.ResultsViewer", autospec=True)
    def test_analyze_command_full(self, mock_results_viewer):
        """Test 'analyze' command full path when no results exist yet."""
        # Create temp workspace
        workspaces_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workspaces_dir)
        self.mock_app_data.get_workspaces_dir.return_value = workspaces_dir

        # Create empty workspace dir
        workspace_dir = os.path.join(workspaces_dir, "my_run_id")
        os.makedirs(workspace_dir)

        # Configure RambleManager mock instance
        self.mock_rm_instance.path = workspace_dir

        def mock_analyze_results():
            boh_file = os.path.join(workspace_dir, "bill_of_health.json")
            with open(boh_file, "w") as f:
                f.write("{}")
            return workspace_dir

        self.mock_rm_instance.analyze_results.side_effect = mock_analyze_results

        result = self.runner.invoke(
            cli.cli,
            ["analyze", "my_run_id"],
            standalone_mode=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.mock_rm_instance.wait_for_jobs.assert_called_once()
        self.mock_rm_instance.analyze_results.assert_called_once()
        mock_results_viewer.assert_called_once()

    def test_analyze_command_errors(self):
        """Test 'analyze' command with invalid flags."""
        test_cases = [
            _CommandErrorTestCase(
                name="empty_run_id",
                args_list=["analyze", ""],
                expected_output="cannot be an empty string",
            ),
            _CommandErrorTestCase(
                name="whitespace_run_id",
                args_list=["analyze", " "],
                expected_output="cannot be an empty string",
            ),
        ]
        for case in test_cases:
            with self.subTest(case.name):
                result = self.runner.invoke(cli.cli, case.args_list)
                self.assertEqual(result.exit_code, 2)
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
        mock_results_viewer.return_value.show_summary.assert_called_once()

        # Test with tests flag
        mock_results_viewer.reset_mock()
        result = self.runner.invoke(
            cli.cli,
            ["results", "my_run_id", "--tests", "t1,t2"],
        )
        self.assertEqual(result.exit_code, 0)
        mock_results_viewer.return_value.show_details.assert_called_once_with(
            tests=["t1", "t2"], resources=None
        )

        # Test with resources flag
        mock_results_viewer.reset_mock()
        result = self.runner.invoke(
            cli.cli,
            ["results", "my_run_id", "--resources", "r1"],
        )
        self.assertEqual(result.exit_code, 0)
        mock_results_viewer.return_value.show_details.assert_called_once_with(
            tests=None, resources=["r1"]
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

    @mock.patch.object(cli, "_list_schedulers", autospec=True)
    @mock.patch.object(cli, "_list_tests", autospec=True)
    @mock.patch.object(cli, "_list_suites", autospec=True)
    def test_list_action(
        self,
        mock_suites,
        mock_tests,
        mock_schedulers,
    ):
        """Test list_action with various flags."""
        test_cases = [
            _ListActionTestCase(
                name="all",
                kwargs=_ListActionKwargs(),
                expected_result=True,
                expected_suites_calls=1,
                expected_tests_call_args=(None, False),
                expected_schedulers_calls=1,
            ),
            _ListActionTestCase(
                name="suites_only",
                kwargs=_ListActionKwargs(suites=True),
                expected_result=False,
                expected_suites_calls=1,
                expected_tests_call_args=None,
                expected_schedulers_calls=0,
            ),
            _ListActionTestCase(
                name="tests_only",
                kwargs=_ListActionKwargs(
                    tests=True,
                    from_suite="s1",
                ),
                expected_result=False,
                expected_suites_calls=0,
                expected_tests_call_args=("s1", True),
                expected_schedulers_calls=0,
            ),
            _ListActionTestCase(
                name="schedulers_only",
                kwargs=_ListActionKwargs(schedulers=True),
                expected_result=False,
                expected_suites_calls=0,
                expected_tests_call_args=None,
                expected_schedulers_calls=1,
            ),
            _ListActionTestCase(
                name="suites_and_tests",
                kwargs=_ListActionKwargs(
                    suites=True,
                    tests=True,
                ),
                expected_result=False,
                expected_suites_calls=1,
                expected_tests_call_args=(None, True),
                expected_schedulers_calls=0,
            ),
            _ListActionTestCase(
                name="tests_and_schedulers",
                kwargs=_ListActionKwargs(
                    tests=True,
                    schedulers=True,
                ),
                expected_result=False,
                expected_suites_calls=0,
                expected_tests_call_args=(None, True),
                expected_schedulers_calls=1,
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
                expected_tests_call_args=(None, True),
                expected_schedulers_calls=1,
            ),
        ]
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
        mock_cli.assert_called_once_with(obj={})

    def test_results_command_no_previous_runs(self):
        """Test results command when no previous runs exist."""
        self.mock_app_data.get_latest_run_id.side_effect = (
            exceptions.NoPreviousRunsFoundException("No runs")
        )
        result = self.runner.invoke(cli.cli, ["results"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("No runs", result.output)

    def test_configure_bucket_command(self):
        """Test 'configure bucket' command."""
        result = self.runner.invoke(
            cli.cli, ["configure", "bucket", "--name", "my-bucket"]
        )
        self.assertEqual(result.exit_code, 0)
        self.mock_app_data.set_bucket.assert_called_once_with("my-bucket")
        self.assertIn("GCS bucket set to: my-bucket", result.output)

        # Test default bucket name
        self.mock_app_data.set_bucket.reset_mock()
        result = self.runner.invoke(cli.cli, ["configure", "bucket"])
        self.assertEqual(result.exit_code, 0)
        self.mock_app_data.set_bucket.assert_called_once_with("chs-results")
        self.assertIn("GCS bucket set to: chs-results", result.output)

    @mock.patch("chs.cli.RambleManager")
    def test_analyze_command_workspace_not_found(self, mock_rm_cls):
        """Test analyze command when workspace is not found."""
        mock_rm_cls.return_value.path = "/non/existent/path"
        result = self.runner.invoke(cli.cli, ["analyze", "missing_run"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Workspace 'missing_run' not found", result.output)


if __name__ == "__main__":
    unittest.main()
