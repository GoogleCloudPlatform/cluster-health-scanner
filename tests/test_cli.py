"""Unit tests for the Cluster Health Scanner (CHS) CLI application.

These tests verify the correctness of the CLI's argument parsing, subcommand
routing, and error handling for various valid and invalid inputs.
"""

import dataclasses
import unittest
from unittest import mock

from chs import cli
import click
from click import testing

CliRunner = testing.CliRunner


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
  expected_return: tuple[str, str, list[str] | None]


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

    self.assertEqual(cli.non_empty_str_callback(ctx, param, "valid"), "valid")
    self.assertIsNone(cli.non_empty_str_callback(ctx, param, None))

  @mock.patch.object(click, "echo", autospec=True)
  def test_list_tests_implementation_hint(self, mock_echo):
    """Test _list_tests prints hint when listing all tests."""
    cli._list_tests(from_suite=None, tests_flag=True)
    mock_echo.assert_any_call(
        "  (Hint: Use --from-suite <suite_name> to filter tests)"
    )

  @mock.patch.object(click, "echo", autospec=True)
  def test_list_tests_implementation_no_hint(self, mock_echo):
    """Test _list_tests does not print hint when filtering by suite."""
    cli._list_tests(from_suite="suite1", tests_flag=True)
    hint_message = "  (Hint: Use --from-suite <suite_name> to filter tests)"
    self.assertTrue(
        all(hint_message not in call.args for call in mock_echo.call_args_list)
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

  def test_run_command(self):
    """Test 'run' command."""
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
            ],
            expected_return=("suite1", "slurm", None),
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
            ],
            expected_return=("suite1", "slurm", ["t1", "t2"]),
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
            ],
            expected_return=("suite1", "slurm", ["t1"]),
        ),
        _RunCommandTestCase(
            name="gke_scheduler",
            args_list=[
                "-v",
                "run",
                "--suite",
                "suite1",
                "--scheduler",
                "gke",
            ],
            expected_return=("suite1", "gke", None),
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
            ],
            expected_return=("suite1", "slurm", ["t1"]),
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
            ],
            expected_return=("suite1", "slurm", ["t1"]),
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
            ],
            expected_return=("suite1", "slurm", ["t1", "t2"]),
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
            ],
            expected_return=("suite1", "slurm", ["t1", "t2"]),
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
            ],
            expected_return=("suite1", "slurm", ["t1", "t2", "t3"]),
        ),
    ]
    for case in test_cases:
      with self.subTest(case.name):
        result = self.runner.invoke(
            cli.cli,
            case.args_list,
            standalone_mode=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.return_value, case.expected_return)

  def test_run_command_errors(self):
    """Test 'run' command with invalid flags."""
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
            ],
            expected_output="cannot be an empty string",
        ),
        _CommandErrorTestCase(
            name="missing_suite",
            args_list=["run", "--scheduler", "slurm"],
            expected_output="Missing option '--suite'",
        ),
        _CommandErrorTestCase(
            name="missing_scheduler",
            args_list=["run", "--suite", "suite1"],
            expected_output="Missing option '--scheduler'",
        ),
        _CommandErrorTestCase(
            name="suite_missing_value",
            args_list=["run", "--suite", "--scheduler", "gke"],
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
            ],
            expected_output="Got unexpected extra argument",
        ),
        _CommandErrorTestCase(
            name="tests_missing_value",
            args_list=[
                "run",
                "--scheduler",
                "gke",
                "--tests",
                "--suite",
                "s1",
            ],
            expected_output="Missing option '--suite' / '-s'",
        ),
        _CommandErrorTestCase(
            name="empty_suite",
            args_list=["run", "--suite", "", "--scheduler", "slurm"],
            expected_output="cannot be an empty string",
        ),
        _CommandErrorTestCase(
            name="empty_scheduler",
            args_list=["run", "--suite", "suite1", "--scheduler", ""],
            expected_output="cannot be an empty string",
        ),
    ]
    for case in test_cases:
      with self.subTest(case.name):
        result = self.runner.invoke(cli.cli, case.args_list)
        self.assertEqual(result.exit_code, 2)
        self.assertIn(case.expected_output, result.output)

  def test_analyze_command(self):
    """Test 'analyze' command."""
    result = self.runner.invoke(
        cli.cli,
        ["analyze", "my_run_id"],
        standalone_mode=False,
    )
    self.assertEqual(result.exit_code, 0)
    self.assertEqual(result.return_value, "my_run_id")

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
        _CommandErrorTestCase(
            name="missing_run_id",
            args_list=["analyze"],
            expected_output="Missing argument 'RUN_ID'",
        ),
    ]
    for case in test_cases:
      with self.subTest(case.name):
        result = self.runner.invoke(cli.cli, case.args_list)
        self.assertEqual(result.exit_code, 2)
        self.assertIn(case.expected_output, result.output)

  def test_ensure_object_called(self):
    """Test that ctx.ensure_object(dict) is called."""
    with mock.patch.object(
        click.Context, "ensure_object", autospec=True
    ) as mock_ensure:
      result = self.runner.invoke(cli.cli, ["list"])
      self.assertEqual(result.exit_code, 0)
      mock_ensure.assert_called_once_with(mock.ANY, dict)

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
        self.assertEqual(mock_suites.call_count, case.expected_suites_calls)
        if case.expected_tests_call_args:
          mock_tests.assert_called_once_with(*case.expected_tests_call_args)
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


if __name__ == "__main__":
  unittest.main()
