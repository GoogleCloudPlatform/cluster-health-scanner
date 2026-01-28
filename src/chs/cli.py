"""Command-line interface implementation for the Cluster Health Scanner (CHS).

This module defines the CLI structure using click and provides the
logic for handling subcommands such as listing resources.
"""

import logging
import click

logger = logging.getLogger(__name__)


def non_empty_str_callback(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
  """Click callback to validate that if a string value is provided, it is not empty."""
  if value is None:
    return None
  stripped_value = value.strip()
  if not stripped_value:
    raise click.BadParameter("cannot be an empty string.", ctx=ctx, param=param)
  return stripped_value


def _list_suites() -> None:
  """Lists available suites."""
  # TODO: Replace with actual suite listing logic
  logger.info("Listed suites.")


def _list_tests(from_suite: str | None, tests_flag: bool) -> None:
  """Lists available tests, potentially filtered by suite."""
  if from_suite:
    # TODO: Add logic to list tests for a specific suite
    logger.info("Listed tests for suite: %r", from_suite)
  else:
    # TODO: Add logic to list all tests
    if tests_flag:
      click.echo("  (Hint: Use --from-suite <suite_name> to filter tests)")
    logger.info("Listed tests.")


def _list_schedulers() -> None:
  """Lists available schedulers."""
  # TODO: Replace with actual scheduler listing logic
  logger.info("Listed schedulers.")


def list_action(
    *, suites: bool, tests: bool, schedulers: bool, from_suite: str | None
) -> bool:
  """Orchestrates listing of available CHS resources.

  Based on the boolean flags, it calls the appropriate listing functions.
  If no flags are set to True, it defaults to listing all resource types.

  Args:
    suites: If True, includes test suites in the list.
    tests: If True, includes tests in the list.
    schedulers: If True, includes schedulers in the list.
    from_suite: If provided, filters tests listed to those belonging to the
      specified suite.

  Returns:
    True if all resource types were listed, False otherwise.
  """
  logger.debug(
      "list_action called with: suites=%r, tests=%r, schedulers=%r,"
      " from_suite=%r",
      suites,
      tests,
      schedulers,
      from_suite,
  )
  show_all = not (suites or tests or schedulers)
  if suites or show_all:
    _list_suites()
  if tests or show_all:
    _list_tests(from_suite, tests)
  if schedulers or show_all:
    _list_schedulers()
  return show_all


def _parse_tests(tests_str: str | None) -> list[str] | None:
  """Parses a comma-separated string of test names.

  Args:
    tests_str: A comma-separated string of test names, or None.

  Returns:
    A list of cleaned test names, or None if the input was None.
  """
  if tests_str is None:
    return None
  return [test.strip() for test in tests_str.split(",") if test.strip()]


def run_action(
    *, suite: str, scheduler: str, tests: str | None
) -> tuple[str, str, list[str] | None]:
  """Validates parameters and orchestrates test suite execution.

  Takes parameters provided via the 'run' command and parses the 'tests'
  argument from a comma-separated string into a list of test names.
  Note: Downstream test execution logic is not yet implemented.

  Args:
    suite: Name of the test suite to run.
    scheduler: Scheduler to use (e.g., 'slurm', 'gke').
    tests: An optional comma-separated string specifying a subset of tests to
      run.

  Returns:
    A tuple containing the suite name, scheduler name, and a list of
    parsed test names if `tests` was provided, otherwise None.
  """
  logger.debug(
      "run_action: Attempting to run suite=%r, scheduler=%r, tests=%r",
      suite,
      scheduler,
      tests,
  )
  parsed_tests = _parse_tests(tests)
  logger.debug("run_action: Parsed tests: %r", parsed_tests)
  # TODO: Add code to run tests here.
  # The `parsed_tests` list should be used when executing the tests.
  return suite, scheduler, parsed_tests


def analyze_action(run_id: str) -> str:
  """Analyzes results from a specific test run.

  Args:
    run_id: The ID of the test run to analyze.

  Returns:
    The run_id of the analyzed test run.
  """
  logger.debug("Analyzing run_id=%r", run_id)
  # TODO: Add code to analyze test results here.
  return run_id


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
  """Cluster Health Scanner (CHS) Tool."""
  ctx.ensure_object(dict)
  if verbose:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.debug("Debug mode enabled.")


@cli.command(name="list")
@click.option("--suites", is_flag=True, help="List available test suites")
@click.option("--tests", is_flag=True, help="List available tests")
@click.option("--schedulers", is_flag=True, help="List available schedulers")
@click.option(
    "--from-suite",
    metavar="SUITE_NAME",
    help="Specify the suite to list tests from (only used with --tests)",
    callback=non_empty_str_callback,
)
def list_command(
    *, suites: bool, tests: bool, schedulers: bool, from_suite: str | None
) -> bool:
  """Lists available resources.

  Args:
    suites: Whether to list available test suites.
    tests: Whether to list available tests.
    schedulers: Whether to list available schedulers.
    from_suite: Optional suite name to filter tests.
  """
  if from_suite and not tests:
    raise click.UsageError("Option '--from-suite' requires option '--tests'.")
  return list_action(
      suites=suites, tests=tests, schedulers=schedulers, from_suite=from_suite
  )


@cli.command()
@click.option(
    "--suite",
    "-s",
    required=True,
    help="Name of the test suite to run",
    callback=non_empty_str_callback,
)
@click.option(
    "--scheduler",
    "-sch",
    required=True,
    help="Scheduler to use (e.g., slurm, gke)",
    callback=non_empty_str_callback,
)
@click.option(
    "--tests",
    "-t",
    help="Comma-separated list of specific tests to run (optional)",
    callback=non_empty_str_callback,
)
def run(
    *, suite: str, scheduler: str, tests: str | None
) -> tuple[str, str, list[str] | None]:
  """Command to run a test suite.

  Args:
    suite: Name of the test suite to run.
    scheduler: Scheduler to use (e.g., slurm, gke).
    tests: Comma-separated list of specific tests to run (optional).
  """
  return run_action(suite=suite, scheduler=scheduler, tests=tests)


@cli.command()
@click.argument("run_id", callback=non_empty_str_callback)
def analyze(run_id: str) -> str:
  """Analyze results from a test run.

  Args:
    run_id: The ID of the test run to analyze.

  Returns:
    The run_id of the analyzed test run.
  """
  return analyze_action(run_id)


def main() -> None:
  """Main entry point for the CHS CLI."""
  logger.info("CHS CLI starting...")
  cli(obj={})
  logger.info("CHS CLI finished.")
