"""Command-line interface implementation for the Cluster Health Scanner (CHS).

This module defines the CLI structure using click and provides the
logic for handling subcommands such as listing resources.
"""

import datetime
import logging
import os
import pathlib
import subprocess

from chs.ramble_manager import RambleManager
from chs.schedulers import get_scheduler
from chs.schedulers.base import BaseScheduler
import click
import platformdirs

logger = logging.getLogger(__name__)


def get_default_workspaces_dir() -> str:
  """Returns the default workspaces directory path, creating it if necessary.

  Uses platformdirs to determine the appropriate user-specific data directory,
  and creates a 'workspaces' subdirectory within it.

  Example paths based on OS:
    Linux: ~/.local/share/cluster-health-scanner/workspaces
    macOS: ~/Library/Application Support/cluster-health-scanner/workspaces

  Returns:
    The path to the workspaces directory as a string.

  Raises:
    OSError: If the directory cannot be created.
  """
  try:
    # TODO: simrankaurb - Edit this once we finalise the location of the
    # workspaces directory. Current options are:
    # 1. tmp directory using tempfile.
    # 2. User's data directory (using platformdirs). - Currently implemented.
    # 3. A custom directory specified by the user using a flag.
    # 4. A subdirectory in the cluster-health-scanner directory.
    app_data_dir = pathlib.Path(
        platformdirs.user_data_dir(appname="cluster-health-scanner")
    )
    workspaces_dir = app_data_dir / "workspaces"
    workspaces_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Using workspaces directory: %s", workspaces_dir)
    return str(workspaces_dir)
  except OSError:
    logger.exception(
        "Failed to create workspaces directory at %s", workspaces_dir
    )
    raise


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
    *,
    suite: str,
    scheduler_obj: BaseScheduler,
    tests: list[str] | None,
    workspace_name: str,
    workspaces_base_dir: str,
    ramble_manager: type[RambleManager],
    machine_type: str,
) -> tuple[str, str, list[str] | None, str, str, list[str]]:
  """Validates parameters and orchestrates test suite execution.

  Args:
    suite: Name of the test suite to run.
    scheduler_obj: Scheduler object to use.
    tests: An optional list of names of specific tests to run.
    workspace_name: The name of the test workspace.
    workspaces_base_dir: The base directory for workspaces.
    ramble_manager: The class to use for RambleManager.
    machine_type: The machine type of the cluster nodes (e.g., a3-high).

  Returns:
    A tuple containing the suite name, scheduler name, a list of
    test names, workspace name, and workspaces base directory, and submitted job
    IDs.


  Raises:
    click.ClickException: If an error occurs during CHS execution.
  """

  logger.debug(
      "run_action: Attempting to run suite=%r, scheduler=%r, tests=%r",
      suite,
      scheduler_obj.name,
      tests,
  )

  ramble_manager = ramble_manager(workspaces_base_dir, workspace_name)

  try:
    ramble_manager.create_workspace()
    ramble_manager.add_custom_repository()
    ramble_manager.configure_workspace(
        suite, scheduler_obj, tests, machine_type=machine_type
    )
    ramble_manager.setup_workspace()
    job_ids = ramble_manager.execute_experiments()
  except click.ClickException:
    # Re-raise ClickExceptions, which are already user-friendly.
    raise
  except (OSError, subprocess.CalledProcessError) as e:
    logger.exception(
        "CHS execution failed for suite=%r, scheduler=%r, tests=%r",
        suite,
        scheduler_obj.name,
        tests,
    )
    raise click.ClickException(f"CHS execution failed: {e!r}") from e
  else:
    return (
        suite,
        scheduler_obj.name,
        tests,
        workspace_name,
        workspaces_base_dir,
        job_ids,
    )


def analyze_action(run_id: str) -> str:
  """Analyzes results from a specific test run.

  Args:
    run_id: The ID of the test run to analyze.

  Returns:
    The run_id of the analyzed test run.

  Raises:
    click.ClickException: If an error occurs during CHS analysis.
  """

  logger.debug("Analyzing run_id=%r", run_id)
  workspaces_base_dir = get_default_workspaces_dir()
  logger.debug("Using workspaces_base_dir: %s", workspaces_base_dir)
  ramble_manager = RambleManager(workspaces_base_dir, run_id)

  try:
    if ramble_manager.check_jobs_running():
      click.echo(
          f"Jobs for run {run_id} are still running. Please try again later."
      )
      logger.debug("Jobs deemed running for run_id=%r", run_id)
      return run_id

    logger.debug(
        "No jobs running for run_id=%r. Proceeding with analysis.", run_id
    )

    output_path = ramble_manager.analyze_results()
    logger.debug("Analysis complete for run_id=%r", run_id)
    click.echo(f"Results generated at: {output_path}")
    click.echo(f"Bill of Health: {output_path}/bill_of_health.json")
    click.echo(f"Logs: {output_path}/experiments")
  except click.ClickException:
    raise
  except (OSError, subprocess.CalledProcessError) as e:
    logger.exception("CHS analysis failed for run_id=%r", run_id)
    raise click.ClickException(f"CHS analysis failed: {e!r}") from e

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
    help="Name of the test suite to run",
    default="a3-ultra",
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
@click.option(
    "--nodes",
    help="Comma-separated list of individual nodes to run on",
    callback=non_empty_str_callback,
)
@click.option(
    "--nodelist",
    help="Slurm node list pattern to run on (e.g., node-[1-3])",
    callback=non_empty_str_callback,
)
@click.option(
    "--partition",
    help="Slurm partition to run on",
    callback=non_empty_str_callback,
)
@click.option(
    "--machine-type",
    required=True,
    help="The machine type of the cluster nodes (e.g., a3-high).",
    callback=non_empty_str_callback,
)
def run(
    *,
    suite: str,
    scheduler: str,
    tests: str | None,
    nodes: str | None,
    nodelist: str | None,
    partition: str | None,
    machine_type: str,
) -> tuple[str, str, list[str] | None, str, str, list[str]]:
  """Command to run a test suite.

  Args:
    suite: Name of the test suite to run.
    scheduler: Scheduler to use (e.g., slurm, gke).
    tests: Comma-separated list of specific tests to run (optional).
    nodes: Comma-separated list of individual nodes to run on.
    nodelist: Slurm node list pattern to run on (e.g., node-[1-3]).
    partition: Slurm partition to run on.
    machine_type: The machine type of the cluster nodes (e.g., a3-high).

  Returns:
    A tuple containing the suite name, scheduler name, a list of
    test names, workspace name, workspaces base directory, and job ids.
  """
  scheduler_obj = get_scheduler(
      scheduler, nodes=nodes, nodelist=nodelist, partition=partition
  )

  if scheduler_obj.name == "slurm":
    nodes_under_test = scheduler_obj.get_nodes_under_test()
    if not nodes_under_test:
      raise click.ClickException("No free nodes found to run on.")
  else:
    # TODO: Add support for other schedulers.
    raise click.ClickException("Only slurm scheduler is supported for now.")

  workspace_name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
  workspaces_base_dir = get_default_workspaces_dir()
  if not tests:
    tests = "nccl-cluster, nccl-pairwise, dcgm"
  parsed_tests = _parse_tests(tests)
  logger.debug("run_action: Parsed tests: %r", parsed_tests)
  logger.debug("Workspace name: %s", workspace_name)
  logger.debug("Workspaces base directory: %s", workspaces_base_dir)
  result = run_action(
      suite=suite,
      scheduler_obj=scheduler_obj,
      tests=parsed_tests,
      workspace_name=workspace_name,
      workspaces_base_dir=workspaces_base_dir,
      ramble_manager=RambleManager,
      machine_type=machine_type,
  )

  click.echo(f"Run submitted successfully.")
  click.echo(f"Request ID (Workspace Name): {workspace_name}")
  if result[5]:  # job_ids
    click.echo(f"Job IDs: {', '.join(result[5])}")
  else:
    click.echo("No Job IDs captured (running locally?)")

  click.echo(f"Query results using the command: chs analyze {workspace_name}")
  click.echo("Note: 'chs analyze' retrieves the latest results.")

  return result


@cli.command()
@click.argument("run_id", required=False, callback=non_empty_str_callback)
def analyze(run_id: str | None) -> str:
  """Analyze results from a test run.

  If no run_id is provided, the most recent run from the workspaces directory
  will be used.

  Args:
    run_id: The ID of the test run to analyze.

  Returns:
    The run_id of the analyzed test run.
  """
  if not run_id:
    workspaces_dir = pathlib.Path(get_default_workspaces_dir())
    if not workspaces_dir.exists():
      raise click.ClickException("No workspaces directory found.")

    valid_runs = []
    # %Y-%m-%d_%H.%M.%S.%f
    date_format = "%Y-%m-%d_%H.%M.%S.%f"
    for d in workspaces_dir.iterdir():
      if d.is_dir():
        try:
          parsed_time = datetime.datetime.strptime(d.name, date_format)
          valid_runs.append((parsed_time, d))
        except ValueError:
          # Directory name doesn't match the expected format; ignore it.
          continue

    if not valid_runs:
      raise click.ClickException("No previous runs found to analyze.")

    # Sort by the parsed datetime to find the latest
    latest_time, latest_dir = max(valid_runs, key=lambda item: item[0])
    run_id = latest_dir.name
    logger.info("No run_id provided. Using latest run: %s", run_id)
    click.echo(f"Analyzing latest run: {run_id}")

  return analyze_action(run_id)


@cli.group()
def show() -> None:
  """Show commands for displaying CHS information."""
  pass


@show.command()
@click.argument("run_id", required=False, callback=non_empty_str_callback)
@click.option(
    "--tests",
    "-t",
    help="Comma-separated list of test names to show details for.",
    callback=non_empty_str_callback,
)
def results(run_id: str | None, tests: str | None) -> None:
  """Show Bill of Health results for a run.

  If no run_id is provided, the most recent run from the workspaces directory
  will be used.
  """
  from chs.results_viewer import ResultsViewer

  workspaces_dir = pathlib.Path(get_default_workspaces_dir())
  if not run_id:
    if not workspaces_dir.exists():
      raise click.ClickException("No workspaces directory found.")

    valid_runs = []
    # %Y-%m-%d_%H.%M.%S.%f
    date_format = "%Y-%m-%d_%H.%M.%S.%f"
    for d in workspaces_dir.iterdir():
      if d.is_dir():
        try:
          parsed_time = datetime.datetime.strptime(d.name, date_format)
          valid_runs.append((parsed_time, d))
        except ValueError:
          # Directory name doesn't match the expected format; ignore it.
          continue

    if not valid_runs:
      raise click.ClickException("No previous runs found to analyze.")

    # Sort by the parsed datetime to find the latest
    latest_time, latest_dir = max(valid_runs, key=lambda item: item[0])
    run_id = latest_dir.name
    logger.info("No run_id provided. Using latest run: %s", run_id)
    click.echo(f"Showing results for latest run: {run_id}")
  current_dir_path = pathlib.Path(os.getcwd()) / run_id / "bill_of_health.json"
  if current_dir_path.exists():
    boh_path = current_dir_path
  else:
    click.echo(f"Could not find bill of health at {current_dir_path}.")
    click.echo(f"Have you run 'chs analyze {run_id}'?")
    return

  viewer = ResultsViewer(boh_path)

  parsed_tests = _parse_tests(tests)
  if parsed_tests:
    viewer.show_details(tests=parsed_tests)
  else:
    viewer.show_summary()


def main() -> None:
  """Main entry point for the CHS CLI."""
  logger.info("CHS CLI starting...")
  try:
    cli(obj={})
  finally:
    logger.info("CHS CLI finished.")
