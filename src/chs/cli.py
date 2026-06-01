"""Command-line interface implementation for the Cluster Health Scanner (CHS).

This module defines the CLI structure using click and provides the
logic for handling subcommands such as listing resources.
"""

import dataclasses
import datetime
import functools
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse
from google.cloud import storage

import click

from chs import (
    BaseScheduler,
    ChsAppData,
    RambleManager,
    ResultsViewer,
    exceptions as ex,
    get_scheduler,
    load_suite,
    resources,
    utils,
    validators,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RunExecutionResult:
    """Structured result of a CHS test suite run."""

    suite: str
    scheduler_name: str
    tests: list[dict[str, Any]] | list[str] | None
    workspace_name: str
    workspaces_base_dir: str
    job_ids: list[str] | None


def handle_chs_exceptions(f):
    """Decorator to handle CHS exceptions."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ex.ChsException as e:
            raise click.ClickException(str(e)) from e

    return wrapper


@contextmanager
def archive_workspace(
    ramble_manager: RambleManager, *, on_success: bool = True
):
    """Context manager to ensure workspace is archived even in error cases."""
    try:
        yield
    except Exception:
        ramble_manager.archive()
        raise
    else:
        if on_success:
            ramble_manager.archive()


@contextmanager
def cancel_jobs_on_failure(
    scheduler_obj: BaseScheduler, prefix: str, active: bool = True
):
    """Context manager to ensure jobs are cancelled on failure."""
    try:
        yield
    except BaseException:
        if active:
            logger.info(
                "Run failed or interrupted. Cleaning up spawned jobs..."
            )
            try:
                scheduler_obj.cancel_jobs_by_prefix(prefix)
                scheduler_obj.cleanup_scheduler_resources()
            except Exception as e:
                logger.error(f"Failed to cancel jobs during cleanup: {e}")
        raise


def _get_suites_list() -> list[str]:
    """Returns lines for available suites."""
    suites_list = utils.suites.get_all_suites()
    if suites_list:
        lines = ["Available Suites:"]
        lines.extend(f"  - {s}" for s in suites_list)
        return lines
    return ["No suites found."]


def _get_tests_list(
    from_suite: str | None, tests_flag: bool, machine_type: str | None = None
) -> list[str]:
    """Returns lines for available tests."""
    if from_suite:
        suite_cfg = load_suite(from_suite)
        tests = [
            t.get("name") if isinstance(t, dict) else t
            for t in suite_cfg.get("tests", [])
        ]
        if machine_type:
            supported_tests = utils.tests.get_tests_by_machine_type(
                machine_type
            )
            tests = [t for t in tests if t in supported_tests]
    elif machine_type:
        tests = utils.tests.get_tests_by_machine_type(machine_type)
    else:
        tests = utils.tests.get_all_supported_tests()

    if tests:
        lines = []
        if from_suite:
            lines.append(f"Available Tests for suite '{from_suite}':")
        else:
            lines.append("Available Tests:")

        if tests_flag:
            hints = []
            if not from_suite:
                hints.append("--from-suite <suite_name>")
            if not machine_type:
                hints.append("--machine-type <machine_type>")
            if hints:
                lines.append(
                    f"  (Hint: Use {' or '.join(hints)} to filter tests)"
                )

        if machine_type:
            lines.insert(1, f"  (Filtered by machine type: {machine_type})")
        lines.extend(f"  - {t}" for t in tests)
        return lines
    return ["No tests found."]


def _get_schedulers_list() -> list[str]:
    """Returns lines for available schedulers."""
    schedulers_list = utils.tests.get_all_supported_schedulers()
    if schedulers_list:
        lines = ["Supported Schedulers:"]
        lines.extend(f"  - {s}" for s in schedulers_list)
        return lines
    return ["No schedulers found."]


def list_action(
    *,
    suites: bool,
    tests: bool,
    schedulers: bool,
    from_suite: str | None,
    machine_type: str | None = None,
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
      machine_type: If provided, filters tests by machine type.

    Returns:
      True if all resource types were listed, False otherwise.
    """
    logger.debug(
        "list_action called with: suites=%r, tests=%r, schedulers=%r,"
        " from_suite=%r, machine_type=%r",
        suites,
        tests,
        schedulers,
        from_suite,
        machine_type,
    )
    show_all = not (suites or tests or schedulers)
    all_lines = []

    if suites or show_all:
        all_lines.extend(_get_suites_list())
        all_lines.append("")

    if tests or show_all:
        all_lines.extend(_get_tests_list(from_suite, tests, machine_type))
        all_lines.append("")

    if schedulers or show_all:
        all_lines.extend(_get_schedulers_list())
        all_lines.append("")

    if all_lines and not all_lines[-1]:
        all_lines.pop()

    if all_lines:
        logger.info("\n".join(all_lines))

    return show_all


def is_latest_chs_run_active(app_data: ChsAppData) -> bool:
    """Checks if the latest chs run is active by inspecting the runs registry.

    Args:
      app_data: ChsAppData instance.

    Returns:
      True if the latest run is active, False otherwise.
    """
    try:
        latest_run_id = app_data.get_latest_run_id()
        if not latest_run_id:
            return False

        workspace_name = app_data.resolve_run_id(latest_run_id)
        workspaces_dir = app_data.get_workspaces_dir()
        ramble_manager = RambleManager(
            base_dir=workspaces_dir,
            workspace_name=workspace_name,
            run_id=latest_run_id,
        )
        if ramble_manager.check_jobs_running():
            logger.warning(
                "The latest CHS run ('%s') is still active.", latest_run_id
            )
            return True
    except (ex.NoPreviousRunsFoundException, ex.WorkspaceNotFoundException):
        return False
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not check for active CHS runs: %s", e)

    return False


def _parse_str_to_list(value: str | None) -> list[str] | None:
    """Parses a comma-separated string.

    Args:
      value: A comma-separated string, or None.

    Returns:
      A list of cleaned strings, or None if the input was None.
    """

    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_run_args(
    scheduler: str,
    machine_type: str,
    tests: str | None,
    suite: str | None,
    validation: str | None = None,
) -> None:
    """Validates core run arguments against supported lists.

    Args:
      scheduler: Name of the scheduler.
      machine_type: Machine type string.
      tests: Comma-separated list of tests, or None.
      suite: Suite name, or None.
      validation: Path to thresholds validation file, or None.

    Raises:
      ex.ChsValidationError: If validation fails.
    """
    validator = validators.ProjectConfigValidator()
    errors = validator.validate(
        {
            "scheduler": scheduler,
            "machine_type": machine_type,
            "suite": suite,
            "tests": _parse_str_to_list(tests),
            "thresholds_path": validation,
        }
    )
    if errors:
        raise ex.ChsValidationError("\n".join(errors))


def run_action(
    *,
    suite: str,
    scheduler_obj: BaseScheduler,
    tests: list[dict[str, Any]] | None,
    workspace_name: str,
    workspaces_base_dir: str,
    machine_type: str,
    dry_run: bool = False,
    timeout: int | None = None,
    deadline: int | None = None,
    validation: str | None = None,
    skip_validation: bool = False,
    run_id: str | None = None,
) -> list[str] | None:
    """Validates parameters and orchestrates test suite execution.

    Args:
      suite: Name of the test suite to run.
      scheduler_obj: Scheduler object to use.
      tests: An optional list of test definitions to run.
      workspace_name: The name of the test workspace.
      workspaces_base_dir: The base directory for workspaces.
      machine_type: The machine type of the cluster nodes (e.g., a3-high).
      dry_run: Set up the workspace but do not execute experiments.
      timeout: The timeout for the job in seconds.
      deadline: The deadline for the job in seconds.
      thresholds: Path to user-defined thresholds YAML file.

    Returns:
      A list of submitted job IDs, or None if dry_run is True.

    Raises:
      ChsExecutionException: If an error occurs during CHS execution.
    """

    logger.debug(
        "run_action: Attempting to run suite=%r, scheduler=%r, tests=%r",
        suite,
        scheduler_obj.name,
        tests,
    )

    ramble_manager = RambleManager(
        workspaces_base_dir, workspace_name, run_id=run_id
    )

    with archive_workspace(ramble_manager, on_success=False):
        try:
            logger.info("Running preflight checks")
            if scheduler_obj:
                tests = scheduler_obj.run_preflight_checks(
                    tests, workspace_name=workspace_name, run_id=run_id
                )
                workspace_config_dir = os.path.join(
                    ramble_manager.path, "configs"
                )
                setup_tests = scheduler_obj.setup_scheduler_resources(
                    workspace_config_dir=workspace_config_dir,
                    tests=tests,
                    machine_type=machine_type,
                    workspace_name=workspace_name,
                )
                if setup_tests is not None:
                    tests = setup_tests

            ramble_manager.create_workspace()
            if not skip_validation:
                ramble_manager.copy_thresholds(validation)

            ramble_manager.add_custom_repository()
            ramble_manager.configure_workspace(
                scheduler_obj,
                tests,
                machine_type=machine_type,
                timeout=timeout,
                deadline=deadline,
            )
            ramble_manager.setup_workspace()
            if dry_run:
                logger.info("Setup complete. Skipping experiment execution.")
                return None

            job_ids = ramble_manager.execute_experiments()
            if not job_ids:
                raise ex.JobSubmissionFailureException(
                    f"Job submission failed for workspace: {workspace_name}. "
                    "Please check internal.log in the workspace directory "
                    "for details."
                )
            return job_ids
        except (OSError, subprocess.CalledProcessError) as e:
            raise ex.ChsExecutionException(
                f"CHS execution failed: {e!r}"
            ) from e


def analyze_action(
    app_data: ChsAppData,
    run_id: str,
    validation: str | None = None,
    skip_validation: bool = False,
) -> str:
    """Analyzes results from a specific test run.

    Args:
      app_data: ChsAppData instance.
      run_id: The ID of the test run to analyze.
      validation: Path to user-defined thresholds YAML file.

    Returns:
      The run_id of the analyzed test run.

    Raises:
      click.ClickException: If an error occurs during CHS analysis.
    """

    logger.debug("Analyzing run_id=%r", run_id)
    workspaces_base_dir = app_data.get_workspaces_dir()
    logger.debug("Using workspaces_base_dir: %s", workspaces_base_dir)
    workspace_name = app_data.resolve_run_id(run_id)
    ramble_manager = RambleManager(
        workspaces_base_dir, workspace_name, run_id=run_id
    )

    if not os.path.exists(ramble_manager.path):
        raise ex.WorkspaceNotFoundException(f"Workspace '{run_id}' not found.")

    with archive_workspace(ramble_manager, on_success=True):
        try:
            workspace_path = pathlib.Path(ramble_manager.path)
            boh_path = workspace_path / "bill_of_health.json"
            results_json_path = workspace_path / "results.latest.json"

            # 1. Check if Bill of Health already exists
            if boh_path.exists() and not validation:
                _show_results(app_data, run_id)
                return run_id

            # 2. If results.latest.json exists (but no BoH), generate BoH
            if results_json_path.exists():
                logger.info(
                    "Found results from this run. Generating Bill of Health..."
                )
                ramble_manager.generate_boh(
                    str(results_json_path),
                    thresholds_path=validation,
                    skip_validation=skip_validation,
                )
                _show_results(app_data, run_id)
                return run_id

            # 3. If neither exists, wait for jobs and then analyze
            logger.debug(f"Waiting for jobs for {run_id}...")
            ramble_manager.wait_for_jobs(app_data=app_data)

            ramble_manager.analyze_results(
                thresholds_path=validation, skip_validation=skip_validation
            )
            logger.debug("Analysis complete for run_id=%r", run_id)
            _show_results(app_data, run_id)
        except (OSError, subprocess.CalledProcessError) as e:
            raise ex.ChsAnalysisException(
                f"CHS analysis failed for run_id={run_id}: {e!r}"
            ) from e
        else:
            return run_id


def _get_gcs_blob(gcs_path: str) -> storage.Blob:
    """Helper to get GCS blob from URI."""
    parsed_uri = urlparse(gcs_path)
    bucket_name = parsed_uri.netloc
    blob_name = parsed_uri.path.lstrip("/")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    return bucket.blob(blob_name)


def _download_file(gcs_path: str, dest_path: str) -> None:
    """Downloads a file from GCS."""
    logger.info("Downloading file from GCS: %s", gcs_path)
    try:
        blob = _get_gcs_blob(gcs_path)
        blob.download_to_filename(dest_path)
    except Exception as e:
        raise ex.ChsException(f"Failed to download file from GCS: {e}") from e


def _write_file_content(f, dest_path: str) -> None:
    """Writes content of extracted file to dest_path."""
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(f, out)


def _extract_from_archive(gcs_path: str, dest_path: str) -> None:
    """Extracts bill_of_health.json from GCS archive."""
    logger.info("Reading results from GCS archive: %s", gcs_path)
    try:
        blob = _get_gcs_blob(gcs_path)
        with blob.open("rb") as gcs_file:
            with tarfile.open(fileobj=gcs_file, mode="r:gz") as tar:
                f = None
                try:
                    f = tar.extractfile("bill_of_health.json")
                    if f is None:
                        logger.debug(
                            "bill_of_health.json found but is not a regular"
                            " file"
                        )
                except KeyError:
                    logger.debug(
                        "bill_of_health.json not found at root, searching"
                        " archive"
                    )

                if f is None:
                    boh_member = None
                    for member in tar.getmembers():
                        if member.name.endswith("bill_of_health.json"):
                            boh_member = member
                            break
                    if not boh_member:
                        raise ex.ChsException("BoH not found in archive")
                    f = tar.extractfile(boh_member)
                    if f is None:
                        raise ex.ChsException(
                            "BoH found in archive but is not a regular file"
                        )

                _write_file_content(f, dest_path)
    except ex.ChsException:
        raise
    except Exception as e:
        raise ex.ChsException(f"Failed to extract from GCS archive: {e}") from e


def _download_boh_from_gcs(gcs_path: str, dest_path: str) -> None:
    """Downloads Bill of Health from GCS to dest_path."""
    if not gcs_path.startswith("gs://"):
        gcs_path = "gs://" + gcs_path

    if gcs_path.endswith(".json"):
        _download_file(gcs_path, dest_path)
    elif gcs_path.endswith(".tar.gz") or gcs_path.endswith(".tgz"):
        _extract_from_archive(gcs_path, dest_path)
    else:
        # Directory
        gcs_json_path = gcs_path.rstrip("/") + "/bill_of_health.json"
        _download_file(gcs_json_path, dest_path)


def _show_results(
    app_data: ChsAppData,
    run_id: str | None,
    tests: str | None = None,
    resources: str | None = None,
    gcs: str | None = None,
) -> None:
    """Show Bill of Health results for a run."""
    workspaces_dir = pathlib.Path(app_data.get_workspaces_dir())

    if gcs:
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".json"
        ) as tmp_file:
            _download_boh_from_gcs(gcs, tmp_file.name)
            viewer = ResultsViewer(tmp_file.name)
    else:
        # Local run
        if not run_id:
            run_id = app_data.get_latest_run_id()
            logger.info("No run_id provided. Using latest run: %s", run_id)
            logger.info("Showing results for latest run: %s", run_id)

        workspace_name = app_data.resolve_run_id(run_id)
        boh_path = workspaces_dir / workspace_name / "bill_of_health.json"
        if not boh_path.exists():
            raise ex.BillOfHealthNotFoundException(
                f"Could not find bill of health at {boh_path}. "
                f"Have you run 'chs analyze {run_id}'?"
            )
        viewer = ResultsViewer(boh_path)

    parsed_tests = _parse_str_to_list(tests)
    parsed_resources = _parse_str_to_list(resources)

    viewer.show_details(tests=parsed_tests, resources=parsed_resources)


def cancel_action(app_data: ChsAppData, run_id: str) -> str:
    """Cancels jobs for a specific test run.

    Args:
      app_data: ChsAppData instance.
      run_id: The ID of the test run to cancel.

    Returns:
      The run_id of the cancelled test run.

    Raises:
      click.ClickException: If an error occurs during CHS job cancellation.
    """
    logger.debug("Canceling run_id=%r", run_id)
    workspaces_base_dir = app_data.get_workspaces_dir()
    logger.debug("Using workspaces_base_dir: %s", workspaces_base_dir)
    workspace_name = app_data.resolve_run_id(run_id)
    ramble_manager = RambleManager(
        workspaces_base_dir, workspace_name, run_id=run_id
    )

    with archive_workspace(ramble_manager, on_success=True):
        try:
            ramble_manager.cancel_experiments()
            logger.debug("Cancellation complete for run_id=%r", run_id)
            logger.info("Jobs for run %s have been canceled.", run_id)
        except (OSError, subprocess.CalledProcessError) as e:
            raise ex.ChsCancellationException(
                f"CHS cancellation failed for run_id={run_id}: {e!r}"
            ) from e
        else:
            return run_id


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Cluster Health Scanner (CHS) Tool."""
    ctx.ensure_object(dict)
    ctx.obj["app_data"] = ChsAppData()
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
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--machine-type",
    help="Filter tests by machine type (only used with --tests)",
    callback=validators.validate_non_empty_string,
)
def list_command(
    *,
    suites: bool,
    tests: bool,
    schedulers: bool,
    from_suite: str | None,
    machine_type: str | None,
) -> bool:
    """Lists available resources.

    \b
    Args:
      suites: Whether to list available test suites.
      tests: Whether to list available tests.
      schedulers: Whether to list available schedulers.
      from_suite: Optional suite name to filter tests.
      machine_type: Optional machine type to filter tests.
    """
    if from_suite and not tests:
        raise click.UsageError(
            "Option '--from-suite' requires option '--tests'."
        )
    if machine_type and not tests:
        raise click.UsageError(
            "Option '--machine-type' requires option '--tests'."
        )
    if machine_type:
        supported_mts = utils.tests.get_all_supported_machine_types()
        if machine_type not in supported_mts:
            raise click.UsageError(
                f"Unsupported machine type '{machine_type}'. "
                f"Supported types are: {', '.join(supported_mts)}"
            )

    return list_action(
        suites=suites,
        tests=tests,
        schedulers=schedulers,
        from_suite=from_suite,
        machine_type=machine_type,
    )


@cli.command()
@handle_chs_exceptions
@click.option(
    "--suite",
    "-s",
    help="Name of the test suite to run",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--scheduler",
    "-sch",
    required=True,
    help="Scheduler to use (e.g., slurm, gke)",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--tests",
    "-t",
    help="Comma-separated list of specific tests to run (optional)",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--nodes",
    help="Comma-separated list of individual nodes to run on",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--nodelist",
    help="Slurm node list pattern to run on (e.g., node-[1-3])",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--nodepool",
    help="GKE node pool to run on",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--partition",
    help="Slurm partition to run on",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--machine-type",
    required=True,
    help="The machine type of the cluster nodes (e.g., a3-high).",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Set up the workspace but do not execute experiments or analyze.",
)
@click.option(
    "--validation",
    help=(
        "Path to user-defined thresholds YAML file (optional). "
        "See src/chs/configs/thresholds.yaml for template."
    ),
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Skip threshold evaluation entirely.",
)
@click.option(
    "--run-id",
    help="Custom run ID for this execution.",
    callback=validators.validate_run_id,
)
@click.pass_obj
def run(
    obj,
    *,
    suite: str | None,
    scheduler: str,
    tests: str | None,
    nodes: str | None,
    nodelist: str | None,
    nodepool: str | None,
    partition: str | None,
    machine_type: str,
    dry_run: bool,
    validation: str | None = None,
    skip_validation: bool = False,
    run_id: str | None = None,
) -> RunExecutionResult:
    """Command to run a test suite.

    \b
    Args:
      suite: Name of the test suite to run.
      scheduler: Scheduler to use (e.g., slurm, gke).
      tests: Comma-separated list of specific tests to run (optional).
      nodes: Comma-separated list of individual nodes to run on.
      nodelist: Slurm node list pattern to run on (e.g., node-[1-3]).
      nodepool: GKE node pool to run on.
      partition: Slurm partition to run on.
      machine_type: The machine type of the cluster nodes (e.g., a3-high).
      dry_run: Set up the workspace but do not execute experiments or analyze.

    Returns:
      A RunExecutionResult object containing details of the run.
    """
    app_data = obj["app_data"]
    if is_latest_chs_run_active(app_data):
        raise ex.ChsRunInProgressException(
            "The latest CHS run is still in progress. Please wait for it to"
            " complete or cancel it before submitting a new run."
        )

    _validate_run_args(scheduler, machine_type, tests, suite, validation)

    timeout = resources.DEFAULT_TIMEOUT
    deadline = resources.DEFAULT_DEADLINE

    if not tests:
        if not suite:
            suite = "lite"

        suite_cfg = load_suite(suite)
        resolved_tests = suite_cfg.get("tests", [])

        supported_tests = utils.tests.get_tests_by_machine_type_and_scheduler(
            machine_type, scheduler
        )
        resolved_tests = [
            t
            for t in resolved_tests
            if (t.get("name") if isinstance(t, dict) else t) in supported_tests
        ]
        if not resolved_tests:
            raise ex.ChsValidationError(
                f"No supported tests found in suite '{suite}' for"
                f"scheduler '{scheduler}' and machine type '{machine_type}'."
            )

        try:
            timeout = int(suite_cfg.get("timeout"))
        except (ValueError, TypeError):
            pass  # nothing to do here, as the default value will be used

        try:
            deadline = int(suite_cfg.get("deadline"))
        except (ValueError, TypeError):
            pass  # nothing to do here, as the default value will be used

        timeout = max(timeout, resources.DEFAULT_TIMEOUT)
        deadline = max(deadline, timeout + 600)

        test_names = [
            t.get("name", str(t)) if isinstance(t, dict) else str(t)
            for t in resolved_tests
        ]
        logger.info(
            f"Loaded {len(resolved_tests)} tests ({', '.join(test_names)}) "
            f"for suite '{suite}' from suites.yaml."
        )
        parsed_tests = resolved_tests
    else:
        # TODO: Validate input tests with supported tests
        parsed_tests_strs = _parse_str_to_list(tests)
        parsed_tests = (
            [{"name": t} for t in parsed_tests_strs]
            if parsed_tests_strs
            else []
        )
        if not suite:
            suite = "custom"

    scheduler_obj = get_scheduler(
        scheduler,
        nodes=nodes,
        nodelist=nodelist,
        nodepool=nodepool,
        partition=partition,
        app_data=app_data,
    )

    if scheduler_obj.name == "slurm" or scheduler_obj.name == "gke":
        nodes_under_test = scheduler_obj.get_nodes_under_test()
        if not nodes_under_test:
            raise ex.NoNodesProvidedException("No nodes provided to run on.")
    else:
        # TODO: Add support for other schedulers.
        raise ex.UnsupportedSchedulerException(
            "Only Slurm and GKE schedulers are supported."
        )

    short_hash = utils.generate_random_suffix()
    if not run_id:
        if dry_run:
            cluster_name = "dry-run-cluster"
        else:
            cluster_name = utils.get_sanitized_cluster_name(scheduler_obj)
        run_id = f"{cluster_name}-{suite}"
    run_id = f"{run_id}-{short_hash}"

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    workspace_name = f"{timestamp}_{run_id}"
    workspaces_base_dir = app_data.get_workspaces_dir()

    logger.debug("run_action: Parsed tests: %r", parsed_tests)
    logger.debug("Workspace name: %s", workspace_name)
    logger.info("Run ID: %s", run_id)
    logger.debug("Workspaces base directory: %s", workspaces_base_dir)

    with cancel_jobs_on_failure(scheduler_obj, "chs-", active=not dry_run):
        job_ids = run_action(
            suite=suite,
            scheduler_obj=scheduler_obj,
            tests=parsed_tests,
            workspace_name=workspace_name,
            workspaces_base_dir=workspaces_base_dir,
            machine_type=machine_type,
            dry_run=dry_run,
            timeout=timeout,
            deadline=deadline,
            validation=validation,
            skip_validation=skip_validation,
            run_id=run_id,
        )

        result = RunExecutionResult(
            suite=suite,
            scheduler_name=scheduler_obj.name,
            tests=parsed_tests,
            workspace_name=workspace_name,
            workspaces_base_dir=workspaces_base_dir,
            job_ids=job_ids,
        )

        # TODO: Revisit command statement normalization when
        # distributing as a Docker image.
        # Normalize command statement to start with "chs"
        command_statement = " ".join(["chs"] + sys.argv[1:])

        app_data.update_run(
            workspace_name,
            {
                "run_id": run_id,
                "workspace_name": workspace_name,
                "workspace_path": os.path.join(
                    workspaces_base_dir, workspace_name
                ),
                "submission_time": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "scheduler": scheduler,
                "command_statement": command_statement,
            },
        )

        if dry_run:
            logger.info("Dry run completed. Workspace setup is ready.")
            return result

        if not job_ids:
            raise ex.JobSubmissionFailureException(
                f"Job submission failed for workspace: {workspace_name}. "
                "Please check internal.log for details."
            )

        logger.info("Run submitted successfully.")
        logger.info("Job IDs: %s", ", ".join(job_ids))

    # Automatically trigger analysis
    analyze_action(
        app_data,
        run_id,
        validation=validation,
        skip_validation=skip_validation,
    )

    return result


@cli.command()
@handle_chs_exceptions
@click.argument(
    "run_id", required=False, callback=validators.validate_non_empty_string
)
@click.option(
    "--validation",
    help="Path to user-defined thresholds YAML file (overrides workspace copy)",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Skip threshold evaluation entirely.",
)
@click.pass_obj
def analyze(
    obj,
    run_id: str | None,
    validation: str | None = None,
    skip_validation: bool = False,
) -> str:
    """Analyze results from a test run.

    If no run_id is provided, the most recent run from the workspaces directory
    will be used.

    \b
    Args:
      obj: Click object containing app_data.
      run_id: The ID of the test run to analyze.
      thresholds: Path to user-defined thresholds YAML file.

    Returns:
      The run_id of the analyzed test run.
    """
    app_data = obj["app_data"]
    if not run_id:
        run_id = app_data.get_latest_run_id()
        logger.info("No run_id provided. Using latest run: %s", run_id)
        logger.info("Analyzing latest run: %s", run_id)

    if validation:
        errors = validators.ThresholdsValidator().validate(
            {"thresholds_path": validation}
        )
        if errors:
            raise ex.ChsValidationError("\n".join(errors))

    return analyze_action(
        app_data, run_id, validation=validation, skip_validation=skip_validation
    )


@cli.command()
@handle_chs_exceptions
@click.pass_obj
def cancel(obj) -> str:
    """Cancel all running CHS jobs (using scheduler from latest run)."""
    app_data = obj["app_data"]
    run_id = app_data.get_latest_run_id()
    return cancel_action(app_data, run_id)


@cli.command()
@handle_chs_exceptions
@click.argument(
    "run_id", required=False, callback=validators.validate_non_empty_string
)
@click.option(
    "--tests",
    "-t",
    help="Comma-separated list of test names to show details for.",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--resources",
    "-r",
    help="Comma-separated list of resource names to show details for.",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--gcs",
    help="Path to Bill of Health in GCS (bucket/path/to/file)",
    callback=validators.validate_non_empty_string,
)
@click.pass_obj
def results(
    obj,
    run_id: str | None,
    tests: str | None,
    resources: str | None,
    gcs: str | None,
) -> None:
    """Show Bill of Health results for a run.

    If no run_id is provided, the most recent run from the workspaces directory
    will be used.
    """
    _show_results(obj["app_data"], run_id, tests, resources, gcs)


@cli.command()
@click.pass_obj
def history(obj) -> None:
    """Show the history of last 5 runs."""
    app_data = obj["app_data"]
    runs = app_data.get_latest_runs(5)

    if not runs:
        click.echo("No runs found.")
        return

    click.echo(f"{'Workspace':<30} {'Details'}")
    click.echo("-" * 90)
    for run in runs.values():
        run_id = run.get("run_id", "N/A")

        command_statement = run.get("command_statement", "")
        if not command_statement:
            # Reconstruct for old runs or runs without stored statement
            scheduler = run.get("scheduler", "unknown")
            command_statement = f"chs run --scheduler {scheduler}"

        click.echo(f"{run_id:<30} {command_statement}")


@cli.group()
def configure() -> None:
    """Configure CHS settings."""
    pass


@configure.command("bucket")
@click.option(
    "--name",
    "bucket_name",
    required=False,
    default="chs-results",
    help="GCS bucket name for storing results",
)
@click.pass_context
def set_bucket_command(ctx: click.Context, bucket_name: str) -> None:
    """Sets the GCS bucket for storing results."""
    app_data = ctx.obj["app_data"]
    # TODO: Create bucket if it doesn't exist
    app_data.set_bucket(bucket_name)
    click.echo(f"GCS bucket set to: {bucket_name}")


def main() -> None:
    """Main entry point for the CHS CLI."""

    logger.info("CHS CLI starting...")
    try:
        cli(obj={})
    finally:
        logger.info("CHS CLI finished.")
