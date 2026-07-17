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

"""Command-line interface implementation for the Cluster Health Scanner (CHS).

This module defines the CLI structure using click and provides the
logic for handling subcommands such as listing resources.
"""

import dataclasses
import datetime
import functools
import logging
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from typing import Any

import click
import questionary
from rich.console import Console

from chs import (
    BaseScheduler,
    ChsAppData,
    RambleManager,
    ResultsViewer,
    detect_scheduler,
    exceptions as ex,
    get_scheduler,
    load_suite,
    is_suite_tiered,
    preflights,
    resources,
    utils,
    validators,
)
from chs.boh_generator import parse_tier_results_from_boh
from chs.tier_manager import TierManager
from chs.utils import theme
from chs.storage import StorageProviderFactory
from chs.cli import list_formatting

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
        except ex.ChsDetachedException:
            raise click.exceptions.Exit(0)
        except ex.ChsException as e:
            raise click.ClickException(str(e)) from e

    return wrapper


def _setup_workspace_logger(workspace_path: str) -> None:
    """Configures a file handler to save all DEBUG logs to the workspace."""
    debug_log_path = os.path.join(workspace_path, "debug.txt")
    root_logger = logging.getLogger()
    if any(
        getattr(h, "baseFilename", "") == debug_log_path
        for h in root_logger.handlers
    ):
        return

    os.makedirs(workspace_path, exist_ok=True)
    file_handler = logging.FileHandler(debug_log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    root_logger.addHandler(file_handler)
    if root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers:
            if handler != file_handler and handler.level == logging.NOTSET:
                handler.setLevel(logging.INFO)


def _archive_and_log(ramble_manager: RambleManager, spinner_label: str) -> None:
    """Archives the workspace and logs the results with a progress spinner."""
    console = Console()
    with theme.get_progress_spinner(spinner_label):
        ws_path, boh_path = ramble_manager.archive()
    storage_provider = StorageProviderFactory.get_provider_for_run(
        ramble_manager.workspace_name
    )
    if ws_path and storage_provider:
        console.print(
            f"[{theme.CHS_SUCCESS_STYLE}]Workspace archived to: {ws_path}"
            f"[/{theme.CHS_SUCCESS_STYLE}]"
        )
    if boh_path and storage_provider:
        console.print(
            f"[{theme.CHS_SUCCESS_STYLE}]Bill of Health saved at: "
            f"{boh_path}[/{theme.CHS_SUCCESS_STYLE}]"
        )


@contextmanager
def archive_workspace(
    ramble_manager: RambleManager, *, on_success: bool = True
):
    """Context manager to ensure workspace is archived even in error cases."""
    try:
        yield
    except ex.ChsDetachedException:
        raise
    except Exception:
        _archive_and_log(
            ramble_manager,
            "[bold blue]Archiving workspace and saving "
            "Bill of Health...[/bold blue]",
        )
        raise
    else:
        if on_success:
            _archive_and_log(
                ramble_manager,
                "[bold blue]Archiving workspace and saving "
                "Bill of Health...[/bold blue]",
            )


@contextmanager
def cancel_jobs_on_failure(
    scheduler_obj: BaseScheduler, prefix: str, active: bool = True
):
    """Context manager to ensure jobs are cancelled on failure."""
    try:
        yield
    except BaseException as e:
        if active:
            is_interrupt = isinstance(e, KeyboardInterrupt)
            spinner_label = (
                "[bold red]Interrupt received. Cleaning up CHS resources..."
                "[/bold red]"
                if is_interrupt
                else "[bold red]Run failed. Cleaning up CHS resources..."
                "[/bold red]"
            )

            with theme.get_progress_spinner(spinner_label):
                try:
                    scheduler_obj.cancel_jobs_by_prefix(prefix)
                    scheduler_obj.cleanup_scheduler_resources()
                except Exception as e:
                    # We catch broad Exception here as a fallback to ensure we
                    # don't hide the original exception that triggered cleanup.
                    logger.error(f"Failed to clean up CHS resources: {e}")
                    Console().print(
                        "\n[bold red]WARNING: Automatic cleanup failed. "
                        "You may have resources from lingering tests on the "
                        "cluster. Please run `chs cancel` to manually clean "
                        "up."
                        "[/bold red]"
                    )

            if is_interrupt:
                Console().print(
                    f"[{theme.CHS_SUCCESS_STYLE}]Successfully cleaned up CHS "
                    "resources."
                    f"[/{theme.CHS_SUCCESS_STYLE}]"
                )
        raise


def _submit_next_tier(
    app_data: ChsAppData,
    ramble_manager: RambleManager,
    run_entry: dict[str, Any],
    workspace_name: str,
    next_tier: int,
    next_tests: list[str],
) -> None:
    """Submits the next tier of diagnostic tests."""
    Console().print(
        f"[{theme.CHS_WARNING_STYLE}]Degraded performance detected. "
        f"Triggering follow-up diagnostics: {', '.join(next_tests)}"
        f"[/{theme.CHS_WARNING_STYLE}]"
    )
    run_entry.update(
        {
            "active_tier": next_tier,
            "active_tier_tests": next_tests,
            "submission_complete": False,
        }
    )
    app_data.update_run(workspace_name, run_entry)

    scheduler = get_scheduler(run_entry.get("scheduler", "kubernetes"))
    with cancel_jobs_on_failure(scheduler, "chs-", active=True):
        with theme.get_progress_spinner("Submitting follow-up diagnostics..."):
            ramble_manager.execute_experiments(
                tier=next_tier, active_tests=next_tests
            )

    Console().print(
        f"[{theme.CHS_SUCCESS_STYLE}]Successfully submitted "
        "follow-up diagnostics."
        f"[/{theme.CHS_SUCCESS_STYLE}]"
    )


def list_action(
    *,
    suites: bool = False,
    tests: bool = False,
    schedulers: bool = False,
    machine_types: bool = False,
    suite: str | None = None,
    machine_type: str | None = None,
    accelerator_type: str | None = None,
) -> bool:
    """Orchestrates listing of available CHS resources.

    Based on the boolean flags, it calls the appropriate listing functions.
    If no flags are set to True, it defaults to listing all resource types.

    Args:
      suites: If True, includes test suites in the list.
      tests: If True, includes tests in the list.
      schedulers: If True, includes schedulers in the list.
      machine_types: If True, includes machine types in the list.
      suite: If provided, filters tests listed to those belonging to the
        specified suite.
      machine_type: If provided, filters tests by machine type.

    Returns:
      True if all resource types were listed, False otherwise.
    """
    logger.debug(
        "list_action called with: suites=%r, tests=%r, schedulers=%r, "
        "machine_types=%r, suite=%r, machine_type=%r",
        suites,
        tests,
        schedulers,
        machine_types,
        suite,
        machine_type,
        accelerator_type,
    )
    show_all = not (suites or tests or schedulers or machine_types)
    console = Console()

    if suites or show_all:
        console.print(list_formatting.get_suites_renderable())
        console.print()

    if tests or show_all:
        for r in list_formatting.get_tests_renderables(
            suite, tests, machine_type, accelerator_type
        ):
            console.print(r)
        console.print()

    if schedulers or show_all:
        console.print(list_formatting.get_schedulers_renderable())
        console.print()

    if machine_types or show_all:
        console.print(list_formatting.get_machine_types_renderable())
        console.print()

    return show_all


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


def _get_test_name(test: dict[str, Any] | str) -> str:
    """Extracts test name string from a test dictionary or string."""
    return test.get("name") if isinstance(test, dict) else str(test)


def _validate_run_args(
    scheduler: str,
    detected_machine_types: list[str],
    tests: str | None,
    suite: str | None,
    validation: str | None = None,
) -> None:
    """Validates core run arguments against supported lists.

    Args:
      scheduler: Name of the scheduler.
      detected_machine_types: List of detected machine types.
      tests: Comma-separated list of tests, or None.
      suite: Suite name, or None.
      validation: Path to thresholds validation file, or None.

    Raises:
      ex.ChsValidationError: If validation fails.
    """
    homogeneity_validator = validators.HomogeneityValidator()
    errors = homogeneity_validator.validate(
        {
            "detected_machine_types": detected_machine_types,
            "scheduler": scheduler,
        }
    )
    if errors:
        raise ex.ChsValidationError("\n".join(errors))

    machine_type = detected_machine_types[0]

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
    active_tests: list[str] | None = None,
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

    _setup_workspace_logger(ramble_manager.path)

    with archive_workspace(ramble_manager, on_success=False):
        try:
            if scheduler_obj:
                workspace_config_dir = os.path.join(
                    ramble_manager.path, "configs"
                )
                tests = scheduler_obj.setup_scheduler_resources(
                    workspace_config_dir=workspace_config_dir,
                    tests=tests,
                    machine_type=machine_type,
                    workspace_name=workspace_name,
                )

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
                logger.debug("Setup complete. Skipping experiment execution.")
                return None

            scheduler_obj.apply_scheduler_resources()

            job_ids = ramble_manager.execute_experiments(
                tier=1, active_tests=active_tests
            )
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
      skip_validation: Whether to skip generating validation criteria.

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

    _setup_workspace_logger(ramble_manager.path)

    if not os.path.exists(ramble_manager.path):
        raise ex.WorkspaceNotFoundException(f"Workspace '{run_id}' not found.")

    with archive_workspace(ramble_manager, on_success=True):
        try:
            workspace_path = pathlib.Path(ramble_manager.path)
            boh_path = workspace_path / "bill_of_health.json"
            results_json_path = workspace_path / "results.latest.json"

            run_entry = app_data.get_run(workspace_name) or {}
            is_tiered = run_entry.get("is_tiered", False)
            is_boh_final = run_entry.get("is_boh_final", not is_tiered)

            # 1. Check if Bill of Health already exists
            if boh_path.exists() and not validation and is_boh_final:
                logger.debug(
                    "Bill of Health already exists. Showing results directly."
                )
                Console().print(
                    f"[{theme.CHS_SUCCESS_STYLE}]Found existing "
                    f"Bill of Health for {run_id}."
                    f"[/{theme.CHS_SUCCESS_STYLE}]"
                )
                _show_results(app_data, run_id)
                return run_id

            # 2. If results.latest.json exists (but no BoH), generate BoH
            if results_json_path.exists() and is_boh_final:
                logger.debug(
                    "Found results from this run. Generating Bill of Health..."
                )
                with theme.get_progress_spinner(
                    "Generating Bill of Health and cleaning up..."
                ):
                    ramble_manager.generate_boh(
                        str(results_json_path),
                        thresholds_path=validation,
                        skip_validation=skip_validation,
                    )
                    try:
                        ramble_manager.cancel_experiments()
                        logger.debug("Cleaned up jobs for run %s", run_id)
                    except (
                        ex.ChsException,
                        subprocess.CalledProcessError,
                    ) as e:
                        logger.warning(
                            "Failed to cleanup tests after analysis: %s", e
                        )
                Console().print(
                    f"[{theme.CHS_SUCCESS_STYLE}]Bill of Health generated "
                    "and CHS resources cleaned up."
                    f"[/{theme.CHS_SUCCESS_STYLE}]"
                )
                _show_results(app_data, run_id)
                return run_id

            # 3. If neither exists, wait for jobs and then analyze
            suite_name = run_entry.get("suite")
            suite_cfg = load_suite(suite_name) if is_tiered else {}
            tier_manager = (
                TierManager(suite_cfg.get("tests", [])) if is_tiered else None
            )

            while not run_entry.get("is_boh_final", False):
                active_tier = run_entry.get("active_tier", 1)
                active_tier_tests_raw = run_entry.get("active_tier_tests") or []
                active_tier_tests = set(
                    t.get("name") if isinstance(t, dict) else t
                    for t in active_tier_tests_raw
                )

                if tier_manager:
                    tier_manager.fast_forward_to_tier(active_tier)

                logger.debug("Waiting for tier %s jobs...", active_tier)
                ramble_manager.wait_for_jobs(
                    app_data=app_data, tier=active_tier if is_tiered else None
                )

                if is_tiered and active_tier == 1:
                    spinner_msg = "Analyzing initial health checks..."
                elif is_tiered:
                    spinner_msg = "Analyzing follow-up diagnostics..."
                else:
                    spinner_msg = "Analyzing logs..."

                with theme.get_progress_spinner(spinner_msg):
                    results_path = ramble_manager.analyze_experiments()
                    boh_dir = ramble_manager.generate_boh(
                        results_path,
                        thresholds_path=validation,
                        skip_validation=skip_validation,
                    )
                Console().print(
                    f"[{theme.CHS_SUCCESS_STYLE}]Successfully "
                    f"analyzed logs.[/{theme.CHS_SUCCESS_STYLE}]"
                )

                next_tests = None
                if tier_manager:
                    boh_file = os.path.join(boh_dir, "bill_of_health.json")
                    tier_results, hard_failures_present = (
                        parse_tier_results_from_boh(boh_file, active_tier_tests)
                    )
                    if hard_failures_present:
                        Console().print(
                            f"[{theme.CHS_WARNING_STYLE}]Hard failure "
                            "detected. Halting follow-up diagnostic tiers."
                            f"[/{theme.CHS_WARNING_STYLE}]"
                        )
                        next_tests = None
                    else:
                        next_tests = tier_manager.get_next_tier_tests(
                            tier_results
                        )

                if not next_tests:
                    with theme.get_progress_spinner(
                        "Cleaning up CHS resources..."
                    ):
                        try:
                            ramble_manager.cancel_experiments()
                            logger.debug("Cleaned up tests for run %s", run_id)
                        except (
                            ex.ChsException,
                            subprocess.CalledProcessError,
                        ) as e:
                            logger.warning(
                                "Failed to cleanup jobs after analysis: %s", e
                            )

                    run_entry["is_boh_final"] = True
                    app_data.update_run(workspace_name, run_entry)
                    Console().print(
                        f"[{theme.CHS_SUCCESS_STYLE}]All checks completed. "
                        "Bill of Health generated."
                        f"[/{theme.CHS_SUCCESS_STYLE}]"
                    )
                    break

                _submit_next_tier(
                    app_data,
                    ramble_manager,
                    run_entry,
                    workspace_name,
                    active_tier + 1,
                    list(next_tests),
                )

            _show_results(app_data, run_id)
        except (OSError, subprocess.CalledProcessError) as e:
            raise ex.ChsAnalysisException(
                f"CHS analysis failed for run_id={run_id}: {e!r}"
            ) from e
        else:
            return run_id


def _show_results(
    app_data: ChsAppData,
    run_id: str | None,
    tests: str | None = None,
    resources: str | None = None,
    details: bool = False,
) -> None:
    """Show Bill of Health results for a run."""
    workspaces_dir = pathlib.Path(app_data.get_workspaces_dir())

    if not run_id:
        run_id = app_data.get_latest_run_id()
        logger.debug("No run_id provided. Using latest run: %s", run_id)
        logger.debug("Showing results for latest run: %s", run_id)

    # Try to resolve run_id to workspace_name
    try:
        workspace_name = app_data.resolve_run_id(run_id)
    except ex.WorkspaceNotFoundException:
        # Fallback to run_id if not found in local registry
        workspace_name = run_id

    boh_path = workspaces_dir / workspace_name / "bill_of_health.json"
    run_provider = StorageProviderFactory.get_provider_for_run(workspace_name)
    boh_uri = (
        run_provider.get_uri(f"bills_of_health/{workspace_name}.json")
        if run_provider
        else None
    )

    if not boh_path.exists():
        logger.debug(
            "Local Bill of Health not found. "
            "Attempting to download from storage..."
        )
        try:
            provider = (
                run_provider or StorageProviderFactory.get_active_provider()
            )
        except Exception as e:
            raise ex.BillOfHealthNotFoundException(
                f"Could not find local bill of health at {boh_path} "
                f"and failed to resolve storage provider: {e}"
            ) from e

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file_path = os.path.join(tmp_dir, "boh.json")
            try:
                provider.download_file(
                    f"bills_of_health/{workspace_name}.json", tmp_file_path
                )
                viewer = ResultsViewer(tmp_file_path, boh_uri=boh_uri)
            except ex.StorageObjectNotFoundException as e:
                logger.debug("Storage object not found: %s", e)
                raise ex.BillOfHealthNotFoundException(
                    f"Results not found for run '{run_id}'.\n"
                    "This usually happens if the jobs are still running, "
                    "the run was canceled, or the upload failed.\n"
                    f"Tip: Run 'chs analyze {run_id}' to re-attach to the run "
                    "and check its status."
                ) from e
            except Exception as e:
                raise ex.BillOfHealthNotFoundException(
                    f"Failed to retrieve Bill of Health for run '{run_id}' "
                    f"from storage: {e}"
                ) from e
    else:
        viewer = ResultsViewer(boh_path, boh_uri=boh_uri)

    parsed_tests = _parse_str_to_list(tests)
    parsed_resources = _parse_str_to_list(resources)

    viewer.show_summary_and_details(
        tests=parsed_tests, resources=parsed_resources, show_details=details
    )


def cancel_action(scheduler_name: str | None = None) -> None:
    """Cancels all active jobs on the cluster and cleans up stale resources.

    Raises:
      click.ClickException: If an error occurs during CHS job cancellation.
    """
    logger.debug("Cleaning all stale resources.")
    scheduler_name = scheduler_name or detect_scheduler()
    scheduler = get_scheduler(scheduler_name)
    try:
        scheduler.cancel_jobs_by_prefix("chs-")
        scheduler.cleanup_scheduler_resources()
    except (
        subprocess.CalledProcessError,
        ex.ChsSchedulerException,
        ex.KueueException,
    ) as e:
        raise ex.ChsCancellationException(
            f"CHS cancellation failed: {e!r}"
        ) from e


def safe_cancel_action(
    scheduler_name: str | None = None,
    auto_approve: bool = False,
    message: str | None = None,
    action_type: str = "cancel",
) -> None:
    """Safely cancels jobs by prompting the user for confirmation first."""
    if message:
        logger.debug(message)

    is_setup = action_type == "setup"

    if utils.is_in_cluster():
        logger.debug("Running in-cluster. Assuming auto-approve.")
        auto_approve = True

    if not auto_approve:
        if is_setup:
            prompt_msg = (
                "This will set up the cluster for a new run, which involves "
                "removing any existing CHS resources.\n"
                "Are you sure you want to proceed?"
            )
        else:
            prompt_msg = (
                "This will cancel ALL running CHS tests and delete ALL "
                "CHS resources on the cluster.\n"
                "Are you sure you want to proceed?"
            )

        if not sys.stdin.isatty():
            logger.warning(prompt_msg)
            raise click.ClickException(
                "Running in non-interactive mode.\n"
                "Please pass --auto-approve to proceed."
            )
        if not click.confirm(prompt_msg, default=False):
            logger.debug(
                f"{'Setup' if is_setup else 'Cancellation'} aborted by user. "
                "Exiting."
            )
            raise click.Abort()
    else:
        action_name = "setup" if is_setup else "cancellation"
        logger.debug(
            f"Auto-approve is enabled. Proceeding with {action_name}..."
        )

    console = Console()

    spinner_text = (
        "[bold blue]Setting up cluster environment...[/bold blue]"
        if is_setup
        else (
            "[bold red]Canceling all active CHS tests and "
            "cleaning up resources...[/bold red]"
        )
    )

    success_text = (
        f"[{theme.CHS_SUCCESS_STYLE}]Successfully prepared the cluster."
        f"[/{theme.CHS_SUCCESS_STYLE}]"
        if is_setup
        else f"[{theme.CHS_SUCCESS_STYLE}]Successfully canceled all "
        f"active CHS resources.[/{theme.CHS_SUCCESS_STYLE}]"
    )

    with theme.get_progress_spinner(spinner_text):
        logger.debug("Initiating global cleanup...")
        cancel_action(scheduler_name)
        logger.debug("Successfully canceled all active CHS resources.")
    console.print(success_text)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=(
        "Tip: You can append --help to any specific subcommand to see its "
        "exact arguments and examples (e.g., chs run --help or "
        "chs list --help)."
    ),
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Cluster Health Scanner (CHS) Tool.

    **TL;DR:** To run a standard health check on your cluster immediately,
    simply type:
        chs run
    """
    ctx.ensure_object(dict)
    ctx.obj["app_data"] = ChsAppData()
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s %(levelname)-8s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        logger.debug("Debug mode enabled.")


@cli.group(name="list", invoke_without_command=True)
@click.pass_context
def list_command(ctx: click.Context) -> bool | None:
    """Lists available resources."""
    if ctx.invoked_subcommand is None:
        return list_action()


@list_command.command(name="suites")
def list_suites() -> bool:
    """List available test suites.

    Test suites are pre-configured bundles of tests designed for specific
    use-cases, like 'lite' or 'burn-in'.
    """
    return list_action(suites=True)


@list_command.command(name="tests")
@click.option(
    "--suite",
    metavar="SUITE_NAME",
    help="Specify the suite to list tests from",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--machine-type",
    help="Filter tests by machine type",
    callback=validators.validate_non_empty_string,
)
def list_tests(suite: str | None, machine_type: str | None) -> bool:
    """List individual hardware and network benchmarks.

    For example: 'nccl-cluster', 'gemm'.
    """
    if machine_type:
        supported_mts = utils.tests.get_all_supported_machine_types()
        if machine_type not in supported_mts:
            raise click.UsageError(
                f"Unsupported machine type '{machine_type}'. "
                f"Supported types are: {', '.join(supported_mts)}"
            )

    return list_action(tests=True, suite=suite, machine_type=machine_type)


@list_command.command(name="schedulers")
def list_schedulers() -> bool:
    """List available schedulers."""
    return list_action(schedulers=True)


@list_command.command(name="machine-types")
def list_machine_types() -> bool:
    """List available machine types."""
    return list_action(machine_types=True)


@list_command.group(name="tpu")
def list_tpu() -> None:
    """List TPU-related resources."""
    pass


@list_tpu.command(name="tests")
def list_tpu_tests() -> bool:
    """List individual TPU tests."""
    return list_action(tests=True, accelerator_type="tpu")


@list_command.group(name="gpu")
def list_gpu() -> None:
    """List GPU-related resources."""
    pass


@list_gpu.command(name="tests")
def list_gpu_tests() -> bool:
    """List individual GPU tests."""
    return list_action(tests=True, accelerator_type="gpu")


def _handle_previous_run(
    app_data: ChsAppData,
    scheduler: str,
    suite: str | None,
    validation: bool,
    skip_validation: bool,
    auto_approve: bool,
) -> RunExecutionResult | None:
    """Checks for a previous run, optionally attaches to it, or cleans up."""
    try:
        latest_run_id = app_data.get_latest_run_id()
        if not latest_run_id:
            safe_cancel_action(
                scheduler, auto_approve=auto_approve, action_type="setup"
            )
            return None

        workspace_name = app_data.resolve_run_id(latest_run_id)
        run_entry = app_data.get_run(workspace_name)
        if not run_entry:
            safe_cancel_action(
                scheduler, auto_approve=auto_approve, action_type="setup"
            )
            return None

        submission_complete = run_entry.get("submission_complete", False)
        scheduler = run_entry.get("scheduler", scheduler)
        scheduler_instance = get_scheduler(scheduler)

        has_no_failed_jobs = False
        if scheduler_instance.has_no_failed_jobs():
            logger.warning("The cluster still has healthy CHS tests.")
            has_no_failed_jobs = True

        if submission_complete and has_no_failed_jobs:
            cleanup_cmd = (
                "./bin/delete_chs_gke.sh"
                if utils.is_in_cluster()
                else "chs cancel"
            )
            logger.warning(
                "Parallel runs are not supported. "
                f"Attaching to ongoing/completed run: {latest_run_id}. "
                f"Run `{cleanup_cmd}` to destroy it."
            )
            analyze_action(app_data, latest_run_id, validation, skip_validation)
            return RunExecutionResult(
                suite=suite or "lite",
                scheduler_name=scheduler,
                tests=None,
                workspace_name=workspace_name,
                workspaces_base_dir=app_data.get_workspaces_dir(),
                job_ids=None,
            )

        if not submission_complete:
            msg = (
                "Previous run died mid-submission. "
                "Cleaning all stale resources"
            )
        else:
            msg = (
                "Previous run found but some tests "
                "errored out. Cleaning all stale resources"
            )
        safe_cancel_action(
            scheduler,
            auto_approve=auto_approve,
            message=msg,
            action_type="setup",
        )
    except ex.NoActiveJobsException:
        safe_cancel_action(
            scheduler,
            auto_approve=auto_approve,
            message=(
                "Previous run found but tests no longer "
                "exist on the cluster. Cleaning all "
                "stale resources."
            ),
            action_type="setup",
        )
    except (ex.NoPreviousRunsFoundException, ex.WorkspaceNotFoundException):
        safe_cancel_action(
            scheduler, auto_approve=auto_approve, action_type="setup"
        )

    return None


@cli.command()
@handle_chs_exceptions
@click.option(
    "--suite",
    "-s",
    help=(
        "Name of the test suite to run (pre-configured bundles of tests "
        "designed for specific use-cases, like 'lite' or 'burn-in')"
    ),
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--tests",
    "-t",
    help=(
        "Comma-separated list of individual hardware and network "
        "benchmarks (e.g., 'nccl-cluster', 'gemm')"
    ),
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--nodes",
    help="Comma-separated list of individual nodes to run on",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--nodelists",
    help="Slurm nodelists to run on (e.g., node-[1-3],node-[4-5])",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--nodepools",
    help="Kubernetes node pools to run on",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--partitions",
    help="Slurm partitions to run on",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Set up the workspace for the detected scheduler but do not execute.",
)
@click.option(
    "--auto-approve",
    "-y",
    is_flag=True,
    help="Automatically confirm execution preview without prompting.",
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
    tests: str | None,
    nodes: str | None,
    nodelists: str | None,
    nodepools: str | None,
    partitions: str | None,
    dry_run: bool = False,
    auto_approve: bool = False,
    validation: str | None = None,
    skip_validation: bool = False,
    run_id: str | None = None,
) -> RunExecutionResult:
    """Command to run a test suite.

    Args:
      suite: Name of the test suite to run.
      tests: Comma-separated list of specific tests to run (optional).
      nodes: Comma-separated list of individual nodes to run on.
      nodelists: Slurm node list pattern to run on (e.g., node-[1-3]).
      nodepools: Kubernetes node pools to run on.
      partitions: Slurm partitions to run on.
      dry_run: Set up the workspace but do not execute experiments or analyze.

    Returns:
      A RunExecutionResult object containing details of the run.
    """
    app_data = obj["app_data"]

    with theme.get_progress_spinner(
        "Discovering hardware and validating arguments..."
    ):
        scheduler = detect_scheduler()

        scheduler_obj = get_scheduler(
            scheduler,
            nodes=nodes,
            nodelists=nodelists,
            nodepools=nodepools,
            partitions=partitions,
        )

        nodes_under_test = scheduler_obj.get_nodes_under_test()
        if not nodes_under_test:
            raise ex.NoNodesProvidedException("No nodes provided to run on.")

        preflights.run_platform_preflights()

        detected_machine_types = scheduler_obj.detect_nodes_hardware(
            nodes_under_test
        )

        if not detected_machine_types or "unknown" in detected_machine_types:
            raise ex.MachineTypeDiscoveryException(
                "Failed to detect machine type of some nodes"
            )

        _validate_run_args(
            scheduler, detected_machine_types, tests, suite, validation
        )

        nodes_under_test = scheduler_obj.filter_partial_tpu_slices(
            nodes_under_test
        )
        if not nodes_under_test:
            raise ex.NoNodesProvidedException(
                "No valid nodes remain after TPU slice validation."
            )

        machine_type = detected_machine_types[0]

        timeout = resources.DEFAULT_TIMEOUT
        deadline = resources.DEFAULT_DEADLINE
        is_tiered = False
        active_tier_tests = None

        if not tests:
            if not suite:
                suite = "lite"

            suite_cfg = load_suite(suite)
            is_tiered = is_suite_tiered(suite_cfg)

            supported_tests = (
                utils.tests.get_tests_by_machine_type_and_scheduler(
                    machine_type, scheduler
                )
            )

            resolved_tests = [
                t if isinstance(t, dict) else {"name": t}
                for t in suite_cfg.get("tests", [])
                if _get_test_name(t) in supported_tests
            ]

            if is_tiered:
                tier_manager = TierManager(suite_cfg.get("tests", []))
                all_tests = tier_manager.get_all_tests()
                if not all_tests:
                    raise ex.ChsValidationError(
                        f"Suite '{suite}' has no tests."
                    )

                initial_tests = tier_manager.get_next_tier_tests({})
                if not initial_tests:
                    raise ex.ChsValidationError(
                        f"Suite '{suite}' has no initial tier tests."
                    )

                active_tier_tests = [
                    t for t in initial_tests if t in supported_tests
                ]
                if not active_tier_tests:
                    raise ex.ChsValidationError(
                        f"No supported initial health checks found in suite"
                        f" '{suite}' for scheduler '{scheduler}' and "
                        f"machine type '{machine_type}'."
                    )
            else:
                active_tier_tests = [_get_test_name(t) for t in resolved_tests]

            if not resolved_tests:
                raise ex.ChsValidationError(
                    f"No supported tests found in suite '{suite}' for "
                    f"scheduler '{scheduler}' and machine type"
                    f" '{machine_type}'."
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

            test_names = [t.get("name") for t in resolved_tests]
            logger.debug(
                f"Loaded {len(resolved_tests)} tests ({', '.join(test_names)}) "
                f"for suite '{suite}' from suites.yaml."
            )
            parsed_tests = resolved_tests
        else:
            parsed_tests_strs = _parse_str_to_list(tests)
            parsed_tests = (
                [{"name": t} for t in parsed_tests_strs]
                if parsed_tests_strs
                else []
            )
            if not suite:
                suite = "custom"

        logger.debug("Running preflight checks")
        parsed_tests = scheduler_obj.run_preflight_checks(parsed_tests)
        if active_tier_tests and parsed_tests is not None:
            remaining_names = {_get_test_name(t) for t in parsed_tests}
            active_tier_tests = [
                t for t in active_tier_tests if t in remaining_names
            ]

    Console().print(
        f"[{theme.CHS_SUCCESS_STYLE}]Validation and hardware discovery "
        f"completed.[/{theme.CHS_SUCCESS_STYLE}]"
    )

    if not dry_run:
        attach_result = _handle_previous_run(
            app_data=app_data,
            scheduler=scheduler,
            suite=suite,
            validation=validation,
            skip_validation=skip_validation,
            auto_approve=auto_approve,
        )
        if attach_result:
            return attach_result
    cluster_name = utils.get_sanitized_cluster_name(scheduler_obj)

    test_names = ", ".join(
        t.get("name") if isinstance(t, dict) else str(t) for t in parsed_tests
    )
    preview_title = (
        "CHS Dry Run Preview (No tests will be submitted):"
        if dry_run
        else "CHS Execution Preview:"
    )
    console = Console()
    console.print(f"[bold]{preview_title}[/bold]")
    console.print(
        f"  Target Cluster:    [{theme.CHS_HEADER_STYLE}]{cluster_name}"
        f"[/{theme.CHS_HEADER_STYLE}]"
    )
    console.print(
        f"  Machine Type:      [{theme.CHS_HEADER_STYLE}]{machine_type}"
        f"[/{theme.CHS_HEADER_STYLE}]"
    )
    console.print(
        f"  Scheduler:         [{theme.CHS_HEADER_STYLE}]{scheduler}"
        f"[/{theme.CHS_HEADER_STYLE}]"
    )
    if nodes_under_test:
        scope = (
            f"[{theme.CHS_HEADER_STYLE}]{len(nodes_under_test)}"
            f"[/{theme.CHS_HEADER_STYLE}] "
        )
    else:
        scope = "all available nodes"
    console.print(f"  Nodes under test:  {scope}")
    console.print(
        f"  Tests Scheduled:   [{theme.CHS_HEADER_STYLE}]{test_names}"
        f"[/{theme.CHS_HEADER_STYLE}]"
    )

    if utils.is_in_cluster():
        logger.debug("Running in-cluster. Assuming auto-approve.")
        auto_approve = True

    if not auto_approve:
        prompt_msg = (
            "Proceed with dry run workspace generation?"
            if dry_run
            else "Proceed with execution?"
        )
        if not sys.stdin.isatty():
            logger.debug(prompt_msg)
            raise click.ClickException(
                "Running in non-interactive mode.\n"
                "Please pass --auto-approve to proceed."
            )
        if not click.confirm(prompt_msg, default=True):
            logger.debug("Run aborted by user.")
            return None
    else:
        logger.debug("--auto-approve flag detected. Bypassing confirmation.")

    short_hash = utils.generate_random_suffix()
    if not run_id:
        run_id = f"{cluster_name}-{suite}"
    run_id = f"{run_id}-{short_hash}"

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    workspace_name = f"{timestamp}_{run_id}"
    workspaces_base_dir = app_data.get_workspaces_dir()

    logger.debug("run_action: Parsed tests: %r", parsed_tests)
    logger.debug("Workspace name: %s", workspace_name)
    console.print(
        f"  Run ID:            [{theme.CHS_WARNING_STYLE}]{run_id}"
        f"[/{theme.CHS_WARNING_STYLE}]"
    )
    logger.debug("Workspaces base directory: %s", workspaces_base_dir)

    # Normalize command statement via an environment override (e.g. for Docker),
    # or fallback to local execution logic.
    cli_name = os.environ.get("CHS_CLI_NAME")
    if not cli_name:
        is_module_run = os.path.basename(sys.argv[0]) == "__main__.py"
        cli_name = "python -m chs" if is_module_run else "chs"

    # Safely quote the arguments, but leave the base CLI name unquoted
    args_string = shlex.join(sys.argv[1:])
    command_statement = f"{cli_name} {args_string}".strip()

    app_data.update_run(
        workspace_name,
        {
            "run_id": run_id,
            "workspace_name": workspace_name,
            "workspace_path": os.path.join(workspaces_base_dir, workspace_name),
            "submission_time": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "scheduler": scheduler_obj.name,
            "command_statement": command_statement,
            "storage": app_data.get_storage_config(),
            "node_to_instance_id": scheduler_obj.node_to_instance_id,
            "deadline": deadline,
            "tests": parsed_tests,
            "suite": suite,
            "is_tiered": is_tiered,
            "active_tier": 1,
            "active_tier_tests": active_tier_tests,
            "submission_complete": False,
        },
    )

    if dry_run:
        status_msg = (
            f"[{theme.CHS_HEADER_STYLE}]Setting up cluster execution "
            f"environment...[/{theme.CHS_HEADER_STYLE}]"
        )
    else:
        if is_tiered:
            tests_str = ", ".join(active_tier_tests)
            status_msg = (
                f"[{theme.CHS_SUCCESS_STYLE}]Preparing environment and "
                f"triggering initial health checks: {tests_str}..."
                f"[/{theme.CHS_SUCCESS_STYLE}]"
            )
        else:
            status_msg = (
                f"[{theme.CHS_SUCCESS_STYLE}]Preparing execution environment "
                f"and submitting tests...[/{theme.CHS_SUCCESS_STYLE}]"
            )
    with cancel_jobs_on_failure(scheduler_obj, "chs-", active=not dry_run):
        with theme.get_progress_spinner(status_msg):
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
                active_tests=active_tier_tests if is_tiered else None,
            )

        if not job_ids and not dry_run:
            raise ex.JobSubmissionFailureException(
                f"Job submission failed for workspace: {workspace_name}. "
                "Please check internal.log for details."
            )

    result = RunExecutionResult(
        suite=suite,
        scheduler_name=scheduler_obj.name,
        tests=parsed_tests,
        workspace_name=workspace_name,
        workspaces_base_dir=workspaces_base_dir,
        job_ids=job_ids,
    )

    if dry_run:
        logger.debug("Dry run completed. Workspace setup is ready.")
        return result

    logger.debug("Run submitted successfully.")
    logger.debug("Job IDs: %s", ", ".join(job_ids))

    if is_tiered:
        console.print(
            f"[{theme.CHS_SUCCESS_STYLE}]Successfully submitted "
            f"initial health check tests.[/{theme.CHS_SUCCESS_STYLE}]"
        )
    else:
        console.print(
            f"[{theme.CHS_SUCCESS_STYLE}]Successfully submitted tests."
            f"[/{theme.CHS_SUCCESS_STYLE}]"
        )

    # Automatically trigger analysis
    analyze_action(
        app_data,
        run_id,
        validation=validation,
        skip_validation=skip_validation,
    )

    return result


def _get_run_id_or_prompt(
    app_data: ChsAppData,
    provided_run_id: str | None,
    prompt_msg: str = "Which run would you like to view results for?",
) -> str:
    """Gets the run ID either from user input or an interactive prompt."""
    if provided_run_id:
        return provided_run_id

    if not sys.stdin.isatty() or utils.is_in_cluster():
        run_id = app_data.get_latest_run_id()
        logger.debug(
            "No run_id provided in non-interactive mode. Using latest run: %s",
            run_id,
        )
        return run_id

    runs = app_data.get_latest_runs(5)
    if not runs:
        raise ex.NoPreviousRunsFoundException(
            "No previous runs found in workspaces directory."
        )

    choices = []
    for ws_name, details in runs.items():
        run_id = details.get("run_id", ws_name)
        started_str = utils.formatting.format_relative_time(
            details.get("submission_time", "")
        )
        label = f"{run_id:<30} (Started {started_str})"
        choices.append(questionary.Choice(title=label, value=run_id))

    choices.append(questionary.Choice(title="Exit", value=None))

    custom_style = questionary.Style(
        [
            ("question", "fg:cyan bold"),
            ("pointer", "fg:cyan bold"),
            ("highlighted", "fg:cyan bold"),
        ]
    )

    selected = questionary.select(
        f"{prompt_msg} (Use arrow keys)",
        choices=choices,
        style=custom_style,
    ).ask()

    if not selected or selected == "Exit":
        raise click.Abort()

    return selected


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

    If no run_id is provided, you will be prompted to select one.

    \b
    Args:
      obj: Click object containing app_data.
      run_id: The ID of the test run to analyze.
      thresholds: Path to user-defined thresholds YAML file.

    Returns:
      The run_id of the analyzed test run.
    """
    app_data = obj["app_data"]
    run_id = _get_run_id_or_prompt(
        app_data, run_id, prompt_msg="Which run would you like to analyze?"
    )
    logger.debug("Analyzing run: %s", run_id)

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
@click.option(
    "--auto-approve",
    "-y",
    is_flag=True,
    help="Automatically confirm cancellation without prompting.",
)
@click.pass_obj
def cancel(obj, auto_approve: bool = False) -> None:
    """Cancel all running CHS jobs and remove stale resources."""
    app_data = obj["app_data"]
    safe_cancel_action(auto_approve=auto_approve)
    try:
        latest_run_id = app_data.get_latest_run_id()
        if latest_run_id:
            workspace_name = app_data.resolve_run_id(latest_run_id)
            workspaces_base_dir = app_data.get_workspaces_dir()
            ramble_manager = RambleManager(
                workspaces_base_dir, workspace_name, run_id=latest_run_id
            )
            if os.path.exists(ramble_manager.path):
                logger.debug(
                    "Archiving the latest workspace after cancellation..."
                )
                _archive_and_log(
                    ramble_manager,
                    "[bold blue]Archiving the latest workspace and saving "
                    "Bill of Health...[/bold blue]",
                )
    except (ex.NoPreviousRunsFoundException, ex.WorkspaceNotFoundException):
        pass
    except (OSError, ex.ChsException) as e:
        logger.warning("Failed to archive workspace after cancellation: %s", e)


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
    "--details",
    is_flag=True,
    help="Print raw metric tables.",
)
@click.pass_obj
def results(
    obj,
    run_id: str | None,
    tests: str | None,
    resources: str | None,
    details: bool,
) -> None:
    """Show Bill of Health results for a run.

    If no run_id is provided, you will be prompted to select one.
    """
    app_data = obj["app_data"]
    run_id = _get_run_id_or_prompt(app_data, run_id)
    _show_results(app_data, run_id, tests, resources, details=details)


@cli.command()
@click.pass_obj
def history(obj) -> None:
    """Show the history of last 5 runs."""
    app_data = obj["app_data"]
    runs = app_data.get_latest_runs(5)

    if not runs:
        click.echo("No runs found.")
        return

    console = Console()
    table = utils.create_styled_table()
    table.add_column("Run ID", overflow="fold")
    table.add_column("Submitted", no_wrap=True)
    table.add_column("Storage", overflow="fold")
    table.add_column("Age", no_wrap=True)
    table.add_column("Command", overflow="fold")

    for run in runs.values():
        run_id = run.get("run_id", "N/A")

        command_statement = run.get("command_statement", "")
        if not command_statement:
            command_statement = "chs run"

        submission_complete = run.get("submission_complete", False)
        sub_text = (
            f"[{theme.CHS_SUCCESS_STYLE}]Y[/{theme.CHS_SUCCESS_STYLE}]"
            if submission_complete
            else f"[{theme.CHS_FAILURE_STYLE}]N[/{theme.CHS_FAILURE_STYLE}]"
        )

        storage_str = utils.formatting.format_storage_path(
            run.get("storage", {})
        )
        age_text = utils.formatting.format_relative_time(
            run.get("submission_time", "")
        )

        table.add_row(
            run_id,
            sub_text,
            storage_str,
            age_text,
            command_statement,
        )

    console.print(table)


def _validate_and_save_storage(app_data: ChsAppData, config: dict) -> None:
    """Helper to validate storage config and save if successful."""
    try:
        provider = StorageProviderFactory.get_provider(config)
        click.echo("Validating storage access...")
        provider.validate()
        app_data.set_storage_config(config)
        click.echo("Storage configured and validated successfully.")
    except Exception as e:
        raise click.ClickException(f"Storage validation failed: {e}")


def _interactive_configure_storage(app_data: ChsAppData) -> None:
    """Guided flow for storage configuration."""
    click.echo("Select storage type:")
    click.echo("  [1] GCS")
    click.echo("  [2] Path")
    choice = click.prompt("Choice", type=int, default=1)

    if choice == 1:
        bucket_name = click.prompt("Enter GCS bucket name")
        if bucket_name.lower().startswith("gs://"):
            bucket_name = bucket_name[5:]
        config = {"type": "gcs", "gcs": {"bucket_name": bucket_name}}
    elif choice == 2:
        path = click.prompt("Enter target directory path (absolute)")
        config = {"type": "path", "path": {"path": path}}
    else:
        raise click.ClickException("Invalid choice.")

    _validate_and_save_storage(app_data, config)


@cli.group()
def configure() -> None:
    """Configure CHS settings."""
    pass


@configure.group("storage", invoke_without_command=True)
@click.pass_context
def configure_storage(ctx: click.Context) -> None:
    """Configure storage settings."""
    if ctx.invoked_subcommand is None:
        app_data = ctx.obj["app_data"]
        _interactive_configure_storage(app_data)


@configure_storage.command("gcs")
@click.argument("bucket_name", required=False)
@click.pass_context
def configure_gcs(ctx: click.Context, bucket_name: str | None) -> None:
    """Configure GCS storage."""
    app_data = ctx.obj["app_data"]
    if not bucket_name:
        bucket_name = click.prompt("Enter GCS bucket name")

    if bucket_name.lower().startswith("gs://"):
        bucket_name = bucket_name[5:]

    config = {"type": "gcs", "gcs": {"bucket_name": bucket_name}}
    _validate_and_save_storage(app_data, config)


@configure_storage.command("path")
@click.argument("path", required=False)
@click.pass_context
def configure_path(ctx: click.Context, path: str | None) -> None:
    """Configure Local Path storage."""
    app_data = ctx.obj["app_data"]
    if not path:
        path = click.prompt("Enter target directory path (absolute)")

    config = {"type": "path", "path": {"path": path}}
    _validate_and_save_storage(app_data, config)


@configure_storage.command("show")
@click.pass_context
def show_storage(ctx: click.Context) -> None:
    """Show currently configured storage."""
    app_data = ctx.obj["app_data"]
    config = app_data.get_storage_config()
    if not config:
        click.echo("No storage configured.")
    else:
        path = utils.formatting.format_storage_path(config)
        click.echo(f"Configured storage: {path}")


@configure_storage.command("remove")
@click.pass_context
def remove_storage(ctx: click.Context) -> None:
    """Remove configured storage."""
    app_data = ctx.obj["app_data"]
    app_data.remove_storage_config()
    click.echo("Storage configuration removed.")


def main() -> None:
    """Main entry point for the CHS CLI."""

    logger.debug("CHS CLI starting...")
    try:
        cli(obj={})
    finally:
        logger.debug("CHS CLI finished.")
