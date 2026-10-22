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
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import warnings
from contextlib import contextmanager
from typing import Any

import click
import questionary
from rich.console import Console
from rich.text import Text

from chs import (
    BaseScheduler,
    ChsAppData,
    RambleManager,
    ResultsViewer,
    detect_scheduler,
    exceptions as ex,
    get_scheduler,
    is_suite_tiered,
    load_suite,
    preflights,
    resources,
    telemetry,
    utils,
    validators,
)
from chs.boh_generator import parse_tier_results_from_boh
from chs.cli import help_command_text
from chs.cli import list_formatting
from chs.cli.background_cleanup import BackgroundCleanup
from chs.storage import StorageProviderFactory
from chs.telemetry import reporting as telemetry_reporting
from chs.tier_manager import TestStatus, TierManager
from chs.utils import theme
from chs.utils.k8s_client import KubeError
from chs.version import get_version

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="google_crc32c"
)

logger = logging.getLogger(__name__)

_CANCELLATION_CLUSTER_ERRORS = (
    subprocess.CalledProcessError,
    KubeError,
    ex.ChsSchedulerException,
)


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
        except KeyboardInterrupt:
            raise click.exceptions.Exit(130)

    return wrapper


@contextmanager
def _setup_workspace_logger(workspace_path: str):
    """Configures a file handler to save all DEBUG logs to the workspace."""
    debug_log_path = os.path.join(workspace_path, "debug.txt")
    root_logger = logging.getLogger()
    if any(
        getattr(h, "baseFilename", "") == debug_log_path
        for h in root_logger.handlers
    ):
        yield
        return

    os.makedirs(workspace_path, exist_ok=True)
    file_handler = logging.FileHandler(debug_log_path)

    original_root_level = root_logger.level
    modified_handlers = []

    try:
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
                    modified_handlers.append(handler)

        yield
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()
        root_logger.setLevel(original_root_level)
        for handler in modified_handlers:
            handler.setLevel(logging.NOTSET)


def _get_href_link(path_or_uri: str) -> str:
    href = path_or_uri
    if path_or_uri.startswith("gs://"):
        href = path_or_uri.replace(
            "gs://", "https://storage.cloud.google.com/", 1
        )
    return href


def _make_clickable_link(path_or_uri: str) -> str:
    href = _get_href_link(path_or_uri)
    return f"[link={href}]{path_or_uri}[/link]"


def _archive_and_log(
    ramble_manager: RambleManager, spinner_label: str | None = None
) -> None:
    """Archives the workspace and logs the results with a progress spinner."""
    console = Console()
    if not spinner_label:
        if ramble_manager.has_bill_of_health():
            spinner_label = (
                "[bold blue]Archiving workspace and saving "
                "Bill of Health...[/bold blue]"
            )
        else:
            spinner_label = "[bold blue]Archiving workspace...[/bold blue]"

    with theme.get_progress_spinner(spinner_label):
        ws_path, boh_json_path, boh_html_path, all_uploads_succeeded = (
            ramble_manager.archive()
        )
    storage_provider = StorageProviderFactory.get_provider_for_run(
        ramble_manager.workspace_name
    )
    updates = {}
    if boh_html_path and storage_provider:
        console.print(
            f"[{theme.CHS_SUCCESS_STYLE}]Interactive health report saved at: "
            f"{_make_clickable_link(boh_html_path)}[/{theme.CHS_SUCCESS_STYLE}]"
        )
        updates["boh_html_uri"] = boh_html_path
    if boh_json_path and storage_provider:
        console.print(
            f"[{theme.CHS_SUCCESS_STYLE}]Bill of Health saved at: "
            f"{_make_clickable_link(boh_json_path)}[/{theme.CHS_SUCCESS_STYLE}]"
        )
        updates["boh_uri"] = boh_json_path
    if ws_path and storage_provider:
        logger.debug(f"Workspace archived to: {_get_href_link(ws_path)}")
        updates["workspace_uri"] = ws_path

    if not all_uploads_succeeded:
        console.print(
            f"[{theme.CHS_FAILURE_STYLE}]Upload to GCS failed for some files, "
            f"check the local paths above.[/{theme.CHS_FAILURE_STYLE}]"
        )
        console.print(
            f"[{theme.CHS_NEUTRAL_STYLE}]Tip:  If mounted, Check your "
            f"docker host volume mount to access the local files for the "
            f"same.[/{theme.CHS_NEUTRAL_STYLE}]"
        )

    if storage_provider and all_uploads_succeeded:
        updates["is_archived"] = True
        ramble_manager._app_data.update_run(
            ramble_manager.workspace_name, updates
        )


@contextmanager
def archive_workspace(
    ramble_manager: RambleManager, *, on_success: bool = True
):
    """Context manager to ensure workspace is archived even in error cases."""
    try:
        yield
    except KeyboardInterrupt:
        raise
    except BaseException:
        if ramble_manager.is_workspace_created:
            try:
                _archive_and_log(ramble_manager)
            except Exception as archive_err:
                logger.debug(
                    "Failed to archive the workspace and upload the "
                    "health report during cleanup. Error: %s",
                    archive_err,
                )
                logger.warning(
                    "Failed to archive the workspace and upload "
                    "the Bill of Health report."
                )
        raise
    else:
        if on_success and ramble_manager.is_workspace_created:
            _archive_and_log(ramble_manager)


@contextmanager
def cancel_jobs_on_failure(scheduler_obj: BaseScheduler, run_id: str):
    """Context manager to ensure the run's jobs are cancelled on failure."""
    try:
        yield
    except BaseException as e:
        is_interrupt = isinstance(e, KeyboardInterrupt)
        spinner_label = (
            "[bold red]Interrupt received. Cancelling run and cleaning up "
            "CHS resources...[/bold red]"
            if is_interrupt
            else "[bold red]Run failed. Cleaning up CHS resources...[/bold red]"
        )

        cleaned_up = False
        with (
            _handle_termination_signals(_continue_cleanup),
            theme.get_progress_spinner(spinner_label),
        ):
            try:
                scheduler_obj.cancel_run(run_id)
                cleaned_up = True
            except KeyboardInterrupt:
                Console().print(
                    "[bold red]WARNING: Cleanup interrupted. "
                    f"Please run `chs cancel {run_id}` to finish cleaning up."
                    "[/bold red]"
                )
                raise
            except Exception as cleanup_err:
                # We catch broad Exception here as a fallback to ensure we
                # don't hide the original exception that triggered cleanup.
                logger.error(
                    "Failed to clean CHS resources. Please run "
                    "`chs cancel %s`. Error: %s",
                    run_id,
                    cleanup_err,
                )
                Console().print(
                    "\n[bold red]WARNING: Automatic cleanup failed. "
                    "You may have resources from lingering tests on the "
                    f"cluster. Please run `chs cancel {run_id}` to manually "
                    "clean up."
                    "[/bold red]"
                )

        if is_interrupt and cleaned_up:
            Console().print(
                f"[{theme.CHS_NEUTRAL_STYLE}]CHS cleanup finished."
                f"[/{theme.CHS_NEUTRAL_STYLE}]"
            )
        raise


def _is_regular_file(fd: int) -> bool:
    try:
        return stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError:
        return False


def _discard_output_after_hangup() -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    for fd in (1, 2):
        if not _is_regular_file(fd):
            os.dup2(devnull, fd)
    if devnull > 2:
        os.close(devnull)


def _continue_cleanup(signum, frame) -> None:
    if signum == signal.SIGHUP:
        _discard_output_after_hangup()


def _interrupt_run(signum, frame) -> None:
    for termination_signal in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(termination_signal) is _interrupt_run:
            signal.signal(termination_signal, _continue_cleanup)
    _continue_cleanup(signum, frame)
    raise KeyboardInterrupt


@contextmanager
def _handle_termination_signals(handler):
    """Handles SIGTERM and SIGHUP with handler unless they are ignored."""
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGHUP):
            if signal.getsignal(signum) in (signal.SIG_DFL, _interrupt_run):
                previous_handlers[signum] = signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


def _submit_next_tier(
    app_data: ChsAppData,
    ramble_manager: RambleManager,
    run_entry: dict[str, Any],
    workspace_name: str,
    next_tier: int,
    next_tests: list[str],
    tier_results: dict[str, Any] | None = None,
) -> None:
    """Submits the next tier of diagnostic tests."""
    has_error = False
    if tier_results:
        has_error = any(s == TestStatus.ERROR for s in tier_results.values())

    reason_msg = (
        "Test execution error or timeout detected."
        if has_error
        else "Degraded performance detected."
    )
    Console().print(
        f"[{theme.CHS_WARNING_STYLE}]{reason_msg} "
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
        "machine_types=%r, suite=%r, machine_type=%r, accelerator_type=%r",
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

    if not accelerator_type:
        console.print(
            Text(
                "(Hint: Append the `--tpu` or `--gpu` flags to easily "
                "filter results by accelerator)",
                style="dim",
            )
        )
        console.print()

    if machine_types or show_all:
        console.print(
            list_formatting.get_machine_types_renderable(accelerator_type)
        )
        console.print()

    if tests or show_all:
        for r in list_formatting.get_tests_renderables(
            suite, tests, machine_type, accelerator_type
        ):
            console.print(r)
        console.print()

    if suites or show_all:
        console.print(list_formatting.get_suites_renderable(accelerator_type))
        console.print()

    if schedulers or show_all:
        console.print(list_formatting.get_schedulers_renderable())
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


def _validate_cli_args(
    suite: str | None,
    tests: str | None,
    nodes: str | None = None,
    nodelists: str | None = None,
    nodepools: str | None = None,
    partitions: str | None = None,
) -> None:
    """Validates core CLI arguments (suite) before scheduler
    detection.

    Args:
      suite: Suite name, or None.
      tests: Comma-separated list (or multiple flags) of tests, or None.
      nodes: Comma-separated list (or multiple flags) of individual nodes,
             or None.
      nodelists: Slurm node list pattern, or None.
      nodepools: Kubernetes node pools, or None.
      partitions: Slurm partitions, or None.

    Raises:
      ex.ChsValidationError: If validation fails.
    """
    cli_validator = validators.CliArgsValidator()
    errors = cli_validator.validate(
        {
            "tests": _parse_str_to_list(tests),
            "suite": suite,
            "nodes": nodes,
            "nodelists": nodelists,
            "nodepools": nodepools,
            "partitions": partitions,
        }
    )
    if errors:
        raise ex.ChsValidationError("\n".join(errors))


def _validate_storage(app_data: ChsAppData) -> None:
    """Validates that GCS storage is configured."""
    storage_config = app_data.get_storage_config()
    if not storage_config or storage_config.get("type") != "gcs":
        raise ex.ChsValidationError(
            "GCS storage must be configured to run CHS. "
            "Please configure GCS using 'chs configure storage gcs'."
        )


def _validate_run_args(
    scheduler: str,
    scheduler_obj: BaseScheduler,
    detected_machine_types: list[str],
    tests: str | None,
) -> None:
    """Validates core run arguments against supported lists.

    Args:
      scheduler: Name of the scheduler.
      scheduler_obj: Resolved scheduler instance.
      detected_machine_types: List of detected machine types.
      tests: Comma-separated list (or multiple flags) of tests, or None.

    Raises:
      ex.ChsValidationError: If validation fails.
    """
    if not detected_machine_types:
        raise ex.ChsValidationError(
            "No valid machine types detected on cluster nodes."
        )

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
            "scheduler_obj": scheduler_obj,
            "machine_type": machine_type,
            "tests": _parse_str_to_list(tests),
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
    deadline: int | None = None,
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
      deadline: The deadline for the job in seconds.

    Returns:
      A list of submitted job IDs.

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

    with _setup_workspace_logger(ramble_manager.path):
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
                        run_id=run_id,
                    )
                    scheduler_obj.apply_scheduler_resources()

                ramble_manager.create_workspace()

                ramble_manager.add_custom_repository()
                ramble_manager.configure_workspace(
                    scheduler_obj,
                    tests,
                    machine_type=machine_type,
                    deadline=deadline,
                )
                ramble_manager.setup_workspace()

                job_ids = ramble_manager.execute_experiments(
                    tier=1, active_tests=active_tests
                )
                if not job_ids:
                    raise ex.JobSubmissionFailureException(
                        f"Job submission failed for "
                        f"workspace: {workspace_name}. "
                        "Please check internal.log in the workspace directory "
                        "for details."
                    )
                return job_ids
            except (OSError, subprocess.CalledProcessError, KubeError) as e:
                raise ex.ChsExecutionException(
                    f"CHS execution failed: {ex.format_subprocess_error(e)}"
                ) from e


def _analyze_action(
    app_data: ChsAppData,
    run_id: str,
) -> str:
    """Analyzes results from a specific test run.

    Args:
      app_data: ChsAppData instance.
      run_id: The ID of the test run to analyze.

    Returns:
      The run_id of the analyzed test run.

    Raises:
      click.ClickException: If an error occurs during CHS analysis.
    """
    logger.debug("Analyzing run_id=%r", run_id)
    workspaces_base_dir = app_data.get_workspaces_dir()
    logger.debug("Using workspaces_base_dir: %s", workspaces_base_dir)
    workspace_name = app_data.resolve_run_id(run_id)
    telemetry.set_telemetry_run_id(workspace_name)
    ramble_manager = RambleManager(
        workspaces_base_dir, workspace_name, run_id=run_id
    )

    with _setup_workspace_logger(ramble_manager.path):
        if not os.path.exists(ramble_manager.path):
            raise ex.WorkspaceNotFoundException(
                f"Workspace '{run_id}' not found."
            )

        workspace_path = pathlib.Path(ramble_manager.path)
        boh_path = workspace_path / "bill_of_health.json"
        results_json_path = workspace_path / "results.latest.json"

        run_entry = app_data.get_run(workspace_name) or {}
        is_tiered = run_entry.get("is_tiered", False)
        is_boh_final = run_entry.get("is_boh_final", not is_tiered)

        # 1. Check if Bill of Health already exists and workspace is archived
        if boh_path.exists() and is_boh_final and ramble_manager.is_archived():
            logger.debug(
                "Bill of Health already exists and workspace is archived. "
                "Showing results directly."
            )
            Console().print(
                f"[{theme.CHS_SUCCESS_STYLE}]Found existing "
                f"Bill of Health for {run_id}."
                f"[/{theme.CHS_SUCCESS_STYLE}]"
            )
            _show_results(app_data, run_id)
            return run_id

        with (
            BackgroundCleanup(run_id) as background_cleanup,
            archive_workspace(ramble_manager, on_success=True),
        ):
            try:
                # 2. If results.latest.json exists (but no BoH), generate BoH
                if results_json_path.exists() and is_boh_final:
                    logger.debug(
                        "Found results from this run. Generating Bill "
                        "of Health..."
                    )
                    with theme.get_progress_spinner(
                        "Generating Bill of Health..."
                    ):
                        ramble_manager.generate_boh(
                            str(results_json_path),
                        )
                        background_cleanup.start(ramble_manager)
                    Console().print(
                        f"[{theme.CHS_SUCCESS_STYLE}]Bill of Health generated."
                        f"[/{theme.CHS_SUCCESS_STYLE}]"
                    )
                    _show_results(app_data, run_id, suppress_boh_uri=True)
                    return run_id

                # 3. If neither exists, wait for jobs and then analyze
                suite_name = run_entry.get("suite")
                suite_cfg = load_suite(suite_name) if is_tiered else {}
                tier_manager = (
                    TierManager(suite_cfg.get("tests", []))
                    if is_tiered
                    else None
                )

                while not run_entry.get("is_boh_final", False):
                    active_tier = run_entry.get("active_tier", 1)
                    active_tier_tests_raw = (
                        run_entry.get("active_tier_tests") or []
                    )
                    active_tier_tests = set(
                        t.get("name") if isinstance(t, dict) else t
                        for t in active_tier_tests_raw
                    )

                    if tier_manager:
                        tier_manager.fast_forward_to_tier(active_tier)

                    logger.debug("Waiting for tier %s jobs...", active_tier)
                    ramble_manager.wait_for_jobs(
                        app_data=app_data,
                        tier=active_tier if is_tiered else None,
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
                        )
                    Console().print(
                        f"[{theme.CHS_SUCCESS_STYLE}]Successfully "
                        f"analyzed logs.[/{theme.CHS_SUCCESS_STYLE}]"
                    )

                    next_tests = None
                    if tier_manager:
                        boh_file = os.path.join(boh_dir, "bill_of_health.json")
                        tier_results, hard_failures_present = (
                            parse_tier_results_from_boh(
                                boh_file, active_tier_tests
                            )
                        )
                        if hard_failures_present:
                            Console().print(
                                f"[{theme.CHS_WARNING_STYLE}]Hard failure "
                                "detected. Halting follow-up diagnostic "
                                "tiers."
                                f"[/{theme.CHS_WARNING_STYLE}]"
                            )
                            next_tests = None
                        else:
                            next_tests = tier_manager.get_next_tier_tests(
                                tier_results
                            )

                    if not next_tests:
                        run_entry["is_boh_final"] = True
                        app_data.update_run(
                            workspace_name, {"is_boh_final": True}
                        )
                        telemetry_reporting.record_test_duration(
                            run_entry.get("submission_time")
                        )
                        Console().print(
                            f"[{theme.CHS_SUCCESS_STYLE}]All checks "
                            "completed. Bill of Health generated."
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
                        tier_results=tier_results,
                    )

                background_cleanup.start(ramble_manager)
                _show_results(app_data, run_id, suppress_boh_uri=True)
            except (OSError, subprocess.CalledProcessError, KubeError) as e:
                err_msg = ex.format_subprocess_error(e)
                raise ex.ChsAnalysisException(
                    f"CHS analysis failed for run_id={run_id}: {err_msg}"
                ) from e

            else:
                return run_id


def _show_results(
    app_data: ChsAppData,
    run_id: str | None = None,
    suppress_boh_uri: bool = False,
) -> None:
    """Shows the Bill of Health summary."""
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
    else:
        telemetry.set_telemetry_run_id(workspace_name)

    boh_path = workspaces_dir / workspace_name / "bill_of_health.json"
    run_provider = StorageProviderFactory.get_provider_for_run(workspace_name)

    boh_uri = None
    if not suppress_boh_uri:
        run_data = app_data.get_run(workspace_name)
        if isinstance(run_data, dict):
            boh_uri = run_data.get("boh_html_uri") or run_data.get("boh_uri")
        if not boh_uri and run_provider:
            html_remote = f"bills_of_health/{workspace_name}.html"
            json_remote = f"bills_of_health/{workspace_name}.json"
            if run_provider.exists(html_remote):
                boh_uri = run_provider.get_uri(html_remote)
            elif run_provider.exists(json_remote):
                boh_uri = run_provider.get_uri(json_remote)

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
                    "the run was canceled, or the upload failed."
                ) from e
            except Exception as e:
                raise ex.BillOfHealthNotFoundException(
                    f"Failed to retrieve Bill of Health for run '{run_id}' "
                    f"from storage: {e}"
                ) from e
    else:
        viewer = ResultsViewer(boh_path, boh_uri=boh_uri)

    telemetry_reporting.record_test_results(viewer.boh)
    viewer.show_summary_and_details()


def cancel_action(run_id: str) -> bool:
    """Cancels a run's jobs and cleans up its resources on the cluster.

    Returns:
      Whether the run was active on the cluster.

    Raises:
      ChsCancellationException: If an error occurs during cancellation.
    """
    logger.debug("Canceling run %s.", run_id)
    scheduler = get_scheduler(detect_scheduler())
    try:
        return scheduler.cancel_run(run_id)
    except _CANCELLATION_CLUSTER_ERRORS as e:
        raise ex.ChsCancellationException(
            f"CHS cancellation failed: {e!r}"
        ) from e


def cancel_all_action() -> None:
    """Cancels every CHS job on the cluster and deletes all CHS resources.

    Raises:
      ChsCancellationException: If an error occurs during cancellation.
    """
    logger.debug("Canceling all CHS runs on the cluster.")
    scheduler = get_scheduler(detect_scheduler())
    try:
        scheduler.cancel_all_runs()
    except _CANCELLATION_CLUSTER_ERRORS as e:
        raise ex.ChsCancellationException(
            f"CHS cancellation failed: {e!r}"
        ) from e


def _confirm_cancellation(prompt_msg: str, auto_approve: bool) -> None:
    """Asks the user to confirm a cancellation unless it is auto-approved.

    Raises:
      click.ClickException: If confirmation is needed in non-interactive mode.
      click.Abort: If the user declines.
    """
    if utils.is_in_cluster():
        logger.debug("Running in-cluster. Assuming auto-approve.")
        auto_approve = True

    if auto_approve:
        logger.debug("Auto-approve is enabled. Proceeding with cancellation...")
        return

    if not sys.stdin.isatty():
        logger.warning(prompt_msg)
        raise click.ClickException(
            "Running in non-interactive mode.\n"
            "Please pass --auto-approve to proceed."
        )
    if not click.confirm(prompt_msg, default=False):
        logger.debug("Cancellation aborted by user. Exiting.")
        raise click.Abort()


def safe_cancel_action(run_id: str, auto_approve: bool = False) -> None:
    """Cancels a run after prompting the user for confirmation."""
    _confirm_cancellation(
        f"This will cancel CHS run '{run_id}' and delete its resources on "
        "the cluster.\nAre you sure you want to proceed?",
        auto_approve,
    )
    with theme.get_progress_spinner(
        f"[bold red]Canceling CHS run '{run_id}' and cleaning up its "
        "resources...[/bold red]"
    ):
        found = cancel_action(run_id)
    if not found:
        Console().print(
            f"[{theme.CHS_WARNING_STYLE}]No active CHS run '{run_id}' was "
            f"found on the cluster.[/{theme.CHS_WARNING_STYLE}]"
        )
        return
    Console().print(
        f"[{theme.CHS_SUCCESS_STYLE}]Successfully canceled CHS run "
        f"'{run_id}'.[/{theme.CHS_SUCCESS_STYLE}]"
    )


def safe_cancel_all_action(auto_approve: bool = False) -> None:
    """Cancels every CHS run after prompting the user for confirmation."""
    _confirm_cancellation(
        "This will cancel ALL CHS runs on the cluster, including runs "
        "started by other users, and delete ALL CHS resources on the "
        "cluster.\nAre you sure you want to proceed?",
        auto_approve,
    )
    with theme.get_progress_spinner(
        "[bold red]Canceling all CHS runs and cleaning up resources..."
        "[/bold red]"
    ):
        cancel_all_action()
    Console().print(
        f"[{theme.CHS_SUCCESS_STYLE}]Successfully canceled all CHS runs on "
        f"the cluster.[/{theme.CHS_SUCCESS_STYLE}]"
    )


@click.group(
    cls=telemetry_reporting.TelemetryGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=(
        "Tip: You can append --help to any specific subcommand to see its "
        "exact arguments and examples (e.g., chs run --help or "
        "chs list --help)."
    ),
)
@help_command_text.style_cli_docstring
@click.version_option(
    version=get_version(),
    prog_name="chs",
    message="%(prog)s %(version)s",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Cluster Health Scanner (CHS) Tool.

    A command-line utility for executing diagnostic and benchmark tests across
    accelerator clusters (Kubernetes or Slurm) to evaluate their health.

    \b
    TL;DR: To get started, configure a pre-existing GCS bucket and run a health
    check:
        chs configure storage gcs <bucket-name>
        chs run --nodepools <nodepool-name>

    \b
    Examples:
      # First, configure a pre-existing GCS storage bucket to store results
      chs configure storage gcs <bucket-name>

    \b
      # Start a diagnostic run on specific Kubernetes nodepools
      chs run --nodepools=pool-1,pool-2

    \b
      # View the interactive health report summary for a specific run
      chs results

    \b
      # List all tests and suites available for execution
      chs list

    \b
      # Cancel your latest run and clean up its resources
      chs cancel
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
@click.option(
    "--tpu", is_flag=True, help="Filter list for TPU-related resources."
)
@click.option(
    "--gpu", is_flag=True, help="Filter list for GPU-related resources."
)
@click.pass_context
def list_command(ctx: click.Context, tpu: bool, gpu: bool) -> bool | None:
    """Lists available resources.

    \b
    Examples:
      # List all resources (tests, suites, etc.)
      chs list

    \b
      # Filter all resources to only show TPU-compatible ones
      chs list --tpu

    \b
      # Filter all resources to only show GPU-compatible ones
      chs list --gpu
    """
    ctx.ensure_object(dict)
    if tpu and gpu:
        raise click.UsageError("Cannot specify both --tpu and --gpu.")

    ctx.obj["accelerator_type"] = "tpu" if tpu else ("gpu" if gpu else None)

    if ctx.invoked_subcommand is None:
        return list_action(accelerator_type=ctx.obj["accelerator_type"])


@list_command.command(name="suites")
@click.option(
    "--tpu", is_flag=True, help="Filter list for TPU-related resources."
)
@click.option(
    "--gpu", is_flag=True, help="Filter list for GPU-related resources."
)
@click.pass_context
def list_suites(ctx: click.Context, tpu: bool, gpu: bool) -> bool:
    """List available test suites.

    Test suites are pre-configured bundles of tests designed for specific
    use-cases, like 'lite' or 'standard'.
    """
    if tpu and gpu:
        raise click.UsageError("Cannot specify both --tpu and --gpu.")
    accel = (
        "tpu" if tpu else ("gpu" if gpu else ctx.obj.get("accelerator_type"))
    )
    return list_action(suites=True, accelerator_type=accel)


@list_command.command(name="tests")
@click.option(
    "--suite",
    metavar="SUITE_NAME",
    help="Specify the suite to list tests from",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--machine-type",
    metavar="MACHINE_TYPE",
    help="Filter tests by machine type",
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--tpu", is_flag=True, help="Filter list for TPU-related resources."
)
@click.option(
    "--gpu", is_flag=True, help="Filter list for GPU-related resources."
)
@click.pass_context
def list_tests(
    ctx: click.Context,
    suite: str | None,
    machine_type: str | None,
    tpu: bool,
    gpu: bool,
) -> bool:
    """List individual hardware and network benchmarks.

    \b
    Examples:
      # List all available tests
      chs list tests

    \b
      # List tests that are part of the 'lite' suite
      chs list tests --suite=lite

    \b
      # List all tests that run on TPUs
      chs list tests --tpu

    \b
      # List TPU tests inside the 'standard' suite
      chs list tests --tpu --suite=standard

    \b
      # Filter tests by a specific machine type
      chs list tests --machine-type=tpu7x-standard-4t
    """
    if machine_type:
        supported_mts = utils.tests.get_all_supported_machine_types()
        if machine_type not in supported_mts:
            raise click.UsageError(
                f"Unsupported machine type '{machine_type}'. "
                f"Supported types are: {', '.join(supported_mts)}"
            )

    if tpu and gpu:
        raise click.UsageError("Cannot specify both --tpu and --gpu.")
    accel = (
        "tpu" if tpu else ("gpu" if gpu else ctx.obj.get("accelerator_type"))
    )

    if machine_type and accel:
        if accel == "tpu" and not resources.is_tpu(machine_type):
            raise click.UsageError(
                f"Machine type '{machine_type}' does not appear to be a TPU. "
                "Cannot use with --tpu."
            )
        if accel == "gpu" and not resources.is_gpu(machine_type):
            raise click.UsageError(
                f"Machine type '{machine_type}' does not appear to be a GPU. "
                "Cannot use with --gpu."
            )

    return list_action(
        tests=True,
        suite=suite,
        machine_type=machine_type,
        accelerator_type=accel,
    )


@list_command.command(name="schedulers")
@click.option(
    "--tpu", is_flag=True, help="Filter list for TPU-related resources."
)
@click.option(
    "--gpu", is_flag=True, help="Filter list for GPU-related resources."
)
@click.pass_context
def list_schedulers(ctx: click.Context, tpu: bool, gpu: bool) -> bool:
    """List available schedulers."""
    if tpu and gpu:
        raise click.UsageError("Cannot specify both --tpu and --gpu.")
    accel = (
        "tpu" if tpu else ("gpu" if gpu else ctx.obj.get("accelerator_type"))
    )
    return list_action(schedulers=True, accelerator_type=accel)


@list_command.command(name="machine-types")
@click.option(
    "--tpu", is_flag=True, help="Filter list for TPU-related resources."
)
@click.option(
    "--gpu", is_flag=True, help="Filter list for GPU-related resources."
)
@click.pass_context
def list_machine_types(ctx: click.Context, tpu: bool, gpu: bool) -> bool:
    """List available machine types."""
    if tpu and gpu:
        raise click.UsageError("Cannot specify both --tpu and --gpu.")
    accel = (
        "tpu" if tpu else ("gpu" if gpu else ctx.obj.get("accelerator_type"))
    )
    return list_action(machine_types=True, accelerator_type=accel)


@cli.command(epilog=help_command_text.generate_run_epilog())
@help_command_text.style_run_docstring
@handle_chs_exceptions
@click.option(
    "--suite",
    "-s",
    metavar="SUITE_NAME",
    help=help_command_text.get_suite_help(),
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--tests",
    "-t",
    multiple=True,
    metavar="TEST_LIST",
    help=help_command_text.get_tests_help(),
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--nodes",
    multiple=True,
    metavar="NODE_LIST",
    help=(
        "Comma-separated list (or multiple flags) of individual nodes to run "
        "the tests on (e.g., --nodes=node1,node2, -n node1 -n node2).\n\n"
        f"{click.style('Note:', fg='yellow', bold=True)} Individual nodes are "
        "NOT supported for TPUs. Please use --nodepools instead."
        "\n\n "
    ),
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--nodelists",
    multiple=True,
    metavar="NODE_PATTERN_LIST",
    help=(
        "Slurm nodelist pattern representing the nodes to target (e.g., "
        "node-[1-3],node-[4-5]).\n\n"
        "Only valid when running CHS against a Slurm scheduler environment."
        "\n\n "
    ),
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--nodepools",
    multiple=True,
    metavar="NODEPOOL_LIST",
    help=(
        "Comma-separated list (or multiple flags) of Kubernetes node pools to "
        "target.\n\n"
        "This is the recommended and required way to target TPU resources."
        "\n\n "
    ),
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--partitions",
    multiple=True,
    metavar="PARTITION_LIST",
    help=(
        "Comma-separated list (or multiple flags) of Slurm partitions to run "
        "the tests on.\n\n"
        "The system will resolve all active nodes under these partitions."
        "\n\n "
    ),
    callback=validators.validate_non_empty_string,
)
@click.option(
    "--auto-approve",
    "-y",
    is_flag=True,
    help=(
        "Automatically confirm the execution preview phase without prompting "
        "the user. Useful for automated scripts and CI environments."
        "\n\n "
    ),
)
@click.option(
    "--run-id",
    help=(
        "Custom run ID for this CHS execution. If omitted, a unique identifier "
        "based on the cluster name and suite (e.g., cluster-name-lite-hash) "
        "will be automatically generated."
        "\n\n "
    ),
    callback=validators.validate_run_id,
)
@click.pass_obj
@_handle_termination_signals(_interrupt_run)
def run(
    obj,
    *,
    suite: str | None,
    tests: str | None,
    nodes: str | None,
    nodelists: str | None,
    nodepools: str | None,
    partitions: str | None,
    auto_approve: bool = False,
    run_id: str | None = None,
) -> RunExecutionResult:
    """Run tests or a suite on the specified cluster hardware.

    Submits a set of diagnostic or benchmark tests (or a predefined suite)
    to run across your selected Kubernetes or Slurm nodes. It creates
    isolated workspaces, executes the jobs, generates a unified
    interactive health report summary, and provides a path or link to it.

    Note: We only support targeting resources of a single machine type. Mixed
    hardware targets within a single run are not currently supported.

    \b
    Tests vs Suite:
      - --tests: Run a custom list of individual benchmarks (e.g., tpu-info,
        tpu-collectives)
      - --suite: Run a pre-packaged bundle of tests. Defaults to the 'lite'
        suite if neither flag is provided.

    \b
    Examples:
      # Run the default 'lite' suite on specific Kubernetes TPU nodepools
      chs run --nodepools=tpu-pool-1,tpu-pool-2

    \b
      # Run specific TPU diagnostics on targeted nodepools
      chs run --tests=tpu-hbm,tpu-collectives --nodepools=tpu-pool-1

    \b
      # Run full standard validation suite across TPU nodepools
      chs run --suite=standard --nodepools=tpu-pool-1
    """
    app_data = obj["app_data"]

    with theme.get_progress_spinner(
        "Discovering hardware and validating arguments..."
    ):
        _validate_cli_args(
            suite=suite,
            tests=tests,
            nodes=nodes,
            nodelists=nodelists,
            nodepools=nodepools,
            partitions=partitions,
        )

        _validate_storage(app_data)

        preflights.run_platform_preflights()

        scheduler = detect_scheduler()
        scheduler_obj = get_scheduler(
            scheduler,
            nodes=nodes,
            nodelists=nodelists,
            nodepools=nodepools,
            partitions=partitions,
        )

        resolved_raw_nodes = scheduler_obj.resolve_raw_nodes()

        if not resolved_raw_nodes:
            target_type = ""
            target_value = ""
            for name, value in [
                ("nodes", nodes),
                ("nodelists", nodelists),
                ("nodepools", nodepools),
                ("partitions", partitions),
            ]:
                if value:
                    target_type = name
                    target_value = value
                    break

            if target_value:
                error_msg = (
                    f"No nodes found for testing in the requested "
                    f"{target_type} ('{target_value}')."
                )
            else:
                error_msg = (
                    "No target flags were provided, and the cluster "
                    "currently has no nodes available for testing."
                )
            raise ex.NoNodesProvidedException(error_msg)

        detected_machine_types = scheduler_obj.detect_nodes_hardware(
            resolved_raw_nodes
        )

        if not detected_machine_types or "unknown" in detected_machine_types:
            nodes_str = ", ".join(resolved_raw_nodes)
            raise ex.MachineTypeDiscoveryException(
                f"Failed to detect machine type for nodes ({nodes_str}). "
                "Ensure nodes are Ready and GCE/Kubernetes node provider "
                "information is accessible."
            )

        _validate_run_args(
            scheduler, scheduler_obj, detected_machine_types, tests
        )

        machine_type = detected_machine_types[0]

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
                deadline = int(
                    suite_cfg.get("deadline", resources.DEFAULT_DEADLINE)
                )
            except (ValueError, TypeError):
                deadline = resources.DEFAULT_DEADLINE

            test_names = [t.get("name") for t in resolved_tests]
            logger.debug(
                f"Loaded {len(resolved_tests)} tests ({', '.join(test_names)}) "
                f"for suite '{suite}' from suites.yaml."
            )
            parsed_tests = resolved_tests
        else:
            parsed_tests_strs = _parse_str_to_list(tests)

            suites_to_search = utils.suites.get_all_suites()
            suite = "custom"

            parsed_tests = []
            for t_name in parsed_tests_strs or []:
                test_cfg = None
                for s_name in suites_to_search:
                    try:
                        s_cfg = load_suite(s_name)
                        for t in s_cfg.get("tests", []):
                            if t.get("name") == t_name:
                                test_cfg = dict(t)
                                break
                    except ex.SuiteNotFoundException:
                        continue
                    if test_cfg:
                        break

                if test_cfg:
                    parsed_tests.append(test_cfg)
                else:
                    parsed_tests.append({"name": t_name})

            if suite != "custom":
                try:
                    s_cfg = load_suite(suite)
                    deadline = int(
                        s_cfg.get("deadline", resources.DEFAULT_DEADLINE)
                    )
                except (ValueError, TypeError, ex.SuiteNotFoundException):
                    pass

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

    with theme.get_progress_spinner(
        f"[{theme.CHS_HEADER_STYLE}]Identifying idle nodes for testing..."
    ):
        nodes_under_test = scheduler_obj.get_nodes_under_test()
    if not nodes_under_test:
        raise ex.NoNodesProvidedException(
            "No valid nodes remain available for testing."
        )

    cluster_name = utils.get_sanitized_cluster_name(scheduler_obj)
    full_cluster_name = scheduler_obj.get_cluster_name()

    scheduled_tests = active_tier_tests if active_tier_tests else parsed_tests
    if not scheduled_tests:
        raise ex.ChsValidationError(
            "No valid tests can be scheduled on this cluster."
        )

    scheduled_test_names = [_get_test_name(t) for t in scheduled_tests]
    test_names = ", ".join(scheduled_test_names)
    preview_title = "CHS Execution Preview:"
    console = Console()
    console.print(f"[bold]{preview_title}[/bold]")
    console.print(
        f"  Target Cluster:    [{theme.CHS_HEADER_STYLE}]{full_cluster_name}"
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
        prompt_msg = "Proceed with execution?"
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

    runs_targeting_nodes = scheduler_obj.find_runs_targeting_nodes(
        nodes_under_test
    )
    if runs_targeting_nodes:
        raise ex.NodesTargetedByOtherRunsException(runs_targeting_nodes)

    short_hash = utils.generate_random_suffix()
    if not run_id:
        run_id = f"{cluster_name}-{suite}"
    run_id = f"{run_id}-{short_hash}"

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    workspace_name = f"{timestamp}_{run_id}"
    telemetry.set_telemetry_run_id(workspace_name)
    telemetry_reporting.record_run(
        scheduler_obj,
        app_data.get_storage_config(),
        machine_type,
        nodes_under_test,
        suite,
        scheduled_test_names,
    )
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
            "machine_type": machine_type,
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

    with cancel_jobs_on_failure(scheduler_obj, run_id):
        with theme.get_progress_spinner(status_msg):
            job_ids = run_action(
                suite=suite,
                scheduler_obj=scheduler_obj,
                tests=parsed_tests,
                workspace_name=workspace_name,
                workspaces_base_dir=workspaces_base_dir,
                machine_type=machine_type,
                deadline=deadline,
                run_id=run_id,
                active_tests=active_tier_tests if is_tiered else None,
            )

        if not job_ids:
            raise ex.JobSubmissionFailureException(
                f"Job submission failed for "
                f"workspace: {workspace_name}. "
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
        _analyze_action(app_data, run_id)

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
    "run_id", required=False, callback=validators.validate_existing_run_id
)
@click.option(
    "--all",
    "cancel_all",
    is_flag=True,
    help=(
        "Cancel EVERY CHS run on the cluster, including runs started by "
        "other users, and delete all CHS resources."
    ),
)
@click.option(
    "--auto-approve",
    "-y",
    is_flag=True,
    help="Automatically confirm cancellation without prompting.",
)
@click.pass_obj
def cancel(
    obj, run_id: str | None, cancel_all: bool, auto_approve: bool = False
) -> None:
    """Cancel a CHS run and clean up its resources.

    Cancels the run's jobs and deletes its resources on the cluster, which
    releases its nodes for other runs. Without RUN_ID, your latest run is
    cancelled. RUN_ID can be any run on the cluster, including runs started
    by other users.

    \b
    Examples:
      # Cancel your latest run
      chs cancel

    \b
      # Cancel a specific run
      chs cancel my-cluster-lite-ab12

    \b
      # Cancel every CHS run on the cluster
      chs cancel --all
    """
    app_data = obj["app_data"]
    if run_id and cancel_all:
        raise click.UsageError("RUN_ID and --all are mutually exclusive.")

    if cancel_all:
        safe_cancel_all_action(auto_approve=auto_approve)
        return

    if not run_id:
        try:
            run_id = app_data.get_latest_run_id()
        except ex.NoPreviousRunsFoundException as e:
            raise ex.NoPreviousRunsFoundException(
                "No local CHS runs found. Pass a RUN_ID to cancel a specific "
                "run, or --all to cancel every CHS run on the cluster."
            ) from e
    try:
        workspace_name = app_data.resolve_run_id(run_id)
    except ex.WorkspaceNotFoundException:
        pass
    else:
        telemetry.set_telemetry_run_id(workspace_name)
    safe_cancel_action(run_id, auto_approve=auto_approve)


@cli.command()
@handle_chs_exceptions
@click.argument(
    "run_id", required=False, callback=validators.validate_non_empty_string
)
@click.pass_obj
def results(
    obj,
    run_id: str | None,
) -> None:
    """Display interactive health report summary for a run in the terminal.

    If no run_id is provided, you will be prompted to select one.
    """
    app_data = obj["app_data"]
    run_id = _get_run_id_or_prompt(app_data, run_id)
    _show_results(app_data, run_id)


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
    # Temporarily bypassing storage type selection to mandate GCS storage
    # click.echo("Select storage type:")
    # click.echo("  [1] GCS")
    # click.echo("  [2] Path")
    # choice = click.prompt("Choice", type=int, default=1)
    #
    # if choice == 1:
    click.echo("Configuring GCS storage...")
    bucket_name = click.prompt("Enter GCS bucket name")
    if bucket_name.lower().startswith("gs://"):
        bucket_name = bucket_name[5:]
    config = {"type": "gcs", "gcs": {"bucket_name": bucket_name}}
    # elif choice == 2:
    #     path = click.prompt("Enter target directory path (absolute)")
    #     config = {"type": "path", "path": {"path": path}}
    # else:
    #     raise click.ClickException("Invalid choice.")

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


# Temporarily disabled to mandate GCS storage.
# Uncomment to re-enable local path storage configuration.
# @configure_storage.command("path")
# @click.argument("path", required=False)
# @click.pass_context
# def configure_path(ctx: click.Context, path: str | None) -> None:
#     """Configure Local Path storage."""
#     app_data = ctx.obj["app_data"]
#     if not path:
#         path = click.prompt("Enter target directory path (absolute)")
#
#     config = {"type": "path", "path": {"path": path}}
#     _validate_and_save_storage(app_data, config)


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


@configure.group("telemetry")
def configure_telemetry() -> None:
    """Configure usage telemetry."""
    pass


@configure_telemetry.command("enable")
@click.pass_context
def enable_telemetry(ctx: click.Context) -> None:
    """Enable telemetry."""
    app_data = ctx.obj["app_data"]
    app_data.set_telemetry_enabled(True)
    env_override = app_data.telemetry_env_override()
    if env_override:
        click.echo(
            f"Telemetry enabled, but it stays off while {env_override} is set."
        )
    else:
        click.echo("Telemetry enabled.")


@configure_telemetry.command("disable")
@click.pass_context
def disable_telemetry(ctx: click.Context) -> None:
    """Disable telemetry."""
    ctx.obj["app_data"].set_telemetry_enabled(False)
    click.echo("Telemetry disabled.")


@configure_telemetry.command("status")
@click.pass_context
def telemetry_status(ctx: click.Context) -> None:
    """Show whether telemetry is enabled."""
    app_data = ctx.obj["app_data"]
    env_override = app_data.telemetry_env_override()
    if env_override:
        click.echo(f"Telemetry is disabled because {env_override} is set.")
    elif app_data.is_telemetry_enabled():
        click.echo("Telemetry is enabled.")
    else:
        click.echo("Telemetry is disabled.")


def main(args: list[str] | None = None) -> None:
    """Main entry point for the CHS CLI."""

    logger.debug("CHS CLI starting...")
    try:
        cli(args=args, obj={})
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        logger.debug("CHS CLI finished.")
