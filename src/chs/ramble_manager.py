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

"""Manages Ramble workspaces for Cluster Health Scanner runs.

This module provides the RambleManager class, which encapsulates
the creation, setup, and execution of Ramble workspaces for CHS test runs.
"""

from collections.abc import Callable
import datetime
import importlib.resources as pkg_resources
import json
from concurrent.futures import ThreadPoolExecutor, wait
import logging
import os
import pathlib
import re
import shutil

import subprocess
import sys
import sysconfig
import time
from typing import Any
import click
from google.protobuf.json_format import MessageToJson

from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.console import Group

from chs import exceptions as ex
from chs.boh_generator import BOHGenerator
from chs.schedulers import get_scheduler
from chs.schedulers.base import BaseScheduler
from chs import resources
from chs.utils.app_data import ChsAppData
from chs.utils.lifecycle import JobStatus
import yaml

logger = logging.getLogger(__name__)


class RambleManager:
    """A manager for Ramble workspaces used in CHS runs."""

    def __init__(
        self,
        base_dir: str,
        workspace_name: str,
        run_command: Callable[
            ..., subprocess.CompletedProcess[str]
        ] = subprocess.run,
        run_id: str | None = None,
    ) -> None:
        """Initializes the RambleManager.

        Args:
          base_dir: The base directory where Ramble workspaces will be created.
          workspace_name: The physical workspace directory name.
          run_command: The function to use for running subprocess commands.
            Defaults to `subprocess.run`.
          run_id: The user-facing run ID.

        Raises:
          click.UsageError: If the Ramble setup script is not found.
          OSError: If the workspace directory cannot be created.
        """

        self._base_dir = base_dir
        self._workspace_name = workspace_name
        self._run_id = run_id
        self._run_command = run_command
        self._app_data = ChsAppData()
        self._path = os.path.join(self._base_dir, self._workspace_name)

        # Ensure the directory exists for logging and execution
        os.makedirs(self._path, exist_ok=True)

        logger.debug(
            "Initialized manager with run_id='%s', workspace_name='%s', "
            "path='%s'",
            self._run_id,
            self._workspace_name,
            self._path,
        )

        site_packages_dir = sysconfig.get_path("purelib")
        logger.debug("Python site-packages directory: %s", site_packages_dir)

        self._ramble_setup_script = os.path.abspath(
            os.path.join(
                site_packages_dir,
                "share/ramble/setup-env.sh",
            )
        )
        if not os.path.exists(self._ramble_setup_script):
            raise click.UsageError("Please ensure CHS is installed correctly.")

    def __repr__(self) -> str:
        return (
            f"RambleManager(run_id='{self._run_id}',"
            f" workspace_name='{self._workspace_name}',"
            f" path='{self._path}')"
        )

    @property
    def path(self) -> str:
        """The absolute path to the Ramble workspace."""
        return self._path

    @property
    def workspace_name(self) -> str:
        """The physical workspace directory name."""
        return self._workspace_name

    @property
    def run_id(self) -> str | None:
        """The user-facing run ID."""
        return self._run_id

    def _prepare_ramble_command(self, cmd: list[str]) -> list[str]:
        """Prepares a Ramble command by sourcing setup script."""
        command_str = (
            f'export RAMBLE_PYTHON="{sys.executable}" && '
            f'source {self._ramble_setup_script} && exec "$@"'
        )
        logger.debug(
            "Sourcing %s before running command.",
            self._ramble_setup_script,
        )
        return ["bash", "-c", command_str, "bash"] + cmd

    def _execute_shell_command(
        self,
        cmd: list[str],
        check: bool = True,
        cwd: str | None = None,
        capture_output: bool = False,
        env: dict[str, str] | None = None,
        error_context: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Executes a command with consistent logging and error handling.

        Args:
          cmd: The command to execute as a list of strings.
          check: If True, raises a CalledProcessError if the command returns a
            non-zero exit code.
          cwd: The current working directory to run the command in.
          capture_output: If True, captures stdout and stderr.
          env: A dictionary of environment variables to add to the command
            execution.
          error_context: Optional context message to prepend to the error
            message if the command fails.

        Returns:
          The CompletedProcess object from the execution command.

        Raises:
          ex.ChsException: If the command fails when check=True.
        """
        logger.debug("Running command: %s", cmd)
        if env:
            run_env = os.environ.copy()
            run_env.update(env)
        else:
            run_env = None

        log_file_path = os.path.join(self.path, "internal.log")

        try:
            with open(log_file_path, "a") as log_file:
                log_file.write(
                    f"\n[{datetime.datetime.now()}] "
                    f"Running command: {' '.join(cmd)}\n"
                )
                log_file.flush()

                completed_process = self._run_command(
                    cmd,
                    cwd=cwd,
                    env=run_env,
                    capture_output=capture_output,
                    text=True,
                    check=False,
                )
                if completed_process.stdout:
                    log_file.write(completed_process.stdout)
                if completed_process.stderr:
                    log_file.write(completed_process.stderr)

                if check and completed_process.returncode != 0:
                    msg = (
                        f"Command failed with exit code "
                        f"{completed_process.returncode}. "
                        f"See {log_file_path} for details."
                    )
                    if error_context:
                        msg = f"{error_context}. {msg}"
                    raise ex.ChsException(msg)

                return completed_process

        except Exception as e:
            msg = (
                f"Failed to execute command: {e}. "
                f"See {log_file_path} for details."
            )
            if error_context:
                msg = f"{error_context}: {msg}"
            raise ex.ChsException(msg)

    def _execute_ramble_command(
        self,
        subcommand: str,
        args: list[str],
        capture_output: bool = False,
        env: dict[str, str] | None = None,
        error_context: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Runs a Ramble subcommand in the workspace context.

        This is a wrapper around _execute_shell_command that prepends
        'ramble -d -w self.path' to the command and sets cwd to self.path.

        Args:
          subcommand: The Ramble subcommand to execute (e.g., 'workspace',
            'on').
          args: A list of arguments for the Ramble subcommand.
          capture_output: Whether to capture stdout/stderr.
          env: A dictionary of environment variables to add to the command
            execution.
          error_context: Optional context message to prepend to the error
            message if the command fails.
          check: If True, raises a ChsException if the command returns a
            non-zero exit code.

        Returns:
          The CompletedProcess object from the execution command.

        Raises:
          ex.ChsException: If the Ramble command fails.
        """
        cmd = ["ramble", "-d", "-w", self.path, subcommand] + args
        exec_cmd = self._prepare_ramble_command(cmd)

        merged_env = env.copy() if env else {}
        if hasattr(self, "_scheduler"):
            merged_env.update(self._scheduler.get_extra_env())

        return self._execute_shell_command(
            exec_cmd,
            check=check,
            cwd=self.path,
            capture_output=capture_output,
            env=merged_env,
            error_context=error_context,
        )

    def create_workspace(self) -> None:
        """Creates the Ramble workspace directory and initializes it.

        Raises:
          click.ClickException: If the workspace creation command fails.
        """
        logger.debug("Creating workspace at: %s", self.path)
        cmd = ["ramble", "-d", "workspace", "create", "-a", "-d", self.path]
        exec_cmd = self._prepare_ramble_command(cmd)
        self._execute_shell_command(
            exec_cmd,
            capture_output=True,
            error_context="Failed to create workspace",
        )

    def copy_thresholds(self, thresholds_path: str | None = None) -> None:
        """Copies the thresholds configuration file into the workspace."""
        import shutil

        dest_thresholds = os.path.join(self.path, "thresholds.yaml")
        if thresholds_path:
            shutil.copyfile(thresholds_path, dest_thresholds)
            logger.info(
                "Copied user thresholds from %s to %s",
                thresholds_path,
                dest_thresholds,
            )
        else:
            internal_thresholds = os.path.join(
                os.path.dirname(__file__), "configs", "thresholds.yaml"
            )
            if os.path.exists(internal_thresholds):
                shutil.copyfile(internal_thresholds, dest_thresholds)
                logger.info("Copied default thresholds to %s", dest_thresholds)
            else:
                logger.warning(
                    "Default thresholds file not found at %s",
                    internal_thresholds,
                )

    def add_custom_repository(self) -> None:
        """Adds the internal custom CHS repository to the Ramble workspace.

        Raises:
          click.ClickException: If the repository addition command fails.
        """
        import chs

        logger.debug("Adding custom CHS repository to workspace...")
        repo_path = pkg_resources.files(chs).joinpath("repos")
        with pkg_resources.as_file(repo_path) as p:
            self._execute_ramble_command(
                subcommand="repo",
                args=["add", str(p)],
                capture_output=True,
                error_context="Failed to add custom repository",
            )

    def configure_workspace(
        self,
        scheduler: BaseScheduler,
        tests: list[str] | None,
        machine_type: str,
        timeout: int | None = None,
        deadline: int | None = None,
    ) -> None:
        """Configures the Ramble workspace for the CHS run.

        Args:
          scheduler: The scheduler plugin to use (e.g., SlurmScheduler).
          tests: An optional list of names of specific tests to run.
          machine_type: The machine type.
          timeout: Suite-level timeout in seconds.
          deadline: Suite-level deadline in seconds.
        """
        self._scheduler = scheduler

        logger.debug("Configuring Ramble workspace in %s...", self.path)
        logger.debug("Configuring workspace for scheduler: %s", scheduler.name)
        if tests:
            logger.debug("Configuring workspace for tests: %s", tests)

        workspace_config_dir = os.path.join(self.path, "configs")
        os.makedirs(workspace_config_dir, exist_ok=True)

        scheduler.write_ramble_config(
            workspace_config_dir=workspace_config_dir,
            tests=tests,
            machine_type=machine_type,
            timeout=timeout,
            deadline=deadline,
            run_id=self.run_id,
            images_dir=self._app_data.get_images_dir(),
        )

    def setup_workspace(self) -> None:
        """Sets up the Ramble workspace by running 'ramble workspace setup'.

        Raises:
          click.ClickException: If the workspace setup command fails.
        """

        logger.debug("Setting up workspace in %s...", self.path)
        self._execute_ramble_command(
            subcommand="workspace",
            args=["setup"],
            capture_output=True,
            env={"TQDM_DISABLE": "1", "TERM": "dumb"},
            error_context="Failed to setup workspace",
        )

    def execute_experiments(self) -> list[str]:
        """Executes experiments in the Ramble workspace.

        This runs the 'all_experiments' script to start the experiments and
        captures Job IDs by scanning the workspace experiments directory within
        the current filesystem.

        Returns:
          A list of submitted Job IDs.

        Raises:
          ex.ChsException: If the experiment execution command fails.
        """

        logger.debug("Executing experiments in %s...", self.path)

        logger.info("Running experiments via all_experiments script...")
        script_path = os.path.join(self.path, "all_experiments")
        self._execute_shell_command(
            cmd=["bash", script_path],
            capture_output=True,
            error_context="Failed to execute all_experiments script",
        )

        job_to_dir = {}
        experiments_dir = os.path.join(self.path, "experiments")
        if os.path.exists(experiments_dir):
            for root, dirs, files in os.walk(experiments_dir):
                if "execute_experiment" in files:
                    job_id = self._scheduler.extract_job_id(root)
                    job_to_dir[job_id] = root
                    dirs.clear()

        if job_to_dir:
            self._update_registry(job_to_dir)

        run_entry = self._app_data.get_run(self._workspace_name) or {}
        run_entry["submission_complete"] = True
        self._app_data.update_run(self._workspace_name, run_entry)
        logger.info(
            "All JobSets submitted successfully. "
            "submission_complete set to True."
        )

        return list(job_to_dir.keys())

    def cancel_experiments(self) -> None:
        """Cancels running experiments in the Ramble workspace.

        This calls the scheduler to cancel all jobs matching the workspace's
        unique prefix, guaranteeing stateless cleanup even if jobs weren't
        registered.

        Raises:
          click.ClickException: If the experiment cancellation command fails.
        """
        registry_entry = self._app_data.get_run(self._workspace_name)
        scheduler_name = (
            registry_entry.get("scheduler", "slurm")
            if registry_entry
            else "slurm"
        )
        logger.debug("Using scheduler '%s' for canceling jobs.", scheduler_name)
        scheduler = get_scheduler(scheduler_name)

        prefix = "chs-"
        logger.debug(
            "Canceling experiments with prefix %s in %s...", prefix, self.path
        )

        # Delete jobs statelessly by prefix to guarantee cleanup.
        scheduler.cancel_jobs_by_prefix(prefix)
        # Clean up associated infrastructure configurations like Kueue
        # statelessly.
        scheduler.cleanup_scheduler_resources()

    def _update_registry(self, job_to_dir: dict[str, str]) -> None:
        """Updates the runs registry with the current run information."""
        logger.debug(
            "Updating registry for run %s with job mappings %s",
            self._workspace_name,
            job_to_dir,
        )
        run_entry = self._app_data.get_run(self._workspace_name) or {}

        job_entries = {
            job_id: {
                "status": JobStatus.PENDING.value,
                "experiment_dir": exp_dir,
            }
            for job_id, exp_dir in job_to_dir.items()
        }

        run_entry.update(
            {
                "run_id": self.run_id,
                "workspace_path": self.path,
                "job_entries": job_entries,
                "submission_time": (
                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                ),
                "scheduler": self._scheduler.name,
            }
        )
        self._app_data.update_run(self._workspace_name, run_entry)

    def get_job_ids(self) -> list[str]:
        """Retrieves job IDs for this workspace from the registry."""
        entry = self._app_data.get_run(self._workspace_name)
        if entry:
            return list(entry.get("job_entries", {}).keys())
        return []

    def _map_jobs_to_log_paths(self) -> dict[str, str]:
        """Maps Job IDs to their expected log paths on the filesystem.

        Uses the experiment_dir registered in runs.json to find the log files.
        """
        run_entry = self._app_data.get_run(self._workspace_name) or {}
        job_entries = run_entry.get("job_entries", {})
        mapping = {}
        for job_id, info in job_entries.items():
            exp_dir = info.get("experiment_dir")
            if not exp_dir or not os.path.exists(exp_dir):
                logger.warning(
                    "Experiment directory %s for job %s does not exist",
                    exp_dir,
                    job_id,
                )
                continue

            try:
                files = os.listdir(exp_dir)
                out_files = [
                    f
                    for f in files
                    if f.endswith(".out")
                    and f != "execute.log"
                    and not f.startswith("execute")
                ]

                if not out_files:
                    logger.debug(
                        "No .out file found in %s for job %s",
                        exp_dir,
                        job_id,
                    )
                    continue

                # Priority 1: File containing job_id
                log_path = None
                for out_file in out_files:
                    if job_id in out_file:
                        log_path = os.path.join(exp_dir, out_file)
                        break

                # Priority 2: Any other .out file (fallback)
                if not log_path:
                    log_path = os.path.join(exp_dir, out_files[0])

                if log_path:
                    mapping[job_id] = log_path
            except OSError as e:
                logger.warning(
                    "Failed to list directory %s for job %s: %s",
                    exp_dir,
                    job_id,
                    e,
                )

        return mapping

    def wait_for_jobs(
        self, app_data: ChsAppData, poll_interval: int = 30
    ) -> None:
        """Waits for all jobs in the workspace to complete and fetches logs.

        Args:
            app_data: ChsAppData to update run status.
            poll_interval: Seconds to wait between checks.
        """
        logger.info("Waiting for jobs to complete for run %s...", self._run_id)
        self._monitor_jobs(app_data, poll_interval)

    def _monitor_jobs(self, app_data: ChsAppData, poll_interval: int) -> None:
        """Generic wait logic to monitor job progress and fetch logs.

        Polls job statuses for all jobs in the registry. When a job completes,
        attempts to fetch its log (if a path is mapped) in parallel.
        Logs progress.
        """
        job_log_map = self._map_jobs_to_log_paths()
        pending_log_fetches = set(job_log_map.keys())
        scheduler = get_scheduler(self.get_scheduler_type())

        registry_jobs = set(self.get_job_ids())
        if not registry_jobs:
            logger.info("No jobs found in registry.")
            return

        # We poll ALL registry jobs for status
        active_jobs = registry_jobs.copy()
        run_entry = app_data.get_run(self._workspace_name) or {}
        QUEUE_TIMEOUT_SECONDS = run_entry.get(
            "deadline", resources.DEFAULT_DEADLINE
        )
        pending_start_times = {}

        max_workers = 32

        logger.debug(
            "Starting log fetching executor with %d workers", max_workers
        )

        start_time = time.time()

        is_fetching_logs = False

        def generate_status_renderable() -> Group | Text:
            if is_fetching_logs:
                return Text(
                    f"Waiting for {len(futures)} pending log "
                    "fetches to complete..."
                )
            elapsed = int(time.time() - start_time)
            elapsed_td = datetime.timedelta(seconds=elapsed)
            spinner = Spinner(
                "dots", text="Waiting for test execution to complete..."
            )
            details = Text(
                f"    • Jobs Remaining: {len(active_jobs)}\n"
                f"    • Elapsed Time:   {elapsed_td}\n\n"
                f"(Press Ctrl+C to detach from this view)"
            )
            return Group(spinner, details)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                live_context = None
                if sys.stdout.isatty():
                    live_context = Live(
                        get_renderable=generate_status_renderable,
                        refresh_per_second=1,
                    )
                    live_context.start()

                try:
                    while active_jobs:
                        statuses = scheduler.get_job_statuses(list(active_jobs))

                        jobs_to_cancel = []
                        still_running = False
                        for job_id, status in statuses.items():
                            if JobStatus.is_terminal(status):
                                # Job finished
                                pending_start_times.pop(job_id, None)
                                if job_id in pending_log_fetches:
                                    log_path = job_log_map.get(job_id)
                                    if log_path:
                                        logger.debug(
                                            "Fetching log for "
                                            "completed job: %s",
                                            job_id,
                                        )
                                        future = executor.submit(
                                            self._safe_fetch_log,
                                            scheduler,
                                            job_id,
                                            log_path,
                                        )
                                        futures.append(future)
                                    pending_log_fetches.remove(job_id)

                                active_jobs.remove(job_id)
                            elif status == JobStatus.PENDING:
                                if job_id not in pending_start_times:
                                    pending_start_times[job_id] = time.time()

                                if (
                                    time.time() - pending_start_times[job_id]
                                    > QUEUE_TIMEOUT_SECONDS
                                ):
                                    logger.error(
                                        "Job %s has been queued for more than "
                                        "%d seconds. Marking for "
                                        "cancellation...",
                                        job_id,
                                        QUEUE_TIMEOUT_SECONDS,
                                    )
                                    jobs_to_cancel.append(job_id)
                                else:
                                    still_running = True
                            else:
                                pending_start_times.pop(job_id, None)
                                still_running = True

                        if jobs_to_cancel:
                            scheduler.cancel_jobs(jobs_to_cancel)
                            for canceled_job in jobs_to_cancel:
                                statuses[canceled_job] = JobStatus.DEADLINE
                                if canceled_job in active_jobs:
                                    active_jobs.remove(canceled_job)
                                pending_start_times.pop(canceled_job, None)

                        try:
                            app_data.update_run_status(
                                self._workspace_name, statuses
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to update run status in registry: %s", e
                            )

                        if still_running:
                            if not live_context:
                                elapsed = int(time.time() - start_time)
                                elapsed_td = datetime.timedelta(seconds=elapsed)
                                logger.info(
                                    "Jobs remaining: %d | Elapsed time: %s",
                                    len(active_jobs),
                                    elapsed_td,
                                )
                            time.sleep(poll_interval)

                    if futures:
                        is_fetching_logs = True
                        if not live_context:
                            logger.debug(
                                "Waiting for %d pending log "
                                "fetches to complete...",
                                len(futures),
                            )
                        wait(futures)
                        for f in futures:
                            try:
                                f.result()
                            except Exception as e:
                                logger.debug("Log fetch task failed: %s", e)
                finally:
                    if live_context:
                        live_context.stop()

        except KeyboardInterrupt:
            logger.info(
                "Detached from active run. "
                "Jobs will continue executing on the cluster."
            )
            logger.info(
                "To compile logs and generate results run: "
                f"chs analyze {self._run_id}"
            )
            logger.info(
                f"When completed, view results with: chs results {self._run_id}"
            )
            raise ex.ChsDetachedException()

        logger.info(
            "All jobs completed and logs fetched for run %s.",
            self._run_id,
        )

    def _safe_fetch_log(
        self, scheduler: BaseScheduler, job_id: str, log_path: str
    ) -> None:
        """Helper to run fetch_job_log safely in a thread."""
        try:
            scheduler.fetch_job_log(job_id, log_path)
            logger.debug("Successfully fetched log for %s", job_id)
        except Exception as e:
            logger.error("Error fetching log for %s: %s", job_id, e)
            raise

    def get_scheduler_type(self) -> str:
        """Retrieves scheduler type for this workspace from the registry."""
        if hasattr(self, "_scheduler"):
            return self._scheduler.name

        entry = self._app_data.get_run(self._workspace_name)
        if entry:
            return entry.get("scheduler", "slurm")
        return "slurm"

    def _get_application_names(self, results_data: dict[str, Any]) -> set[str]:
        """Extracts unique application names from results data.

        Args:
            results_data: The dictionary containing the Ramble results.

        Returns:
            A set of application names found in the results.
        """
        app_names = set()
        for experiment in results_data.get("experiments", []):
            app_name = experiment.get("application_name")
            if app_name:
                app_names.add(app_name)
        return app_names

    def _get_success_criteria_mapping(
        self, names: set[str], key_type: str = "applications"
    ) -> dict[str, dict[str, str]]:
        """Fetches success criteria mapping for a set of names.

        Args:
            names: A set of names (applications or workflow_managers).
            key_type: The type of keys ("applications" or "workflow_managers").

        Returns:
            A dictionary mapping names to their success criteria mapping
            (which maps criteria names to FOM regexes).
        """
        mapping_result = {}

        for name in names:
            item_mapping = {}
            try:
                process = self._execute_ramble_command(
                    subcommand="info",
                    args=[
                        "--type",
                        key_type,
                        name,
                        "--attrs",
                        "success_criteria",
                        "-v",
                    ],
                    capture_output=True,
                    env={"TQDM_DISABLE": "1", "TERM": "dumb"},
                    check=False,
                )

                if process.returncode == 3:
                    logger.debug(
                        "No success criteria mapping found for %s %s"
                        " (ramble info exit code 3).",
                        key_type,
                        name,
                    )
                    continue
                elif process.returncode != 0:
                    logger.warning(
                        "Fetching success criteria for %s %s "
                        "failed with exit code %s. "
                        "Continuing without success criteria.",
                        key_type,
                        name,
                        process.returncode,
                    )
                    continue

                output = process.stdout
                if "success_criteria" not in output:
                    continue

                parts = output.split("success_criteria #")
                if len(parts) < 2:
                    continue

                yaml_content = parts[1].strip()
                if yaml_content.startswith("####################"):
                    yaml_content = yaml_content[20:].strip()

                def quote_formula(match):
                    return match.group(1) + json.dumps(match.group(2).strip())

                yaml_content = re.sub(
                    r"^(\s*formula:\s*)(.*)$",
                    quote_formula,
                    yaml_content,
                    flags=re.MULTILINE,
                )

                criteria_defs = yaml.safe_load(yaml_content)
                if not isinstance(criteria_defs, dict):
                    continue

                for criteria_name, definition in criteria_defs.items():
                    if isinstance(definition, dict):
                        item_mapping[criteria_name] = definition

                if item_mapping:
                    mapping_result[name] = item_mapping

                logger.debug(
                    "Success criteria mapping for %s %s: %s",
                    key_type,
                    name,
                    item_mapping,
                )

            except (
                yaml.YAMLError,
                Exception,
            ) as e:
                logger.warning(
                    "Fetching success criteria for %s %s "
                    "failed with error: %s. "
                    "Continuing without success criteria.",
                    key_type,
                    name,
                    e,
                )

        return mapping_result

    def analyze_results(
        self, thresholds_path: str | None = None, skip_validation: bool = False
    ) -> str:
        """Analyzes and stores the results of the Ramble experiments.

        Analyzes the results and stores BoH in GCS/ BigQuery for later
        retrieval.

        Returns:
          The path to the directory containing the analysis results and Bill of
          Health.

        Raises:
          click.ClickException: If the experiment analyze command fails.
        """

        logger.debug("Analyzing experiments in %s...", self.path)
        self._execute_ramble_command(
            subcommand="workspace",
            args=[
                "analyze",
                "-f",
                "json",
                "--fom-origin-types",
                "modifier",
                "--fom-origin-types",
                "application",
            ],
            capture_output=True,
            env={"TQDM_DISABLE": "1", "TERM": "dumb"},
            error_context="Failed to analyze results",
        ).stdout

        results_json_path = os.path.join(self.path, "results.latest.json")

        if not os.path.exists(results_json_path):
            raise ex.ChsException(
                "chs analyze command failed. Recheck the run id and try "
                "rerunning the command."
            )
        self._cleanup_inputs_dir()
        return self.generate_boh(
            results_json_path,
            thresholds_path=thresholds_path,
            skip_validation=skip_validation,
        )

    def generate_boh(
        self,
        results_json_path: str,
        thresholds_path: str | None = None,
        skip_validation: bool = False,
    ) -> str:
        """Generates Bill of Health from results.latest.json.

        Args:
          results_json_path: Path to the results.latest.json file.
          thresholds_path: Path to thresholds YAML file.

        Returns:
          The path to the directory containing the analysis results and Bill of
          Health.
        """

        workspace_dir = pathlib.Path(self._path)

        storage_config = self._app_data.get_storage_config()
        gcs_base_path = None
        if storage_config and storage_config.get("type") == "gcs":
            bucket_name = storage_config.get("gcs", {}).get("bucket_name")
            if bucket_name:
                gcs_base_path = (
                    f"gs://{bucket_name}/workspaces/{self._workspace_name}"
                )

        boh_generator = BOHGenerator(
            results_json_path,
            base_log_dir=str(workspace_dir),
            cluster_type=self.get_scheduler_type(),
            gcs_base_path=gcs_base_path,
            thresholds_path=thresholds_path,
            skip_validation=skip_validation,
        )
        logger.debug("Generating Bill of Health from %s", results_json_path)
        boh = boh_generator.generate()
        logger.debug("Bill of Health generated successfully.")

        boh_path = workspace_dir / "bill_of_health.json"
        with open(boh_path, "w") as f:
            f.write(MessageToJson(boh, preserving_proto_field_name=True))

        # Cleanup jobs after successful analysis
        try:
            self.cancel_experiments()
            logger.debug("Cleaned up jobs for run %s", self._run_id)
        except Exception as e:
            logger.warning("Failed to cleanup jobs after analysis: %s", e)

        return str(workspace_dir)

    def _upload_boh(self, provider) -> None:
        """Uploads Bill of Health using the provider."""
        boh_path = os.path.join(self.path, "bill_of_health.json")
        if not os.path.exists(boh_path):
            logger.warning(
                "Bill of Health file %s does not exist. Skipping BOH archival.",
                boh_path,
            )
            return

        logger.info("Uploading Bill of Health...")
        try:
            provider.upload_file(
                boh_path, f"bills_of_health/{self._workspace_name}.json"
            )
            logger.info("Bill of Health uploaded successfully.")
        except Exception as e:
            raise ex.ChsArchiveException(
                f"Failed to archive Bill of Health: {e}"
            ) from e

    def _upload_workspace(self, provider) -> None:
        """Archives workspace locally and uploads using the provider."""
        if not os.path.exists(self.path):
            logger.warning(
                "Workspace path %s does not exist. Skipping archival.",
                self.path,
            )
            return

        logger.info("Archiving workspace locally...")
        try:
            # Archive locally (without --upload-url)
            self._execute_ramble_command(
                "workspace",
                [
                    "archive",
                    "-t",
                    "--archive-pattern",
                    "bill_of_health.json",
                    "--archive-pattern",
                    "internal.log",
                ],
                capture_output=True,
                error_context="Failed to archive workspace",
            )

            # Scan for the generated tarball
            archive_dir = pathlib.Path(self.path) / "archive"
            archives = list(archive_dir.glob("*.tar.gz")) + list(
                archive_dir.glob("*.tgz")
            )

            if not archives:
                raise ex.ChsArchiveException(
                    "Workspace archive file not found."
                )

            # Use the first match (should be only one if clean)
            archive_file = archives[0]
            logger.info("Uploading workspace archive %s...", archive_file.name)

            provider.upload_file(
                str(archive_file), f"workspaces/{self._workspace_name}.tar.gz"
            )
            logger.info("Workspace archive uploaded successfully.")

            # Clean up local archive
            try:
                archive_file.unlink()
                logger.debug("Deleted local archive file: %s", archive_file)
            except OSError as e:
                logger.debug(
                    "Failed to delete local archive file %s: %s",
                    archive_file,
                    e,
                )

        except Exception as e:
            logger.warning(
                "Non-fatal failure while archiving and uploading "
                "the workspace: %s",
                e,
            )

    def archive(self) -> None:
        """Orchestrates archival of BoH and workspace using active provider."""
        if not os.path.exists(self.path) or not os.path.exists(
            os.path.join(self.path, ".ramble-workspace")
        ):
            logger.debug(
                "Workspace %s is not a valid Ramble workspace. "
                "Skipping archival.",
                self.path,
            )
            return

        try:
            from chs.storage import StorageProviderFactory

            provider = StorageProviderFactory.get_active_provider()
        except Exception as e:
            logger.debug(
                "Storage provider not configured or failed to load. "
                "Skipping archival: %s",
                e,
            )
            return

        try:
            self._upload_boh(provider)
        finally:
            self._upload_workspace(provider)

    def _cleanup_inputs_dir(self) -> None:
        """Deletes the inputs directory in the workspace."""
        inputs_dir = os.path.join(self.path, "inputs")
        if os.path.exists(inputs_dir):
            logger.debug("Deleting inputs directory: %s", inputs_dir)
            try:
                shutil.rmtree(inputs_dir)
            except OSError as e:
                logger.warning(
                    "Failed to delete inputs directory %s: %s", inputs_dir, e
                )

    def cleanup_workspace(self) -> None:
        # TODO: Implement the logic for this method.
        pass
