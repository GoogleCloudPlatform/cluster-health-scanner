"""Manages Ramble workspaces for Cluster Health Scanner runs.

This module provides the RambleManager class, which encapsulates
the creation, setup, and execution of Ramble workspaces for CHS test runs.
"""

from collections.abc import Callable
import datetime
import importlib.resources as pkg_resources
import json
import logging
import os
import pathlib
import re
import shutil

import subprocess
import sysconfig
import time
from typing import Any
import click
from google.protobuf.json_format import MessageToJson

from chs import exceptions as ex
from chs.boh_generator import BOHGenerator
from chs.schedulers import get_scheduler
from chs.schedulers.base import BaseScheduler
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
    ) -> None:
        """Initializes the RambleManager.

        Args:
          base_dir: The base directory where Ramble workspaces will be created.
          workspace_name: A unique name for this CHS workspace.
          run_command: The function to use for running subprocess commands.
            Defaults to `subprocess.run`.

        Raises:
          click.UsageError: If the Ramble setup script is not found.
          OSError: If the workspace directory cannot be created.
        """

        self._base_dir = base_dir
        self._workspace_name = workspace_name
        self._path = os.path.join(self._base_dir, self._workspace_name)
        self._app_data = ChsAppData()

        # Ensure the directory exists for logging and execution
        os.makedirs(self._path, exist_ok=True)

        logger.debug(
            "Initialized manager with workspace_name='%s', path='%s'",
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
            f"RambleManager(workspace_name='{self._workspace_name}',"
            f" path='{self._path}')"
        )

    @property
    def path(self) -> str:
        """The absolute path to the Ramble workspace."""
        return self._path

    def _prepare_ramble_command(self, cmd: list[str]) -> list[str]:
        """Prepares a Ramble command by sourcing setup script."""
        command_str = f'source {self._ramble_setup_script} && exec "$@"'
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
          The CompletedProcess object from subprocess.run.

        Raises:
          click.ClickException: If the command fails when check=True.
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

                process = subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=run_env,
                    capture_output=True,
                    # We use text=True to get strings instead of bytes
                    text=True,
                    check=False,  # We handle check manually
                )

                if process.stdout:
                    log_file.write(process.stdout)

                if process.stderr:
                    log_file.write(process.stderr)

                if check and process.returncode != 0:
                    msg = (
                        f"Command failed with exit code {process.returncode}. "
                        f"See {log_file_path} for details."
                    )
                    if error_context:
                        msg = f"{error_context}. {msg}"
                    raise ex.ChsException(msg)

                return process

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
          check: If True, raises a ChsException if the command returns a
            non-zero exit code.

        Returns:
          The CompletedProcess object from subprocess.run.

        Raises:
          click.ClickException: If the Ramble command fails.
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

    def add_custom_repository(self) -> None:
        """Adds the internal custom CHS repository to the Ramble workspace.

        Raises:
          click.ClickException: If the repository addition command fails.
        """
        import chs

        logger.debug("Adding custom CHS repository to workspace...")
        with pkg_resources.path(chs, "repos") as p:
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

        This runs 'ramble on' to start the experiments and captures Job IDs.

        Returns:
          A list of submitted Job IDs.

        Raises:
          click.ClickException: If the experiment execution command fails.
        """

        logger.debug("Executing experiments in %s...", self.path)
        output = self._execute_ramble_command(
            subcommand="on",
            args=[],
            capture_output=True,
            env={"TQDM_DISABLE": "1", "TERM": "dumb"},
            error_context="Failed to execute experiments",
        ).stdout

        job_ids = self._scheduler.parse_job_ids(output)
        logger.debug("Captured Job IDs: %s", job_ids)

        self._update_registry(job_ids)
        return job_ids

    def cancel_experiments(self) -> None:
        """Cancels running experiments in the Ramble workspace.

        This retrieves job IDs from the registry and cancels them via the
        scheduler.

        Raises:
          click.ClickException: If the experiment cancellation command fails.
        """
        registry_entry = self._app_data.get_run(self._workspace_name)
        if not registry_entry:
            logger.warning("No registry entry found. Cannot cancel jobs.")
            click.echo(
                f"No registry entry found for run {self._workspace_name}."
            )
            return

        job_ids = self.get_job_ids()
        if not job_ids:
            logger.warning("No job IDs found in registry. Cannot cancel jobs.")
            click.echo(f"No job IDs found for run {self._workspace_name}.")
            return

        scheduler_name = registry_entry.get("scheduler", "slurm")
        logger.debug("Using scheduler '%s' for canceling jobs.", scheduler_name)
        scheduler = get_scheduler(scheduler_name)

        logger.debug("Canceling experiments %s in %s...", job_ids, self.path)
        scheduler.cancel_jobs(job_ids)

    def _update_registry(self, job_ids: list[str]) -> None:
        """Updates the runs registry with the current run information."""
        logger.debug(
            "Updating registry for run %s with job_ids %s",
            self._workspace_name,
            job_ids,
        )
        run_entry = {
            "run_id": self._workspace_name,
            "workspace_path": self.path,
            "job_entries": {
                job_id: {"status": JobStatus.PENDING.value}
                for job_id in job_ids
            },
            "submission_time": (
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            ),
            "scheduler": self._scheduler.name,
        }
        self._app_data.update_run(self._workspace_name, run_entry)

    def get_job_ids(self) -> list[str]:
        """Retrieves job IDs for this workspace from the registry."""
        entry = self._app_data.get_run(self._workspace_name)
        if entry:
            return list(entry.get("job_entries", {}).keys())
        return []

    def check_jobs_running(self) -> bool:
        """Checks if any jobs for this workspace are still running."""
        registry_entry = self._app_data.get_run(self._workspace_name)
        if not registry_entry:
            logger.debug("No registry entry found for checking.")
            return False

        job_ids = self.get_job_ids()
        if not job_ids:
            logger.debug("No job IDs found for checking.")
            return False

        scheduler_name = registry_entry.get("scheduler", "slurm")
        logger.debug("Using scheduler '%s' for checking jobs.", scheduler_name)
        scheduler = get_scheduler(scheduler_name)

        return scheduler.check_jobs_running(job_ids)

    def _map_jobs_to_log_paths(self) -> dict[str, str]:
        """Maps Job IDs to their expected log paths on the filesystem.

        Scans the 'experiments' directory for jobset.yaml files, extracts the
        job name (which is the Job ID), and maps it to the '{experiment}.out'
        file in the same directory. Matches against registry jobs.
        """
        registry_job_ids = set(self.get_job_ids())
        mapping = {}
        experiments_dir = os.path.join(self.path, "experiments")

        if not os.path.exists(experiments_dir):
            return mapping

        for root, _, files in os.walk(experiments_dir):
            if "jobset.yaml" in files:
                try:
                    with open(os.path.join(root, "jobset.yaml"), "r") as f:
                        content = f.read()

                        # Parse all docs (e.g. Service + JobSet) to find match.
                        docs = list(yaml.safe_load_all(content))
                        job_name = None
                        for doc in docs:
                            if not doc:
                                continue
                            name = doc.get("metadata", {}).get("name")
                            if name and name in registry_job_ids:
                                job_name = name
                                break

                        if job_name and job_name in registry_job_ids:
                            # Find .out file (ignoring execute.log).
                            out_files = [
                                f
                                for f in files
                                if f.endswith(".out")
                                and f != "execute.log"
                                and not f.startswith("execute")
                            ]

                            if not out_files:
                                logger.warning(
                                    "No .out file found in %s for job %s",
                                    root,
                                    job_name,
                                )
                                continue

                            # Priority 1: File containing job_name
                            log_path = None
                            for out_file in out_files:
                                if job_name in out_file:
                                    log_path = os.path.join(root, out_file)
                                    break

                            # Priority 2: Any other .out file (fallback)
                            if not log_path:
                                log_path = os.path.join(root, out_files[0])

                            if log_path:
                                mapping[job_name] = log_path
                except Exception as e:
                    logger.warning(
                        "Failed to parse jobset.yaml in %s: %s", root, e
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
        logger.info(
            "Waiting for jobs to complete for run %s...", self._workspace_name
        )
        self._monitor_jobs(app_data, poll_interval)

    def _monitor_jobs(self, app_data: ChsAppData, poll_interval: int) -> None:
        """Generic wait logic to monitor job progress and fetch logs.

        Polls job statuses for all jobs in the registry. When a job completes,
        attempts to fetch its log (if a path is mapped). Logs progress.
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

        while active_jobs:
            statuses = scheduler.get_job_statuses(list(active_jobs))

            still_running = False
            for job_id, status in statuses.items():
                if JobStatus.is_terminal(status):
                    # Job finished
                    if job_id in pending_log_fetches:
                        log_path = job_log_map.get(job_id)
                        if log_path:
                            try:
                                scheduler.fetch_job_log(job_id, log_path)
                            except Exception as e:
                                logger.error(
                                    "Error fetching log for %s: %s", job_id, e
                                )
                        pending_log_fetches.remove(job_id)

                    active_jobs.remove(job_id)
                else:
                    still_running = True

            try:
                app_data.update_run_status(self._workspace_name, statuses)
            except Exception as e:
                logger.warning("Failed to update run status in registry: %s", e)

            if still_running:
                logger.info(
                    "Jobs remaining: %d. Waiting %s seconds...",
                    len(active_jobs),
                    poll_interval,
                )
                time.sleep(poll_interval)

        logger.info(
            "All jobs completed and logs fetched for run %s.",
            self._workspace_name,
        )

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

    def analyze_results(self) -> str:
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
        return self.generate_boh(results_json_path)

    def generate_boh(self, results_json_path: str) -> str:
        """Generates Bill of Health from results.latest.json.

        Args:
          results_json_path: Path to the results.latest.json file.

        Returns:
          The path to the directory containing the analysis results and Bill of
          Health.
        """

        workspace_dir = pathlib.Path(self._path)

        boh_generator = BOHGenerator(
            results_json_path,
            base_log_dir=str(workspace_dir),
            cluster_type=self.get_scheduler_type(),
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
            logger.debug("Cleaned up jobs for run %s", self._workspace_name)
        except Exception as e:
            logger.warning("Failed to cleanup jobs after analysis: %s", e)

        return str(workspace_dir)

    def archive_and_upload_boh(self) -> None:
        """Archives Bill of Health and uploads to GCS."""
        boh_path = os.path.join(self.path, "bill_of_health.json")
        if not os.path.exists(boh_path):
            logger.warning(
                "Bill of Health file %s does not exist. Skipping BOH archival.",
                boh_path,
            )
            return

        bucket_name = self._app_data.get_bucket()
        if not bucket_name:
            logger.warning("No GCS bucket configured. Skipping BOH archival.")
            return

        upload_url = (
            f"gs://{bucket_name}/bills_of_health/{self._workspace_name}.json"
        )
        logger.info("Uploading Bill of Health to %s", upload_url)

        try:
            self._execute_shell_command(
                ["gcloud", "storage", "cp", boh_path, upload_url],
                error_context="Failed to upload Bill of Health",
            )
            logger.info("Bill of Health uploaded successfully.")
        except Exception as e:
            raise ex.ChsArchiveException(
                f"Unexpected error during BOH archival: {e}"
            ) from e

    def archive_and_upload_workspace(self) -> None:
        """Archives workspace artifacts and uploads to GCS."""
        if not os.path.exists(self.path):
            logger.warning(
                "Workspace path %s does not exist. Skipping archival.",
                self.path,
            )
            return

        bucket_name = self._app_data.get_bucket()
        if not bucket_name:
            logger.warning(
                "No GCS bucket configured. Skipping workspace archival."
            )
            return

        upload_url = (
            f"gs://{bucket_name}/workspaces/{self._workspace_name}.tar.gz"
        )
        logger.info("Archiving workspace to %s", upload_url)

        try:
            # TODO: Add gcloud pre-flight check
            self._execute_ramble_command(
                "workspace",
                [
                    "archive",
                    "-t",
                    "--upload-url",
                    upload_url,
                    "--archive-pattern",
                    "bill_of_health.json",
                    "--archive-pattern",
                    "internal.log",
                ],
                capture_output=True,
                error_context="Failed to archive workspace",
            )
            logger.info("Workspace archived and uploaded successfully.")

        except (ex.ChsException, Exception) as e:
            logger.warning(
                "Non-fatal failure while archiving and uploading "
                "the workspace: %s",
                e,
            )

    def archive(self) -> None:
        """Orchestrates archival and upload of BoH and workspace."""
        try:
            self.archive_and_upload_boh()
        finally:
            self.archive_and_upload_workspace()

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
