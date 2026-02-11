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
import shutil
import subprocess
import sysconfig
from typing import Any
import chs
from chs.boh_generator import BOHGenerator
from chs.schedulers import get_scheduler
from chs.schedulers.base import BaseScheduler
import click
from google.protobuf.json_format import MessageToJson
import platformdirs

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
      run_command: The function to use for running subprocess commands. Defaults
        to `subprocess.run`.

    Raises:
      click.UsageError: If the Ramble setup script is not found.
    """
    self._base_dir = base_dir
    self._workspace_name = workspace_name
    self._path = os.path.join(self._base_dir, self._workspace_name)
    self._run_command = run_command
    logger.debug(
        "Initialized RambleManager with workspace_name='%s', path='%s'",
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
      raise click.UsageError(
          "Please ensure CHS is installed correctly."
      )

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
        "Sourcing %s before running ramble command.",
        self._ramble_setup_script,
    )
    return ["bash", "-c", command_str, "bash"] + cmd

  def _execute_shell_command(
      self,
      cmd: list[str],
      check: bool = True,
      cwd: str | None = None,
      capture_output: bool = False,
  ) -> subprocess.CompletedProcess[str]:
    """Executes a command with consistent logging and error handling.

    Args:
      cmd: The command to execute as a list of strings.
      check: If True, raises a CalledProcessError if the command returns a
        non-zero exit code.
      cwd: The current working directory to run the command in.
      capture_output: If True, captures stdout and stderr.

    Returns:
      The CompletedProcess object from subprocess.run.

    Raises:
      click.ClickException: If the command fails when check=True.
    """
    logger.debug("Running command: %s", cmd)
    try:
      return self._run_command(
          cmd,
          check=check,
          text=True,
          cwd=cwd,
          stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
          stderr=subprocess.PIPE if capture_output else None,
      )
    except subprocess.CalledProcessError as e:
      logger.error("Command failed: %s", cmd)
      logger.debug("Command stdout: %s", e.stdout)
      logger.debug("Command stderr: %s", e.stderr)
      raise click.ClickException(
          f"Command '{' '.join(cmd)}' failed with exit code {e.returncode}."
      )

  def _execute_ramble_command(
      self,
      subcommand: str,
      args: list[str],
      capture_output: bool = False,
  ) -> subprocess.CompletedProcess[str]:
    """Runs a Ramble subcommand in the workspace context.

    This is a wrapper around _execute_shell_command that prepends
    'ramble -w self.path' to the command and sets cwd to self.path.

    Args:
      subcommand: The Ramble subcommand to execute (e.g., 'workspace', 'on').
      args: A list of arguments for the Ramble subcommand.
      capture_output: Whether to capture stdout/stderr.

    Returns:
      The CompletedProcess object from subprocess.run.

    Raises:
      click.ClickException: If the Ramble command fails.
    """
    cmd = ["ramble", "-w", self.path, subcommand] + args
    exec_cmd = self._prepare_ramble_command(cmd)
    return self._execute_shell_command(
        exec_cmd, check=True, cwd=self.path, capture_output=capture_output
    )

  def create_workspace(self) -> None:
    """Creates the Ramble workspace directory and initializes it.

    Raises:
      click.ClickException: If the workspace creation command fails.
    """
    logger.debug("Creating Ramble workspace at: %s", self.path)
    cmd = ["ramble", "workspace", "create", "-a", "-d", self.path]
    exec_cmd = self._prepare_ramble_command(cmd)
    self._execute_shell_command(exec_cmd)

  def add_custom_repository(self) -> None:
    """Adds the internal custom CHS repository to the Ramble workspace.

    Raises:
      click.ClickException: If the repository addition command fails.
    """
    logger.debug("Adding custom CHS repository to Ramble workspace...")
    with pkg_resources.path(chs, "repos") as p:
      self._execute_ramble_command(subcommand="repo", args=["add", str(p)])

  def configure_workspace(
      self,
      suite: str,
      scheduler: BaseScheduler,
      tests: list[str] | None,
      machine_type: str,
  ) -> None:
    """Configures the Ramble workspace for the CHS run.

    Args:
      suite: The name of the test suite to run.
      scheduler: The scheduler plugin to use (e.g., SlurmScheduler).
      tests: An optional list of names of specific tests to run.
      machine_type: The machine type.
    """
    self._scheduler = scheduler

    logger.debug("Configuring Ramble workspace in %s...", self.path)
    logger.debug("Configuring workspace for suite: %s", suite)
    logger.debug("Configuring workspace for scheduler: %s", scheduler.name)
    if tests:
      logger.debug("Configuring workspace for tests: %s", tests)

    workspace_config_dir = os.path.join(self.path, "configs")
    os.makedirs(workspace_config_dir, exist_ok=True)

    scheduler.write_ramble_config(
        workspace_config_dir=workspace_config_dir,
        suite=suite,
        tests=tests,
        machine_type=machine_type,
    )

  def setup_workspace(self) -> None:
    """Sets up the Ramble workspace by running 'ramble workspace setup'.

    Raises:
      click.ClickException: If the workspace setup command fails.
    """
    logger.debug("Setting up Ramble workspace in %s...", self.path)
    self._execute_ramble_command(subcommand="workspace", args=["setup"])

  def execute_experiments(self) -> list[str]:
    """Executes experiments in the Ramble workspace.

    This runs 'ramble on' to start the experiments and captures Job IDs.

    Returns:
      A list of submitted Job IDs.

    Raises:
      click.ClickException: If the experiment execution command fails.
    """
    logger.debug("Executing Ramble experiments in %s...", self.path)
    output = self._execute_ramble_command(
        subcommand="on", args=[], capture_output=True
    ).stdout

    job_ids = self._scheduler.parse_job_ids(output)
    logger.debug("Captured Job IDs: %s", job_ids)

    self._update_registry(job_ids)
    return job_ids

  def _get_registry_path(self) -> pathlib.Path:
    """Returns the path to the runs registry file."""
    app_data_dir = pathlib.Path(
        platformdirs.user_data_dir(appname="cluster-health-scanner")
    )
    app_data_dir.mkdir(parents=True, exist_ok=True)
    return app_data_dir / "runs.json"

  def _update_registry(self, job_ids: list[str]) -> None:
    """Updates the runs registry with the current run information."""
    registry_path = self._get_registry_path()
    logger.debug(
        "Updating registry at %s for run %s with job_ids %s",
        registry_path,
        self._workspace_name,
        job_ids,
    )
    run_entry = {
        "run_id": self._workspace_name,
        "workspace_path": self.path,
        "job_ids": job_ids,
        "submission_time": (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        ),
        "scheduler": self._scheduler.name,
    }

    try:
      if registry_path.exists():
        with open(registry_path, "r") as f:
          registry = json.load(f)
      else:
        registry = {}

      registry[self._workspace_name] = run_entry

      with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
      logger.debug("Updated registry at %s", registry_path)
    except (OSError, json.JSONDecodeError) as e:
      logger.warning("Failed to update runs registry: %s", e)

  def get_job_ids(self) -> list[str]:
    """Retrieves job IDs for this workspace from the registry."""
    registry_path = self._get_registry_path()
    try:
      if registry_path.exists():
        with open(registry_path, "r") as f:
          registry = json.load(f)
        entry = registry.get(self._workspace_name)
        if entry:
          return entry.get("job_ids", [])
    except (OSError, json.JSONDecodeError) as e:
      logger.warning("Failed to read runs registry: %s", e)
    return []

  def check_jobs_running(self) -> bool:
    """Checks if any jobs for this workspace are still running."""
    registry_entry = self._get_registry_entry()
    if not registry_entry:
      logger.debug("No registry entry found for checking.")
      return False

    job_ids = registry_entry.get("job_ids", [])
    if not job_ids:
      logger.debug("No job IDs found for checking.")
      return False

    # TODO: Make this dynamic based on the config
    scheduler_name = registry_entry.get("scheduler", "slurm")
    logger.debug("Using scheduler '%s' for checking jobs.", scheduler_name)
    scheduler = get_scheduler(scheduler_name)

    return scheduler.check_jobs_running(job_ids)

  def _get_registry_entry(self) -> dict[str, Any] | None:
    """Retrieves the registry entry for this workspace."""
    registry_path = self._get_registry_path()
    try:
      if registry_path.exists():
        with open(registry_path, "r") as f:
          registry = json.load(f)
        return registry.get(self._workspace_name)
    except (OSError, json.JSONDecodeError) as e:
      logger.warning("Failed to read runs registry: %s", e)
    return None

  def analyze_results(self) -> str:
    """Analyzes and stores the results of the Ramble experiments.

    Analyzes the results and stores BoH in GCS/ BigQuery for later retrieval.

    Returns:
      The path to the directory containing the analysis results and Bill of
      Health.

    Raises:
      click.ClickException: If the experiment analyze command fails.
    """
    logger.debug("Analyzing Ramble experiments in %s...", self.path)
    self._execute_ramble_command(
        subcommand="workspace",
        args=["analyze", "-f", "json"],
        capture_output=True,
    ).stdout
    results_json_path = os.path.join(self.path, "results.latest.json")

    if not os.path.exists(results_json_path):
      raise click.ClickException(
          "chs analyze command failed. Recheck the run id and try rerunning"
          " the command."
      )

    home_dir = pathlib.Path(os.getcwd())
    output_dir = home_dir / self._workspace_name
    output_dir.mkdir(parents=True, exist_ok=True)

    boh_generator = BOHGenerator(
        results_json_path, base_log_dir=str(output_dir)
    )
    logger.debug("Generating Bill of Health from %s", results_json_path)
    boh = boh_generator.generate()
    logger.debug("Bill of Health generated successfully.")

    boh_path = output_dir / "bill_of_health.json"
    with open(boh_path, "w") as f:
      f.write(MessageToJson(boh, preserving_proto_field_name=True))

    experiments_dir = os.path.join(self.path, "experiments")
    if os.path.exists(experiments_dir):
      shutil.copytree(
          experiments_dir,
          output_dir / "experiments",
          dirs_exist_ok=True,
          ignore=shutil.ignore_patterns("chs-*"),
      )
      logger.debug(
          "Copied experiments from %s to %s",
          experiments_dir,
          output_dir / "experiments",
      )
    else:
      logger.debug("No experiments directory found at %s", experiments_dir)

    return str(output_dir)

  def cleanup_workspace(self) -> None:
    """Cleans up the Ramble workspace directory."""
    # TODO: Implement the logic for this method.
    pass
