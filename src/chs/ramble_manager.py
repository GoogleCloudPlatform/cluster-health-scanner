"""Manages Ramble workspaces for Cluster Health Scanner runs.

This module provides the RambleManager class, which encapsulates
the creation, setup, and execution of Ramble workspaces for CHS test runs.
"""

from collections.abc import Callable
import importlib.resources as pkg_resources
import logging
import os
import shutil
import subprocess
import sysconfig
import chs
import click

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
      self, cmd: list[str], check: bool = True, cwd: str | None = None
  ) -> subprocess.CompletedProcess[str]:
    """Executes a command with consistent logging and error handling.

    Args:
      cmd: The command to execute as a list of strings.
      check: If True, raises a CalledProcessError if the command returns a
        non-zero exit code.
      cwd: The current working directory to run the command in.

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
          stdout=subprocess.DEVNULL,
      )
    except subprocess.CalledProcessError as e:
      logger.error("Command failed: %s", cmd)
      raise click.ClickException(
          f"Command '{' '.join(cmd)}' failed with exit code {e.returncode}."
      ) from e

  def _execute_ramble_command(
      self, subcommand: str, args: list[str]
  ) -> subprocess.CompletedProcess[str]:
    """Runs a Ramble subcommand in the workspace context.

    This is a wrapper around _execute_shell_command that prepends
    'ramble -w self.path' to the command and sets cwd to self.path.

    Args:
      subcommand: The Ramble subcommand to execute (e.g., 'workspace', 'on').
      args: A list of arguments for the Ramble subcommand.

    Returns:
      The CompletedProcess object from subprocess.run.

    Raises:
      click.ClickException: If the Ramble command fails.
    """
    cmd = ["ramble", "-w", self.path, subcommand] + args
    exec_cmd = self._prepare_ramble_command(cmd)
    return self._execute_shell_command(exec_cmd, check=True, cwd=self.path)

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
      self, suite: str, scheduler: str, tests: list[str] | None
  ) -> None:
    """Configures the Ramble workspace for the CHS run.

    Args:
      suite: The name of the test suite to run.
      scheduler: The scheduler to use (e.g., slurm, gke).
      tests: An optional list of names of specific tests to run.
    """

    logger.debug("Configuring Ramble workspace in %s...", self.path)
    logger.debug("Configuring workspace for suite: %s", suite)
    logger.debug("Configuring workspace for scheduler: %s", scheduler)
    if tests:
      logger.debug("Configuring workspace for tests: %s", tests)

    workspace_config_dir = os.path.join(self.path, "configs")

    with pkg_resources.path(chs, "ramble.yaml") as p:
      shutil.copy(p, os.path.join(workspace_config_dir, "ramble.yaml"))
      logger.debug("Copied ramble.yaml to %s", workspace_config_dir)

  def setup_workspace(self) -> None:
    """Sets up the Ramble workspace by running 'ramble workspace setup'.

    Raises:
      click.ClickException: If the workspace setup command fails.
    """
    logger.debug("Setting up Ramble workspace in %s...", self.path)
    self._execute_ramble_command(subcommand="workspace", args=["setup"])

  def execute_experiments(self) -> None:
    """Executes experiments in the Ramble workspace.

    This runs 'ramble on' to start the experiments.

    Raises:
      click.ClickException: If the experiment execution command fails.
    """
    logger.debug("Executing Ramble experiments in %s...", self.path)
    self._execute_ramble_command(subcommand="on", args=[])

  def analyze_results(self) -> None:
    """Analyzes and stores the results of the Ramble experiments.

    Analyzes the results and stores BoH in GCS/ BigQuery for later retrieval.

    Raises:
      click.ClickException: If the experiment analyze command fails.
    """
    # TODO: Bill of health generation goes here.
    logger.debug("Analyzing Ramble experiments in %s...", self.path)
    self._execute_ramble_command(subcommand="workspace", args=["analyze"])

  def cleanup_workspace(self) -> None:
    """Cleans up the Ramble workspace directory."""
    # TODO: Implement the logic for this method.
    pass
