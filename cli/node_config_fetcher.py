# Copyright 2024 Google LLC
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

"""For a given node, fetches the current configuration settings."""

import asyncio
import subprocess
from typing import Callable

import click

import config as configcheck_config
import dependency_version_parser


class NodeConfigFetcher:
  """Fetches the current configuration settings for a given node."""

  def __init__(
      self,
      name: str,
      project: str,
      dependency_parsers: list[
          dependency_version_parser.DependencyVersionParser
      ],
      zone: str,
      run_async: bool = False,
      sudo: bool = False,
      verbose: bool = False,
  ):
    """Creates a Config Fetcher for a given node.

    Args:
      name: The name of the node.
      project: The project of the node.
      dependency_parsers: A list of dependency parsers for the node.
      zone: The zone of the node.
      run_async: If true, run the fetcher in an async mode.
      sudo: Whether to run remote commands with sudo. Defaults to False.
      verbose: Whether to enable verbose logging. Defaults to False.
    """
    self.name = name
    self.zone = zone
    self.project = project
    self.dependency_parsers = dependency_parsers
    self.run_async = run_async
    self.sudo = sudo
    self.verbose = verbose

  def _get_remote_exec_cmd(
      self, cmd: list[str], sudo: bool = False
  ) -> list[str]:
    """Fetches the command result from the remote node."""
    remote_cmd = [
        'gcloud',
        'compute',
        'ssh',
        '--zone',
        f'{self.zone}',
        '--project',
        f'{self.project}',
        f'{self.name}',
        '--',
    ]
    if sudo:
      remote_cmd.append('sudo')
    remote_cmd.extend(cmd)
    return remote_cmd

  def _run_cmd_async(self, cmd: str):
    """Runs a bash command and returns the output."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
      raise subprocess.CalledProcessError(
          process.returncode, cmd, stderr, stdout
      )
    return stdout.decode('utf-8')

  def _run_cmd(self, cmd: str) -> str:
    """Runs a bash command and returns the output."""
    result = subprocess.run(
        cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
      raise subprocess.CalledProcessError(
          result.returncode, cmd, result.stderr, result.stdout
      )
    return result.stdout

  async def _fetch_config_internal(
      self,
      cmd_runner: Callable[[str], str],
  ) -> configcheck_config.NodeConfig:
    """Internal impl to fetch a given node configuration.

    Args:
      cmd_runner: A function to run a bash command and return the output.

    Returns:
      The node configuration.
    """
    node_config = configcheck_config.NodeConfig(name=self.name)
    for dependency_fetcher in self.dependency_parsers:
      if dependency_fetcher.local_exec:
        cmd = dependency_fetcher.cmd
      else:
        cmd = self._get_remote_exec_cmd(
            cmd=dependency_fetcher.cmd, sudo=self.sudo
        )
      cmd = ' '.join(cmd)
      if self.verbose:
        click.echo(
            _format_cmd_for_logging(
                dependency_name=dependency_fetcher.name, cmd=cmd
            )
        )
      try:
        cmd_output = await asyncio.to_thread(lambda cmd=cmd: cmd_runner(cmd))
        dependency_config = dependency_fetcher.parse_version(cmd_output)
      except subprocess.CalledProcessError:
        dependency_config = configcheck_config.DependencyConfig(
            name=dependency_fetcher.name,
            version='Error Fetching Dependency',
        )
      node_config.dependencies[dependency_fetcher.name] = dependency_config
    return node_config

  async def fetch_config_async(self) -> configcheck_config.NodeConfig:
    """Asynchronously fetches the current configuration settings for a given node."""
    return await self._fetch_config_internal(self._run_cmd_async)

  def fetch_config(self) -> configcheck_config.NodeConfig:
    """Fetches the current configuration settings for a given node."""
    return asyncio.run(self._fetch_config_internal(self._run_cmd))


def _format_cmd_for_logging(
    dependency_name: str, cmd: str, color: str = 'yellow'
) -> str:
  """Formats a command for logging."""
  return (
      click.style(
          'Fetching dependency ',
          fg=color,
      )
      + click.style(
          dependency_name,
          fg=color,
          bold=True,
      )
      + click.style(
          ' with command: ',
          fg=color,
      )
      + click.style(
          cmd,
          fg=color,
          bold=True,
      )
  )
