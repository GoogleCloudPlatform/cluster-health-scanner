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

"""A base class for defining the interface of healthscan checks."""

import abc
import signal
import sys
from typing import Any

import click


class Check(abc.ABC):
  """A standard implementation of a healthscan check."""

  def sigint_handler(
      self,
      signum: Any,
      frame: Any,
  ) -> None:
    """Handler for SIGINT signal.

    Args:
      signum: The signal number.
      frame: The current stack frame.
    """
    print(f'Received {signum} signal on frame {frame}. Exiting...')
    # Perform any necessary cleanup actions here
    # For example: close file handlers, release resources, etc.
    click.echo(
        click.style(
            '\nCLEANING UP...',
            fg='red',
            bold=True,
        )
    )
    self.clean_up()
    # Stops proceeding anything the CLI is doing
    sys.exit(0)

  def __init__(
      self,
      name: str,
      description: str,
      machine_type: str,
      supported_machine_types: frozenset[str],
      dry_run: bool = False,
  ):
    """Initialize a check to run on a cluster.

    Args:
      name: The name of the check.
      description: The description of the check.
      machine_type: The machine type of the cluster to run the check on.
      supported_machine_types: The machine types supported by the check.
      dry_run: Whether to run the check in dry run mode.

    Raises:
      click.Abort: If the machine type is not supported.
    """
    self.name = name
    self.description = description
    self.machine_type = machine_type
    self.supported_machine_types = supported_machine_types
    self.dry_run = dry_run

    # Handle SIGINT signal to clean up
    signal.signal(
        signal.SIGINT,
        self.sigint_handler,
    )

     # Validate machine type before running check
    if not self._is_supported_machine_type(self.machine_type):
      click.echo(
          click.style(
              text=(
                  f'`{self.name}` check '
                  f'does not support machine type {self.machine_type}.'
              ),
              fg='red',
              bold=True,
          )
      )
      raise click.Abort()

  def _is_supported_machine_type(self, machine_type: str) -> bool:
    """Returns whether Check supports given machine type.

    Args:
      machine_type: The machine type to validate.

    Returns:
      True if the Check supports the given machine type, False otherwise.
    """
    # Check explicitly stated supported machine types
    return machine_type in self.supported_machine_types

  @abc.abstractmethod
  def set_up(self):
    """Set up for the check on a cluster."""

  def clean_up(self) -> None:
    """Clean up after the check on a cluster."""

  @abc.abstractmethod
  def run(
      self,
      timeout_sec: int | None = None,
      startup_sec: int = 30,
  ) -> str | None:
    """Run the check.

    Args:
      timeout_sec: The timeout in seconds for the check.
      startup_sec: The time in seconds to wait for the health runner to start.

    Returns:
      The name of the health runner pod.
    """
