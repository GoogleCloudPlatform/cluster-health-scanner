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

"""Represents a Dependency of a node, and provides methods to parse the version of the dependency."""

from collections.abc import Callable

import config


class DependencyVersionParser:
  """Fetches the version of a given node."""

  def __init__(
      self,
      name: str,
      cmd: list[str],
      parse_version_fn: Callable[
          [str, str], config.DependencyConfig
      ] = lambda name, result: config.DependencyConfig(
          name=name, version=result
      ),
      local_exec: bool = False,
  ):
    """Initializes the DependencyVersionParser.

    Args:
      name: The name of the dependency to fetch the version of.
      cmd: The command to run to fetch the version of the dependency.
      parse_version_fn: Optional. The function to use to parse the version of
        the dependency. If not provided, defaults to the return value of cmd.
      local_exec: Optional. Execute command on local machine. Defaults to remote
        execution (False).
    """
    self.name = name
    self.cmd = cmd
    self._parse_version_fn = parse_version_fn
    self.local_exec = local_exec

  def parse_version(self, cmd_result: str) -> config.DependencyConfig:
    """Fetches the version of the dependency.

    Args:
      cmd_result: The command result to parse the version from.

    Returns:
      The version of the dependency.
    """
    return self._parse_version_fn(self.name, cmd_result)
