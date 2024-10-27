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

"""Class for fetching attributes from a VM.

The AttributeFetcher defines a pattern to be iterated over to fetch attributes
from a VM and store them in a DependencyConfiguration proto.
"""

import subprocess
from typing import Callable, Sequence

import vmconfig_pb2


class AttributeFetcher:
  """Class for fetching attributes from a VM."""

  def __init__(
      self,
      name: str,
      attr_version_fetch_cmd: Sequence[str],
      dependency_config_parser: Callable[
          [str], Sequence[vmconfig_pb2.DependencyConfiguration]
      ] | None = None,
  ):
    """Creates an AttributeFetcher object.

    Args:
      name: The name of the attribute to fetch.
      attr_version_fetch_cmd: The bash command to be executed on the VM to fetch
        the attribute value.
      dependency_config_parser: A function that takes the raw output of the
        command to fetch the attribute and returns a list of
        DependencyConfiguration protos.
    """
    self._name = name
    self._attr_version_fetch_cmd = attr_version_fetch_cmd
    self._dependency_config_parser = dependency_config_parser

  def _extract_dependency_config(
      self, raw_cmd_output: str
  ) -> Sequence[vmconfig_pb2.DependencyConfiguration]:
    """Builds a list of DependencyConfiguration protos for the attr.

    Defaults to a simple parser that returns a single DependencyConfiguration
    proto with the name and version fields set.

    Args:
      raw_cmd_output: The raw output of the command to fetch the attribute.

    Returns:
      A list of DependencyConfiguration protos for the attribute.
    """
    if self._dependency_config_parser is None:
      return [
          vmconfig_pb2.DependencyConfiguration(
              name=self._name, version=raw_cmd_output
          )
      ]
    else:
      return self._dependency_config_parser(raw_cmd_output)

  def fetch_attr(self) -> Sequence[vmconfig_pb2.DependencyConfiguration]:
    """Fetches the attribute from the VM and returns a sequence of protos.

    Returns:
      A sequence of DependencyConfiguration protos for the given attribute, as
      parsed by the dependency_config_parser.
    """
    raw_cmd_output = subprocess.check_output(
        args=self._attr_version_fetch_cmd
    ).decode('utf-8').strip()
    return self._extract_dependency_config(raw_cmd_output)
