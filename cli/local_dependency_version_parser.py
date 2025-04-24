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

"""An implementation of DependencyVersionParser that runs an SCP command on the local machine."""

from collections.abc import Callable

import config
import dependency_version_parser

LOCAL_FILE_PATH = '~/.local/share/cluster_diag/logs'


class LocalDependencyVersionParser(
    dependency_version_parser.DependencyVersionParser
):
  """A DependencyVersionParser that runs the command locally."""

  def __init__(
      self,
      dep_name: str,
      node_name: str,
      zone: str,
      remote_file_path: str,
      parse_version_fn: Callable[[str, str], config.DependencyConfig],
  ):
    """Initializes the LocalDependencyVersionParser.

    Args:
      dep_name: The name of the dependency to fetch the version of.
      node_name: The name of the node to fetch the version from.
      zone: The zone of the node to fetch the version from.
      remote_file_path: The path to the file on the node to fetch.
      parse_version_fn: Required. The function to use to parse the version of
        the dependency.
    """
    super().__init__(
        name=dep_name,
        cmd=_get_scp_cmd(
            node_name,
            zone,
            remote_file_path,
            f'{LOCAL_FILE_PATH}/{node_name}',
        ),
        parse_version_fn=parse_version_fn,
        local_exec=True,
    )
    self.node_name = node_name
    self.zone = zone
    self.remote_file_path = remote_file_path


def _get_scp_cmd(
    node_name: str, zone: str, remote_file_path: str, local_file_path: str
) -> list[str]:
  """Returns the scp command to download the file from the remote node."""
  return [
      'mkdir',
      '-p',
      local_file_path,
      '&&',
      'gcloud',
      'compute',
      'scp',
      '--tunnel-through-iap',
      f'{node_name}:{remote_file_path}',
      local_file_path,
      f'--zone={zone}',
  ]
