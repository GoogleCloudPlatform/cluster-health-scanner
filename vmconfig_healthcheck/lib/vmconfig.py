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

"""Library for helper functions supporting VMConfigs.

Helper functions can include:
  - Building VMConfigurations
  - Providing commands to fetch dependency versions
  - Helper functions to parse outputs.
"""

from typing import Sequence

import attribute_fetcher
import vmconfig_pb2


def build_vm_config(
    attributes: Sequence[attribute_fetcher.AttributeFetcher],
) -> vmconfig_pb2.NodeConfiguration:
  """Builds a VMConfig proto from a list of AttributeFetchers.

  Args:
    attributes: A list of AttributeFetchers to use to build the VMConfig proto.

  Returns:
    A VMConfig proto built from the list of AttributeFetchers.
  """
  # Hostname is a special case for attributes, so we explicitly handle it here.
  hostname = (
      attribute_fetcher.AttributeFetcher(
          name="hostname",
          attr_version_fetch_cmd=["hostname"],
      )
      .fetch_attr()[0]
      .version
  )

  configurations = []
  for attr in attributes:
    configurations.extend(attr.fetch_attr())

  return vmconfig_pb2.NodeConfiguration(
      hostname=hostname,
      configurations=configurations,
  )
