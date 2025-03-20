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

"""Defines data objects to represent the configuration settings for a node."""

import collections
import dataclasses


@dataclasses.dataclass
class DependencyConfig:
  """Represents the configuration settings for a single dependency."""

  name: str
  version: str
  config_settings: dict[str, str] | None = None


@dataclasses.dataclass
class NodeConfig:
  """Represents the configuration settings for a node."""

  name: str
  dependencies: dict[str, DependencyConfig] = dataclasses.field(
      default_factory=collections.defaultdict
  )

  def to_csv(self) -> list[str]:
    """Returns the node diff as a CSV string."""
    csv_rows = [self.name]
    sorted_deps = dict(sorted(self.dependencies.items()))
    for dep in sorted_deps.values():
      csv_rows.append(f"{dep.version}")
    return csv_rows


@dataclasses.dataclass
class DependencyDiff:
  """Represents the difference between two dependency configs."""
  DEFAULT_DIFF_EXPLANATION = "No Diff!"
  name: str
  diff_explanation: str = DEFAULT_DIFF_EXPLANATION
  experiment_dependency: DependencyConfig | None = None
  golden_dependency: DependencyConfig | None = None


@dataclasses.dataclass
class NodeDiff:
  """Represents the difference between two node configs."""

  name: str
  dependency_diffs: dict[str, DependencyDiff] = dataclasses.field(
      default_factory=collections.defaultdict
  )

  def to_csv(self) -> list[str]:
    """Returns the node diff as a CSV string."""
    csv_rows = [self.name]
    sorted_diffs = dict(sorted(self.dependency_diffs.items()))
    for dependency_diff in sorted_diffs.values():
      csv_rows.append(f"{dependency_diff.diff_explanation}")
    return csv_rows
