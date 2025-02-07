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

"""Check for running a Network Performance test on a cluster."""

import common
import check
import gke_check

NAME = 'neper'
_DESCRIPTION = 'Runs a Network Performance test on a cluster.'


def get_check_for_orchestrator(
    orchestrator: str,
    machine_type: str,
    nodes: list[str],
    run_only_on_available_nodes: bool,
    dry_run: bool = False,
) -> check.Check:
  """Returns the appropriate check for the given orchestrator."""
  match orchestrator:
    case 'gke':
      return GkeNeperCheck(
          machine_type=machine_type,
          nodes=nodes,
          run_only_on_available_nodes=run_only_on_available_nodes,
          dry_run=dry_run,
      )
    case _:
      raise ValueError(f'Unsupported orchestrator: {orchestrator}')


class GkeNeperCheck(gke_check.GkeCheck):
  """Runs a Network Performance test on a cluster."""
  _SUPPORTED_MACHINE_TYPES = common.SUPPORTED_MACHINE_TYPES

  launch_label = 'aiinfra/neper-healthcheck-test'

  results_labels = [
      'aiinfra/neper-healthcheck-runtime-sec',
      'aiinfra/neper-healthcheck-result',
  ]

  def __init__(
      self,
      machine_type: str,
      nodes: list[str],
      run_only_on_available_nodes: bool = False,
      dry_run: bool = False,
      **kwargs,
  ):
    super().__init__(
        name=NAME,
        description=_DESCRIPTION,
        machine_type=machine_type,
        supported_machine_types=self._SUPPORTED_MACHINE_TYPES,
        launch_label=self.launch_label,
        results_labels=self.results_labels,
        nodes=nodes,
        run_only_on_available_nodes=run_only_on_available_nodes,
        timeout_sec=15 * 60,
        dry_run=dry_run,
        **kwargs,
    )
