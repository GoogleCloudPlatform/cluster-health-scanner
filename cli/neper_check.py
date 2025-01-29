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

import check


class NeperCheck(check.Check):
  """Runs a Network Performance test on a cluster."""

  _description = 'Runs a Network Performance test on a cluster.'

  name = 'neper'

  launch_label = 'aiinfra/neper-healthcheck-test'

  results_labels = [
      'aiinfra/neper-healthcheck-runtime-sec',
      'aiinfra/neper-healthcheck-result',
  ]

  def __init__(self, orchestrator: str, machine_type: str, nodes: list[str]):
    super().__init__(
        name=self.name,
        description=self._description,
        orchestrator=orchestrator,
        machine_type=machine_type,
        launch_label=self.launch_label,
        results_labels=self.results_labels,
        nodes=nodes,
        timeout_sec=10 * 60,
    )
