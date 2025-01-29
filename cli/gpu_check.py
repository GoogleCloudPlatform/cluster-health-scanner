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

"""Check for running a DCGM test on a cluster."""

import check


class GpuCheck(check.Check):
  """A check that runs a DCGM test on a cluster."""

  _description = 'Runs a DCGM test on a cluster.'

  name = 'gpu'

  launch_label = 'aiinfra/gpu-healthcheck-test'

  results_labels = [
      'aiinfra/gpu-healthcheck-runtime-sec',
      'aiinfra/gpu-healthcheck-result',
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
        timeout_sec=5 * 60,
    )
