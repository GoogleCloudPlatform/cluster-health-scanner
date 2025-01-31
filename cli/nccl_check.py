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

"""Check for running a NCCL bandwidth test on a cluster."""

import gke_check


class NcclCheck(gke_check.GkeCheck):
  """Runs a NCCL bandwidth test on a cluster."""

  _description = 'Runs a NCCL bandwidth test on a cluster.'

  name = 'nccl'

  launch_label = 'aiinfra/nccl-healthcheck-test'

  results_labels = [
      'aiinfra/nccl-healthcheck-runtime-sec',
      'aiinfra/nccl-healthcheck-pre-result',
      'aiinfra/nccl-healthcheck-result',
      'aiinfra/nccl-healthcheck-bandwidth',
  ]

  def __init__(self, machine_type: str, nodes: list[str]):
    super().__init__(
        name=self.name,
        description=self._description,
        machine_type=machine_type,
        launch_label=self.launch_label,
        results_labels=self.results_labels,
        nodes=nodes,
        timeout_sec=15 * 60,
    )
