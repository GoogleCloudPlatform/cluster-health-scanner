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

"""Common logic for the cluster_diag CLI."""

from collections.abc import Callable
from typing import Any

SUPPORTED_ORCHESTRATORS = frozenset([
    'gke',
    'slurm',
])

SUPPORTED_MACHINE_TYPES = frozenset([
    'a3-highgpu-8g',
    'a3-megagpu-8g',
    'a3-ultragpu-8g',
    'a4-highgpu-8g',
    'a4x-highgpu-4g',
])


def run_for_orchestrator(
    orchestrator: str,
    gke_function: Callable[[], Any],
    slurm_function: Callable[[], Any],
) -> Any:
  """Run a function for a given orchestrator, or throw an error if unsupported.

  Args:
    orchestrator: The orchestrator to run the function for.
    gke_function: The function to run for a GKE orchestrator.
    slurm_function: The function to run for a Slurm orchestrator.

  Returns:
    The result of the function run.
  """
  match orchestrator:
    case 'gke':
      return gke_function()
    case 'slurm':
      return slurm_function()
    case _:
      raise ValueError(
          f'Unsupported orchestrator: {orchestrator}. Supported'
          f' orchestrators: {SUPPORTED_ORCHESTRATORS}'
      )
