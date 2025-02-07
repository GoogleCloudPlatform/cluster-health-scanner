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

import common
import check
import gke_check
import slurm_check

NAME = 'nccl'
_DESCRIPTION = 'Runs a NCCL bandwidth test on a cluster.'


def get_check_for_orchestrator(
    orchestrator: str,
    machine_type: str,
    partition: str,
    nodes: list[str],
    run_only_on_available_nodes: bool,
    dry_run: bool = False,
) -> check.Check:
  """Returns the appropriate check for the given orchestrator."""
  match orchestrator:
    case 'gke':
      return GkeNcclCheck(
          machine_type=machine_type,
          nodes=nodes,
          run_only_on_available_nodes=run_only_on_available_nodes,
          dry_run=dry_run,
      )
    case 'slurm':
      return SlurmNcclCheck(
          machine_type=machine_type,
          partition=partition,
          nodes=nodes,
          dry_run=dry_run,
      )
    case _:
      raise ValueError(f'Unsupported orchestrator: {orchestrator}')


class GkeNcclCheck(gke_check.GkeCheck):
  """Runs a NCCL bandwidth test on a GKE cluster."""

  _SUPPORTED_MACHINE_TYPES = common.SUPPORTED_MACHINE_TYPES

  launch_label = 'aiinfra/nccl-healthcheck-test'

  results_labels = [
      'aiinfra/nccl-healthcheck-runtime-sec',
      'aiinfra/nccl-healthcheck-pre-result',
      'aiinfra/nccl-healthcheck-result',
      'aiinfra/nccl-healthcheck-bandwidth',
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


class SlurmNcclCheck(slurm_check.SlurmCheck):
  """A check that runs a DCGM test on a Slurm cluster."""

  _SUPPORTED_MACHINE_TYPES = common.SUPPORTED_MACHINE_TYPES

  _check_flag = 'nccl'

  def __init__(
      self,
      machine_type: str,
      partition: str,
      nodes: list[str],
      dry_run: bool = False,
  ):
    super().__init__(
        name=NAME,
        description=_DESCRIPTION,
        machine_type=machine_type,
        check_flag=self._check_flag,
        partition=partition,
        nodes=nodes,
        supported_machine_types=self._SUPPORTED_MACHINE_TYPES,
        dry_run=dry_run,
    )
