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

import yaml

import common
import check
import gke_check
import slurm_check

NAME = 'gpu'
_DESCRIPTION = 'Runs a DCGM test on a cluster.'


def get_check_for_orchestrator(
    orchestrator: str,
    machine_type: str,
    partition: str,
    nodes: list[str],
    run_only_on_available_nodes: bool,
    dry_run: bool = False,
    gcs_bucket_name: str | None = None,
) -> check.Check:
  """Returns the appropriate check for the given orchestrator."""
  match orchestrator:
    case 'gke':
      return GkeGpuCheck(
          machine_type=machine_type,
          nodes=nodes,
          run_only_on_available_nodes=run_only_on_available_nodes,
          dry_run=dry_run,
          gcs_bucket_name=gcs_bucket_name,
      )
    case 'slurm':
      return SlurmGpuCheck(machine_type, partition, nodes, dry_run=dry_run)
    case _:
      raise ValueError(f'Unsupported orchestrator: {orchestrator}')


class GkeGpuCheck(gke_check.GkeCheck):
  """A check that runs a DCGM test on a cluster."""

  _SUPPORTED_MACHINE_TYPES = common.SUPPORTED_MACHINE_TYPES

  launch_label = 'aiinfra/gpu-healthcheck-test'

  test_result_label = 'aiinfra/gpu-healthcheck-result'

  results_labels = [
      'aiinfra/gpu-healthcheck-runtime-sec',
      test_result_label,
  ]

  def __init__(
      self,
      machine_type: str,
      nodes: list[str],
      run_only_on_available_nodes: bool = False,
      dry_run: bool = False,
      gcs_bucket_name: str | None = None,
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
        timeout_sec=20 * 60,
        dry_run=dry_run,
        test_result_label=self.test_result_label,
        **kwargs,
    )
    self.gcs_bucket_name = gcs_bucket_name

  def _get_gpu_helm_install_flags_value(self) -> str:
    """Constructs the value for the HELM_INSTALL_FLAGS environment variable."""
    values_file_path = self._get_values_file()
    with open(values_file_path, 'r') as f:
      values = yaml.safe_load(f)

    hc_image_tag = values['health_checks']['gpu_healthcheck']['env'][
        'HC_IMAGE_TAG'
    ]

    helm_install_flags = [
        f'--set health_check.image.tag={self.machine_type}_{hc_image_tag}',
        f'--set health_check.env.INSTANCE_TYPE={self.machine_type}',
    ]

    if self.gcs_bucket_name:
      helm_install_flags.append(
          f'--set health_check.env.GCS_BUCKET_NAME={self.gcs_bucket_name}'
      )

    # The flags string needs to be quoted properly if it contains spaces.
    return f"\"{' '.join(helm_install_flags)}\""

  def _get_helm_env_vars(self):
    """Overrides base method to add GPU-specific helm install flags."""
    # Get the base helm env vars from the parent class (e.g., N_NODES).
    additional_helm_env_vars = super()._get_helm_env_vars()
    if additional_helm_env_vars is None:
      additional_helm_env_vars = {}

    additional_helm_env_vars[
        f'health_checks.{self.name}_healthcheck.env.HELM_INSTALL_FLAGS'
    ] = self._get_gpu_helm_install_flags_value()

    return additional_helm_env_vars


class SlurmGpuCheck(slurm_check.SlurmCheck):
  """A check that runs a DCGM test on a Slurm cluster."""

  _SUPPORTED_MACHINE_TYPES = common.SUPPORTED_MACHINE_TYPES

  _check_flag = 'dcgm'

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
