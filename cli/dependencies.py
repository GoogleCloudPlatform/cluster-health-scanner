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

"""Common dependencies for configcheck."""

import re
import subprocess

import config
import dependency_version_parser
import local_dependency_version_parser


def _parse_driver_version(name, cmd_result: str) -> config.DependencyConfig:
  """Parses the driver version from the command result."""
  version_set = set(version for version in cmd_result.strip().split())
  if len(version_set) != 1:
    print(f'Expected exactly one driver version, got: {version_set}')
    raise ValueError(f'Expected exactly one driver version, got: {version_set}')
  return config.DependencyConfig(
      name=name,
      version=version_set.pop(),
  )


def _parse_generic_version(name, cmd_result: str) -> config.DependencyConfig:
  """Parses a generic X.Y....Z version from the command result."""
  return config.DependencyConfig(
      name=name,
      version=re.sub(r'[^A-Za-z0-9\.]+', '', cmd_result),
  )


def _parse_nccl_configs(name, cmd_result: str) -> config.DependencyConfig:
  """Parses the nccl configs from the command result."""
  configs = {
      line.split('=')[0]: line.split('=')[1]
      for line in cmd_result.strip().split('\n')
  }
  return config.DependencyConfig(
      name=name,
      version='Abridged. See diffs for details.',
      config_settings=configs,
  )


def _parse_nccl_version(name, cmd_result: str) -> config.DependencyConfig:
  """Parses the nccl version from the command result."""
  match = re.search(r'libnccl\.so\.([0-9\.]+)', cmd_result)
  if match:
    return config.DependencyConfig(name=name, version=match.group(1))
  raise ValueError(f'Failed to parse nccl version from: {cmd_result}')


def _build_nvidia_smi_command(
    args: list[str], nvidia_smi_path: str
) -> list[str]:
  """Builds the nvidia-smi command with the given arguments and path."""
  return [nvidia_smi_path] + args


def _build_cos_version_command(grep_value: str) -> list[str]:
  """Builds the cos version command with the given grep value."""
  return [
      'cat',
      _OS_RELEASE_FILE,
      '|',
      'grep',
      grep_value,
      '|',
      'egrep',
      '-o',
      r"'[0-9\\.]+'",
  ]


def _build_base_dependency_parsers(
    nvidia_smi_path: str,
    cos_build_id_grep: str,
) -> list[dependency_version_parser.DependencyVersionParser]:
  return [
      dependency_version_parser.DependencyVersionParser(
          name='cosVersion',
          cmd=_build_cos_version_command(cos_build_id_grep),
          parse_version_fn=_parse_generic_version,
      ),
      dependency_version_parser.DependencyVersionParser(
          name='cudaVersion',
          cmd=_build_nvidia_smi_command(
              _CUDA_VERSION_SED_ARGS,
              nvidia_smi_path,
          ),
          parse_version_fn=_parse_generic_version,
      ),
  ]


_OS_RELEASE_FILE = '/etc/os-release'
_COS_BUILD_ID_GREP = 'BUILD_ID'
_COS_VERSION_ID_GREP = '"VERSION_ID"'
_NVIDIA_SMI_PATH_GKE = '../.././home/kubernetes/bin/nvidia/bin/nvidia-smi'
_NVIDIA_SMI_PATH_SLURM = 'nvidia-smi'
_CUDA_VERSION_SED_ARGS = [
    '|',
    'sed',
    '-n',
    '"3p"',
    '|',
    'sed',
    r'"s/.*CUDA Version: \+\(.*\)|.*/\1/"',
]


BASE_DEPENDENCY_PARSERS = [
    dependency_version_parser.DependencyVersionParser(
        name='cosVersion',
        cmd=_build_cos_version_command(_COS_BUILD_ID_GREP),
        parse_version_fn=_parse_generic_version,
    ),
    dependency_version_parser.DependencyVersionParser(
        name='cudaVersion',
        cmd=_build_nvidia_smi_command(
            _CUDA_VERSION_SED_ARGS,
            '{nvidia_smi_path}',
        ),
        parse_version_fn=_parse_generic_version,
    ),
]


GKE_DEPENDENCY_PARSERS = _build_base_dependency_parsers(
    _NVIDIA_SMI_PATH_GKE, _COS_BUILD_ID_GREP
) + [
    dependency_version_parser.DependencyVersionParser(
        name='gpuDriverVersion',
        cmd=_build_nvidia_smi_command(
            ['--query-gpu=driver_version', '--format=csv,noheader'],
            _NVIDIA_SMI_PATH_GKE,
        ),
        parse_version_fn=_parse_driver_version,
    ),
    dependency_version_parser.DependencyVersionParser(
        name='ncclVersion',
        cmd=([
            'ldconfig',
            '-v',
            '|',
            'grep',
            '"libnccl.so"',
            '|',
            'tail',
            '-n1',
            '|',
            'sed',
            '-r',
            r'"s/^.*\.so\.//"',
        ]),
        parse_version_fn=_parse_generic_version,
    ),
]


SLURM_DEPENDENCY_PARSERS = _build_base_dependency_parsers(
    _NVIDIA_SMI_PATH_SLURM, _COS_VERSION_ID_GREP
) + [
    dependency_version_parser.DependencyVersionParser(
        name='gpuDriverVersion',
        cmd=_build_nvidia_smi_command(
            ['--query-gpu=driver_version', '--format=csv,noheader'],
            _NVIDIA_SMI_PATH_SLURM,
        ),
        parse_version_fn=_parse_driver_version,
    ),
    dependency_version_parser.DependencyVersionParser(
        name='ncclVersion',
        cmd=([
            'readlink',
            '-f',
            '/var/lib/tcpx/lib64/libnccl.so',
        ]),
        parse_version_fn=_parse_nccl_version,
    ),
]


def _parse_nccl_plugin_version(name, file_path: str) -> config.DependencyConfig:
  """Parses the nccl plugin version from the command result."""
  plugin_version = subprocess.run(
      ' '.join(['nm', '-gD', file_path, '|', 'grep', 'ncclNetPlugin_v']),
      shell=True,
      check=True,
      capture_output=True,
      text=True,
  ).stdout
  plugin_version_match = re.search(r'\b([a-zA-Z0-9_]+)$', plugin_version)
  if not plugin_version_match:
    raise ValueError(
        f'Failed to parse nccl plugin version from: {plugin_version}'
    )

  plugin_version = plugin_version_match.group(1)
  return config.DependencyConfig(
      name=name,
      version=plugin_version,
  )


def get_dynamic_dependency_parsers(
    node_name: str,
    zone: str,
    pod_name: str | None = None,
    workload_container: str | None = None,
) -> list[dependency_version_parser.DependencyVersionParser]:
  """Returns the dynamic dependency parsers for a given node.

  Dynamic parsers are parsers that require specific context to be fetched.
  For example, NCCL configs are fetched from the workload container, while
  NCCL plugin version requires a node name and zone.

  Args:
    node_name: The name of the node.
    zone: The zone of the node.
    pod_name: The name of the pod.
    workload_container: The name of the workload container to fetch configs
      from. If not specified, NCCL configs will not be fetched.

  Returns:
    A list of dynamic dependency parsers.
  """
  # 
  parsers = [
      local_dependency_version_parser.LocalDependencyVersionParser(
          dep_name='ncclPluginVersion',
          node_name=node_name,
          zone=zone,
          remote_file_path='/home/kubernetes/bin/nvidia/lib64/libnccl-net.so',
          parse_version_fn=lambda name, file_path: _parse_nccl_plugin_version(
              name,
              f'{local_dependency_version_parser.LOCAL_FILE_PATH}/{node_name}/libnccl-net.so',
          ),
      ),
      dependency_version_parser.DependencyVersionParser(
          name='Workload Container',
          cmd=([
              'echo',
              workload_container if workload_container else 'None Found',
          ]),
      ),
  ]
  if workload_container and pod_name:
    parsers.extend([
        dependency_version_parser.DependencyVersionParser(
            name='ncclConfigs',
            cmd=([
                'echo',
                '-n',
                pod_name,
                '|',
                'xargs',
                '-I',
                '{}',
                'kubectl',
                'exec',
                '{}',
                '-c',
                f'{workload_container}',
                '--',
                'env',
                '|',
                'grep',
                '-E',
                '"NCCL|LD_LIBRARY"',
            ]),
            parse_version_fn=_parse_nccl_configs,
        ),
    ])
  else:
    parsers.append(
        dependency_version_parser.DependencyVersionParser(
            name='ncclConfigs',
            cmd=([
                'echo',
                'Error: Workload Container required',
            ]),
        )
    )
  return parsers


def get_slurm_dynamic_dependency_parsers(
    node_name: str,
    zone: str,
) -> list[dependency_version_parser.DependencyVersionParser]:
  """Returns the dynamic dependency parsers for a given slurm node.

  Args:
    node_name: The name of the node.
    zone: The zone of the node.

  Returns:
    A list of dynamic dependency parsers.
  """
  # 
  parsers = [
      local_dependency_version_parser.LocalDependencyVersionParser(
          dep_name='ncclPluginVersion',
          node_name=node_name,
          zone=zone,
          remote_file_path='/var/lib/tcpx/lib64/libnccl-net.so',
          parse_version_fn=lambda name, file_path: _parse_nccl_plugin_version(
              name,
              f'{local_dependency_version_parser.LOCAL_FILE_PATH}/{node_name}/libnccl-net.so',
          ),
      ),
      # 
  ]
  return parsers
