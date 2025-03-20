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


_NVIDIA_SMI_PATH = '../.././home/kubernetes/bin/nvidia/bin/nvidia-smi'

DEPENDENCY_PARSERS = [
    dependency_version_parser.DependencyVersionParser(
        name='cosVersion',
        cmd=([
            'cat',
            '/etc/os-release',
            '|',
            'grep',
            'BUILD_ID',
            '|',
            'egrep',
            '-o',
            r"'[0-9\\.]+'",
        ]),
        parse_version_fn=_parse_generic_version,
    ),
    dependency_version_parser.DependencyVersionParser(
        name='gpuDriverVersion',
        cmd=([
            _NVIDIA_SMI_PATH,
            '--query-gpu=driver_version',
            '--format=csv,noheader',
        ]),
        parse_version_fn=_parse_driver_version,
    ),
    dependency_version_parser.DependencyVersionParser(
        name='cudaVersion',
        cmd=([
            _NVIDIA_SMI_PATH,
            '|',
            'sed',
            '-n',
            '"3p"',
            '|',
            'sed',
            r'"s/.*CUDA Version: \+\(.*\)|.*/\1/"',
        ]),
        parse_version_fn=_parse_generic_version,
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
    workload_container: str | None = None,
) -> list[dependency_version_parser.DependencyVersionParser]:
  """Returns the dynamic dependency parsers for a given node.

  Dynamic parsers are parsers that require specific context to be fetched.
  For example, NCCL configs are fetched from the workload container, while
  NCCL plugin version requires a node name and zone.

  Args:
    node_name: The name of the node.
    zone: The zone of the node.
    workload_container: The name of the workload container to fetch configs
      from. If not specified, NCCL configs will not be fetched.

  Returns:
    A list of dynamic dependency parsers.
  """
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
  if workload_container:
    parsers.extend([
        dependency_version_parser.DependencyVersionParser(
            name='ncclConfigs',
            cmd=([
                'crictl',
                'pods',
                '--latest',
                '|',
                'awk',
                "'NR>1 {print $6}'",
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
