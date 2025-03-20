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

"""Library for parsing a golden qualified configuration from JSON to NodeConfig."""

import json
from typing import Any

from google.cloud import storage

import config
import dependency_version_parser


def get_golden_configs(
    dependency_parsers: list[dependency_version_parser.DependencyVersionParser],
    machine_type: str,
) -> list[config.NodeConfig]:
  """Returns the golden configurations for all supported machine types."""
  qualified_configurations = _get_qualified_configurations(
      bucket_name='aiinfra-qualified-cluster-configurations',
      blob_name=f'{machine_type}/qualified_versions.json',
  )
  return [
      _parse_qualified_configuration(
          config,
          [parser.name for parser in dependency_parsers],
      )
      for config in _filter_qualified_configurations_for_status(
          qualified_configuration_versions=qualified_configurations['version'],
          status='ACTIVE',
      )
  ]


def _parse_qualified_configuration(
    qualified_configuration: dict[str, str | list[str]],
    dependency_names: list[str],
) -> config.NodeConfig:
  """Parses a qualified configuration into a NodeConfig proto."""
  dependency_configs = {}
  for dependency_name in dependency_names:
    dependency_version = qualified_configuration.get(dependency_name, None)
    if isinstance(dependency_version, str):
      dependency_configs[dependency_name] = config.DependencyConfig(
          name=dependency_name,
          version=str(dependency_version),
      )
    elif isinstance(dependency_version, list):
      deps = {
          pair.split('=')[0]: pair.split('=')[1] for pair in dependency_version
      }
      dependency_configs[dependency_name] = config.DependencyConfig(
          name=dependency_name,
          version='',
          config_settings=deps,
      )
  return config.NodeConfig(
      name='GoldenConfig',
      dependencies=dependency_configs,
  )


def _filter_qualified_configurations_for_status(
    qualified_configuration_versions: list[dict[str, str | list[str]]],
    status: str,
) -> list[dict[str, Any]]:
  """Filters qualified configurations for a given machine type."""
  return [
      config_version
      for config_version in qualified_configuration_versions
      if config_version['status'] == status
  ]


def _get_qualified_configurations(bucket_name, blob_name) -> dict[str, Any]:
  """Reads data from a Google Cloud Storage bucket.

  Args:
      bucket_name: The name of the GCS bucket.
      blob_name: The path to the file within the bucket.

  Returns:
      The contents of the file as a string.
  """

  # Authenticate using the service account key file
  storage_client = storage.Client.create_anonymous_client()

  # Get the bucket and blob objects
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(blob_name)

  return json.loads(blob.download_as_text())
