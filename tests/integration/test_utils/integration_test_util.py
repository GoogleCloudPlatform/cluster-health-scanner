# Copyright 2026 Google LLC
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

import contextlib
import pytest

from chs.cli import cli
from tests.integration.mocks.gcp_mock import mock_gce_instances, mock_gcs_bucket
from tests.integration.mocks.ramble_mock import mock_ramble_workloads


def run_suite(suite_name: str, nodepools: str) -> None:
    cli(
        [
            "run",
            "--suite",
            suite_name,
            "--nodepools",
            nodepools,
            "-y",
        ],
        standalone_mode=False,
    )


@contextlib.contextmanager
def mock_chs_environment(cluster_name: str, output_gen):
    """Consolidated context manager mocking all external network boundaries
    (GCP, GCS, Ramble) required to run CHS suites.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock_gce_instances(cluster_name))
        stack.enter_context(mock_gcs_bucket())
        stack.enter_context(mock_ramble_workloads(output_gen))
        yield


def get_nodepools_csv(cluster_config: dict) -> str:
    """Extracts a comma-separated string of nodepools from cluster config."""
    nodepools = [
        pool.get("name")
        for pool in cluster_config.get("nodepools", [])
        if isinstance(pool, dict)
    ]
    if not nodepools:
        config_path = cluster_config.get("_config_path_str", "Unknown config")
        pytest.skip(f"No nodepools defined in config: {config_path}")

    return ",".join(nodepools)
