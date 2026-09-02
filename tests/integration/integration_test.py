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

import pytest
import os
import json

from tests.integration.test_utils import integration_test_util as test_utils
from chs.utils.app_data import ChsAppData
from tests.integration.mocks.output_generator import RambleOutputGenerator


def temp_validation():
    # Verify output from the run
    app_data = ChsAppData()
    latest_run_id = app_data.get_latest_run_id()
    workspace_name = app_data.resolve_run_id(latest_run_id)
    workspace_path = os.path.join(app_data.get_workspaces_dir(), workspace_name)

    boh_file = os.path.join(workspace_path, "bill_of_health.json")

    assert os.path.exists(
        boh_file
    ), f"Bill of Health file not found at: {boh_file}"

    with open(boh_file, "r") as f:
        data = json.load(f)
        if "total_nodes" not in data and "workloads" not in data:
            pass


@pytest.mark.tpu
def test_standard_suite(active_cluster):
    cluster_config_path = active_cluster["_config_path_str"]
    cluster_name = active_cluster.get("cluster_name", "chs-tpu-v6e-cluster")

    output_gen = RambleOutputGenerator(cluster_config_path)
    nodes_str = test_utils.get_nodepools_csv(active_cluster)

    with test_utils.mock_chs_environment(cluster_name, output_gen):
        try:
            test_utils.run_suite("standard", nodes_str)
        except SystemExit as e:
            if e.code != 0:
                pytest.fail(f"CHS CLI run failed with exit code: {e.code}")

        temp_validation()


@pytest.mark.gpu
def test_standard_suite_gpu(active_cluster):
    cluster_config_path = active_cluster["_config_path_str"]
    cluster_name = active_cluster.get("cluster_name", "chs-gpu-cluster")

    output_gen = RambleOutputGenerator(cluster_config_path)
    nodes_str = test_utils.get_nodepools_csv(active_cluster)

    with test_utils.mock_chs_environment(cluster_name, output_gen):
        try:
            test_utils.run_suite("standard", nodes_str)
        except SystemExit as e:
            if e.code != 0:
                pytest.fail(f"CHS CLI run failed with exit code: {e.code}")

        temp_validation()
