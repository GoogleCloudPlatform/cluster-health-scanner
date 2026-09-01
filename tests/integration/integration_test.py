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

from tests.integration.mocks.output_generator import RambleOutputGenerator
from tests.integration.test_utils import integration_test_util as test_utils


def test_standard_suite(active_cluster):
    cluster_config_path = active_cluster["_config_path_str"]
    cluster_name = active_cluster.get("cluster_name", "chs-cluster")

    output_gen = RambleOutputGenerator(cluster_config_path)
    nodes_str = test_utils.get_nodepools_csv(active_cluster)

    with test_utils.mock_chs_environment(cluster_name, output_gen):
        exit_code, stdout, stderr = test_utils.run_suite_and_capture(
            "standard", nodes_str
        )
        report = test_utils.validate_integration_run(
            active_cluster, exit_code, stdout, stderr
        )
        assert report.passed, (
            f"Integration validation failed for {cluster_config_path}:\n"
            f"{report.format_summary()}"
        )
