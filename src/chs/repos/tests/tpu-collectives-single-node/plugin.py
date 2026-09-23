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

"""TPU collective communication check confined to a single host.

This test is reachable from the engine but is not listed in the test catalog,
so no suite can select it today. It is carried over unchanged; whether to list
it or drop it is decided separately.
"""

from chs.testkit import SliceTestPlugin
from chs.testkit.tpu import apply_collective_mesh_parameters


class Plugin(SliceTestPlugin):
    name = "tpu-collectives-single-node"
    category = "Network & Collectives"
    description = "Single host TPU collectives check"
    long_description = (
        "Runs the TPU collectives benchmark across the chips of one host to "
        "measure intra-host bandwidth (GB/s), separating a bad link inside a "
        "host from a bad link between hosts."
    )
    application = "tpu-collectives"
    config_key = "tpu-collectives-single-node"
    experiment_label = "tpu-collectives-single-node"
    default_workloads = ("all_gather", "psum", "all_to_all")
    benchmark_config = "collectives_all_gather.yaml"

    def prepare_variables(self, variables, workload, request):
        del request
        variables["benchmarks_yaml_file"] = f"collectives_{workload}.yaml"
        variables["execution_topology"] = "2x2x1"

    def finalize_context(self, context, workload, request):
        del workload, request
        apply_collective_mesh_parameters(context["variables"])
