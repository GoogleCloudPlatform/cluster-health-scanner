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

"""TPU collective communication bandwidth check."""

from chs.testkit import SliceTestPlugin
from chs.testkit.tpu import apply_collective_mesh_parameters


class Plugin(SliceTestPlugin):
    name = "tpu-collectives"
    category = "Network & Collectives"
    description = "Cluster collectives testing"
    long_description = (
        "Times distributed cross-chip operations (like All-Reduce and "
        "All-Gather) to measure the real-world bandwidth limits of the "
        "interconnect infrastructure."
    )
    application = "tpu-collectives"
    default_workloads = ("all_gather", "psum", "all_to_all")
    # Only reached if a caller clears the file chosen per workload below.
    benchmark_config = "collectives_all_gather.yaml"

    def prepare_variables(self, variables, workload, request):
        del request
        variables["benchmarks_yaml_file"] = f"collectives_{workload}.yaml"

    def finalize_context(self, context, workload, request):
        """Fills in the mesh shape, which depends on the slice assigned.

        Runs after the context is built because the topology it needs comes
        from the machine configuration and from the slice this experiment
        landed on, neither of which is known earlier.
        """
        del workload, request
        apply_collective_mesh_parameters(context["variables"])
