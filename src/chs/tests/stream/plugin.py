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

"""GPU memory bandwidth check."""

from chs.testkit import SingleNodePlugin


class Plugin(SingleNodePlugin):
    """Measures memory bandwidth on each GPU with the STREAM benchmark.

    One rank is pinned to each GPU so that the number reported is that GPU's
    own bandwidth rather than an average across the node.
    """

    name = "stream"
    category = "Memory"
    description = "Memory bandwidth benchmark"
    long_description = (
        "Measures sustainable GPU memory bandwidth (GB/s) using the STREAM "
        "benchmark to verify memory subsystem throughput."
    )
    application = "stream"
    default_workloads = ("standard",)
    workload_in_experiment_name = False
    # Slurm needs to be told to give each rank its own GPU; Kubernetes gets the
    # same effect from the pod's device request.
    extra_sbatch_headers = {"--gpu-bind": "single:1"}
