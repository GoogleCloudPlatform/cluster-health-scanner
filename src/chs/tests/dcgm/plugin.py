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

"""NVIDIA DCGM hardware diagnostics."""

from chs.testkit import SingleNodePlugin


class Plugin(SingleNodePlugin):
    name = "dcgm"
    category = "GPU Diagnostics"
    description = "Deep hardware diagnostics"
    long_description = (
        "Runs DCGM diagnostic suites to stress-test PCIe, memory, and compute "
        "engines, measuring hardware stability and identifying faults, thermal "
        "issues, or power throttling."
    )
    application = "nvidia-dcgm"
    default_workloads = ("diag",)
    workload_in_experiment_name = False
    experiment_suffix = "-level-{diag_level}"
    extra_context = {"matrix": ["node", "diag_level"]}
