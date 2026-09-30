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

"""Sustained load on the TPU interconnect."""

from chs.testkit import SlicePlugin


class Plugin(SlicePlugin):
    name = "tpu-network-stress"
    category = "Network & Collectives"
    description = (
        "Intense point-to-point network stress test for TPU interconnects"
    )
    long_description = (
        "Continuously swaps massive random payloads across the TPU high-speed "
        "interconnect (ICI) links to measure peak sustained network bandwidth "
        "and validate health against bit-flips."
    )
    application = "tpu-network-stress"
    default_workloads = ("stress",)
    workload_in_experiment_name = False
