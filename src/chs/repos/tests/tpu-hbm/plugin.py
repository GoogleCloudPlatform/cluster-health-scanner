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

"""TPU high bandwidth memory check."""

from chs.testkit import SliceTestPlugin


class Plugin(SliceTestPlugin):
    name = "tpu-hbm"
    category = "Memory"
    description = "High Bandwidth Memory test"
    long_description = (
        "Generates performance traces to measure the absolute architectural "
        "read and write bandwidth limits of the HBM."
    )
    application = "tpu-hbm"
    default_workloads = ("check",)
    workload_in_experiment_name = False
    benchmark_config = "hbm_copy_bf16.yaml"
