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

"""Sustained memory load on the TPU."""

from chs.testkit import SlicePlugin


class Plugin(SlicePlugin):
    """Holds the chips' memory under continuous load.

    Marginal memory often passes a single bandwidth measurement and only shows
    up as errors or degradation after being driven for a while.
    """

    name = "tpu-hbm-stress"
    category = "Memory"
    description = (
        "Heavy synthetic load testing High Bandwidth Memory (HBM) stability"
    )
    long_description = (
        "Checks the bottlenecks of the High Bandwidth Memory (HBM) using rapid "
        "integer operations to strictly measure maximum memory bandwidth and "
        "validate data integrity."
    )
    application = "tpu-hbm-stress"
    default_workloads = ("stress",)
    workload_in_experiment_name = False
