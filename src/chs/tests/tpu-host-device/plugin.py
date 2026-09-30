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

"""Host to TPU transfer bandwidth check."""

from chs.testkit import SlicePlugin


class Plugin(SlicePlugin):
    """Measures transfer bandwidth between host memory and the TPU.

    This is the path training data travels, so a degraded link starves the
    accelerators even when every chip is healthy.
    """

    name = "tpu-host-device"
    category = "Host"
    description = "Host to device bandwidth check"
    long_description = (
        "Measures the pure transfer latency and maximum bandwidth capability "
        "by repeatedly bouncing bulk data payloads tightly between the Host "
        "CPU and the TPU."
    )
    application = "tpu-host-device"
    default_workloads = ("check",)
    workload_in_experiment_name = False
    benchmark_config = "host_device.yaml"
