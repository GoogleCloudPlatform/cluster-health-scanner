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
from chs.testkit.context import load_benchmark_config


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
    benchmark_config = "host_to_device.yaml"
    benchmark_config_d2h = "device_to_host.yaml"

    def finalize_context(self, context, workload, request):
        variables = context["variables"]
        filename = variables.get("benchmarks_yaml_d2h_file")
        if not filename:
            filename = self.benchmark_config_d2h
            variables["benchmarks_yaml_d2h_file"] = filename

        content = load_benchmark_config(filename, self.benchmarks_dir)
        variables["benchmarks_yaml_d2h_escaped"] = (
            content.replace("\\", "\\\\")
            .replace("'", "'\\''")
            .replace("\n", "\\n")
        )
