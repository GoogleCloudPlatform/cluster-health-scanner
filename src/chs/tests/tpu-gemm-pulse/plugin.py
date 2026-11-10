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

"""TPU multi-cycle idle-to-load GEMM pulse and checksum check."""

from chs.testkit import SlicePlugin


class Plugin(SlicePlugin):
    """Pulses each TPU between idle pauses and chained GEMM load cycles.

    Repeated idle-to-full load steps exercise power delivery, sequencers, and
    app-links while bit-exact per-window 64-bit checksums catch silent data
    corruption across local devices.
    """

    name = "tpu-gemm-pulse"
    category = "Compute"
    description = "Multi-cycle GEMM load-step pulse and determinism check"
    long_description = (
        "Cycles the TPU through repeated idle-to-full chained GEMM load steps "
        "to stress power delivery and load transitions while verifying "
        "bit-exact 64-bit checksums across local devices on each host."
    )
    application = "tpu-gemm-pulse"
    default_workloads = ("check",)
    workload_in_experiment_name = False
    benchmark_config = "gemm_pulse_16384_bf16.yaml"
