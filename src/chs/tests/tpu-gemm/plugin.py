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

"""TPU matrix multiplication throughput check."""

from chs.testkit import SlicePlugin


class Plugin(SlicePlugin):
    """Measures sustained matrix multiplication throughput on each TPU chip.

    A chip that is throttling or clocked low reaches a lower throughput while
    still reporting itself healthy.
    """

    name = "tpu-gemm"
    category = "Compute"
    description = "General Matrix Multiply benchmark"
    long_description = (
        "Performs generalized matrix multiplications (GEMM) through the TPU to "
        "calculate exactly how many peak TFLOP/s the compute pipeline can "
        "sustain."
    )
    application = "tpu-gemm"
    default_workloads = ("check",)
    benchmark_config = "gemm_16384_bf16.yaml"
