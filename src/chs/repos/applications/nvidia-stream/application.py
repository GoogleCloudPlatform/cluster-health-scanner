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


from ramble.appkit import *


class NvidiaStream(ExecutableApplication):
    """NVIDIA Stream Benchmark application

    This application runs the NVIDIA stream benchmark typically utilizing:
    stream-gpu-test.sh
    """

    name = "nvidia-stream"
    maintainers("bytetwin")

    tags("micro-benchmark", "gpu")

    executable(
        "execute",
        "./stream-gpu-test.sh "
        "--n {n_elements} "
        "--gpu-affinity {gpu_affinity} "
        "--cpu-affinity {cpu_affinity} "
        "--mem-affinity {mem_affinity}",
        use_mpi=True,
    )

    workload("standard", executables=["execute"])

    workload_variable(
        "n_elements",
        default="150000",
        description="Number of elements",
        workloads=["standard"],
    )

    workload_variable(
        "gpu_affinity",
        default="",
        description="GPU affinity mapping",
        workloads=["standard"],
    )

    workload_variable(
        "cpu_affinity",
        default="",
        description="CPU affinity mapping",
        workloads=["standard"],
    )

    workload_variable(
        "mem_affinity",
        default="",
        description="Memory affinity mapping",
        workloads=["standard"],
    )

    device_regex = r"(?:\[[0-9]+,)?(?P<device_id>[0-9]+)(?:\]<stdout>)?:\s*Device\s+[0-9]+:\s+.*?=\s+(?P<peak_bw>[0-9]+\.[0-9]+)\s+GB/s\s+PEAK"

    function_regex = r"(?:\[[0-9]+,)?(?P<device_id>[0-9]+)(?:\]<stdout>)?:\s*(?P<function>Copy|Scale|Add|Triad):\s+(?P<rate>[0-9]+\.[0-9]+)\s+(?P<avg_time>[0-9]+\.[0-9]+)\s+(?P<min_time>[0-9]+\.[0-9]+)\s+(?P<max_time>[0-9]+\.[0-9]+)"

    figure_of_merit_context(
        "function",
        regex=function_regex,
        output_format="Device {device_id} {function}",
    )

    figure_of_merit(
        "Device id",
        fom_regex=function_regex,
        group_name="device_id",
        units="",
        contexts=["function"],
    )

    figure_of_merit(
        "Peak Bandwidth",
        fom_regex=device_regex,
        group_name="peak_bw",
        units="GB/s",
        contexts=["function"],
    )

    figure_of_merit(
        "Achieved Bandwidth",
        fom_regex=function_regex,
        group_name="rate",
        units="MB/s",
        contexts=["function"],
    )

    figure_of_merit(
        "Avg time",
        fom_regex=function_regex,
        group_name="avg_time",
        units="s",
        contexts=["function"],
    )

    figure_of_merit(
        "Min time",
        fom_regex=function_regex,
        group_name="min_time",
        units="s",
        contexts=["function"],
    )

    figure_of_merit(
        "Max time",
        fom_regex=function_regex,
        group_name="max_time",
        units="s",
        contexts=["function"],
    )
