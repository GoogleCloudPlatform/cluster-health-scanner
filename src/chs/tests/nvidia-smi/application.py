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


class NvidiaSmi(ExecutableApplication):
    """NVIDIA SMI application for cluster health scanner"""

    name = "nvidia-smi"
    maintainers("sarthakag", "simrankaurb")
    tags("diagnostics", "gpu")

    executable(
        "execute",
        template=[
            "nvidia-smi --query-gpu=index,name,driver_version,pstate,pci.bus_id,serial,uuid,power.draw,power.limit,clocks.gr,clocks.mem --format=csv,noheader,nounits",
            'echo "GPU Count: $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"',
            '(nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true)',
        ],
        use_mpi=False,
    )
    workload("local", executables=["execute"])

    figure_of_merit_context(
        "gpu",
        regex=r"^(?P<gpu_index>\d+),",
        output_format="GPU {gpu_index}",
    )

    figure_of_merit(
        "GPU Name",
        fom_regex=r"^\d+,\s*(?P<gpu_name>[^,]+),",
        group_name="gpu_name",
        units="",
        fom_type=FomType.INFO,
        contexts=["gpu"],
    )

    figure_of_merit(
        "Driver Version",
        fom_regex=r"^\d+,\s*[^,]+,\s*(?P<driver_version>[^,]+),",
        group_name="driver_version",
        units="",
        fom_type=FomType.INFO,
        contexts=["gpu"],
    )

    figure_of_merit(
        "Performance State",
        fom_regex=r"^\d+,\s*[^,]+,\s*[^,]+,\s*(?P<pstate>[^,]+),",
        group_name="pstate",
        units="",
        fom_type=FomType.INFO,
        contexts=["gpu"],
    )

    figure_of_merit(
        "PCI Bus ID",
        fom_regex=r"^\d+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(?P<pci_bus_id>[^,]+),",
        group_name="pci_bus_id",
        units="",
        fom_type=FomType.INFO,
        contexts=["gpu"],
    )

    figure_of_merit(
        "Serial Number",
        fom_regex=r"^\d+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(?P<serial>[^,]+),",
        group_name="serial",
        units="",
        fom_type=FomType.INFO,
        contexts=["gpu"],
    )

    figure_of_merit(
        "UUID",
        fom_regex=r"^\d+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(?P<uuid>[^,]+),",
        group_name="uuid",
        units="",
        fom_type=FomType.INFO,
        contexts=["gpu"],
    )

    figure_of_merit(
        "Power Draw",
        fom_regex=r"^\d+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(?P<power_draw>[^,]+),",
        group_name="power_draw",
        units="W",
        fom_type=FomType.MEASURE,
        contexts=["gpu"],
    )

    figure_of_merit(
        "Power Limit",
        fom_regex=r"^\d+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(?P<power_limit>[^,]+),",
        group_name="power_limit",
        units="W",
        fom_type=FomType.INFO,
        contexts=["gpu"],
    )

    figure_of_merit(
        "Graphics Clock",
        fom_regex=r"^\d+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(?P<graphics_clock>[^,]+),",
        group_name="graphics_clock",
        units="MHz",
        fom_type=FomType.INFO,
        contexts=["gpu"],
    )

    figure_of_merit(
        "Memory Clock",
        fom_regex=r"^\d+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(?P<mem_clock>[^,]+)",
        group_name="mem_clock",
        units="MHz",
        fom_type=FomType.INFO,
        contexts=["gpu"],
    )

    figure_of_merit(
        "GPU Count",
        fom_regex=r"GPU Count: (?P<gpu_count>\d+)",
        group_name="gpu_count",
        units="",
    )

    figure_of_merit(
        "CUDA Version",
        fom_regex=r"CUDA Version: (?P<cuda_version>[0-9.]+)",
        group_name="cuda_version",
        units="",
        fom_type=FomType.INFO,
    )

    success_criteria(
        "gpu_count_check",
        mode="fom_comparison",
        fom_name="GPU Count",
        formula="{value} == {gpus_per_node}",
    )
